#!/usr/bin/env python3
"""
Run track.py on all .avi/.tif files in a directory tree, N files at a time.

Each file runs in its own subprocess. Output is streamed to per-file log files
in a 'logs/' subfolder next to this script, so you can tail them independently.

Usage:
    python run_fasttrack_parallel.py <directory> -c <config.yaml>
    python run_fasttrack_parallel.py ./data -c configs/IR_medium.yaml -j 4

After all jobs finish, pumping_summary.csv and pumping_events.csv files are
collected and three per-particle statistics are computed:

  active_rate_hz         — AMPD pumps during active windows / active_duration_s
  entry_rate_per_min     — quiescent bouts / track_duration_s × 60
  mean_bout_duration_s   — total_quiescent_time_s / n_quiescent_bouts

Six Poisson GLMMs (via R/lme4) are fitted across two balanced comparisons:
  food type    (fed animals, OP50 vs JUb39)  : rate, entry rate, exit rate
  feeding state (OP50 animals, fed vs off-food): rate, entry rate, exit rate

Supply a metadata CSV to label conditions:
    python run_fasttrack_parallel.py ./data -c configs/... --metadata meta.csv

The metadata CSV must have a 'folder_name' column plus: genotype, food, condition.

Notes:
    - Default workers: 4.
    - Monitor individual jobs with: tail -f logs/<name>.log
    - Use --no-summary to skip batch collection.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

VIDEO_EXTENSIONS      = {".avi", ".mp4", ".tif", ".tiff"}
QUIESCENT_IPI_THRESH  = 1.0     # seconds — unambiguous pump-miss boundary (20 frames at 20 fps)
MIN_ACTIVE_DURATION_S = 5.0     # seconds — minimum active window to compute AMPD rate


# ── R script (passed paths via command-line args) ────────────────────────────
R_SCRIPT = r"""
missing_pkgs <- Filter(function(p) !requireNamespace(p, quietly = TRUE),
                       c("lme4", "lmerTest", "emmeans"))
if (length(missing_pkgs) > 0) {
  cat("ERROR: missing R packages:", paste(missing_pkgs, collapse = ", "), "\n")
  cat("Install them from an R console with:\n")
  cat('  install.packages(c("lme4", "lmerTest", "emmeans"), type = "binary")\n')
  quit(status = 1)
}
library(lme4)
library(emmeans)

args          <- commandArgs(trailingOnly = TRUE)
data_csv      <- args[1]
results_csv   <- args[2]
contrasts_csv <- args[3]

df            <- read.csv(data_csv, stringsAsFactors = FALSE)
df$video_path <- factor(df$video_path)
df$genotype   <- factor(df$genotype)
if ("WT" %in% levels(df$genotype)) {
  df$genotype <- relevel(df$genotype, ref = "WT")
}

dispersion_stat <- function(model) {
  sum(residuals(model, type = "pearson")^2) / df.residual(model)
}

# Returns list(coefs, contrasts, model) or NULL on failure.
fit_count_model <- function(fml_str, data, offset_col, model_id, emm_specs) {
  cat("Fitting", model_id, "...\n")
  if (nrow(data) == 0) { cat("  No data\n"); return(NULL) }

  data[["._offset_"]] <- log(data[[offset_col]])
  data <- data[is.finite(data[["._offset_"]]), ]
  if (nrow(data) == 0) { cat("  No finite offsets\n"); return(NULL) }

  fml <- as.formula(paste0(fml_str, " + offset(._offset_)"))

  m <- tryCatch(
    glmer(fml, data = data, family = poisson,
          control = glmerControl(optimizer = "bobyqa",
                                 optCtrl   = list(maxfun = 2e5))),
    error = function(e) { cat("  Poisson error:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(m)) return(NULL)

  disp     <- dispersion_stat(m)
  fam_used <- "poisson"
  cat("  dispersion:", round(disp, 2), "\n")

  if (disp > 2) {
    cat("  overdispersed — trying negative binomial\n")
    m_nb <- tryCatch(
      glmer.nb(fml, data = data,
               control = glmerControl(optimizer = "bobyqa",
                                      optCtrl   = list(maxfun = 2e5))),
      error = function(e) { cat("  NB error:", conditionMessage(e), "\n"); NULL }
    )
    if (!is.null(m_nb)) { m <- m_nb; fam_used <- "negbinom" }
  }

  # Fixed effects table
  cf            <- as.data.frame(coef(summary(m)))
  names(cf)     <- c("estimate", "std_error", "z_value", "p_value")
  cf$term       <- rownames(cf)
  cf$model_id   <- model_id
  cf$family     <- fam_used
  cf$dispersion <- round(disp, 3)
  rownames(cf)  <- NULL
  coefs <- cf[, c("model_id", "family", "dispersion", "term",
                  "estimate", "std_error", "z_value", "p_value")]

  # Post-hoc pairwise contrasts via emmeans (Tukey adjustment)
  contrasts_df <- tryCatch({
    emm   <- emmeans(m, specs = as.formula(emm_specs))
    contr <- as.data.frame(pairs(emm, adjust = "tukey"))
    contr$model_id <- model_id
    contr$family   <- fam_used
    # Rename columns to consistent names across model types
    names(contr)[names(contr) == "z.ratio"] <- "z_value"
    names(contr)[names(contr) == "t.ratio"] <- "z_value"
    names(contr)[names(contr) == "p.value"] <- "p_value"
    names(contr)[names(contr) == "SE"]      <- "std_error"
    contr
  }, error = function(e) {
    cat("  emmeans failed:", conditionMessage(e), "\n")
    NULL
  })

  list(coefs = coefs, contrasts = contrasts_df)
}

coef_list     <- list()
contrast_list <- list()
has_cols <- function(df, ...) all(c(...) %in% names(df))

# ── Comparison 1: food type (fed animals, OP50 vs JUb39) ────────────────────
if (has_cols(df, "food", "condition",
             "n_pumps_scipy_active", "active_duration_s",
             "n_quiescent_bouts", "track_duration_s",
             "total_quiescent_time_s")) {

  d1      <- subset(df, condition == "fed")
  d1$food <- relevel(factor(d1$food), ref = "OP50")
  fml1    <- "n_resp ~ genotype * food + (1 | video_path)"
  specs1  <- "~ genotype * food"

  d1r        <- subset(d1, active_duration_s >= 5)
  d1r$n_resp <- d1r$n_pumps_scipy_active
  r <- fit_count_model(fml1, d1r, "active_duration_s",       "food_rate",  specs1)
  if (!is.null(r)) { coef_list[["food_rate"]]  <- r$coefs; contrast_list[["food_rate"]]  <- r$contrasts }

  d1e        <- d1
  d1e$n_resp <- d1e$n_quiescent_bouts
  r <- fit_count_model(fml1, d1e, "track_duration_s",         "food_entry", specs1)
  if (!is.null(r)) { coef_list[["food_entry"]] <- r$coefs; contrast_list[["food_entry"]] <- r$contrasts }

  d1x        <- subset(d1, total_quiescent_time_s > 0)
  d1x$n_resp <- d1x$n_quiescent_bouts
  r <- fit_count_model(fml1, d1x, "total_quiescent_time_s",   "food_exit",  specs1)
  if (!is.null(r)) { coef_list[["food_exit"]]  <- r$coefs; contrast_list[["food_exit"]]  <- r$contrasts }
}

# ── Comparison 2: feeding state (OP50 animals, fed vs off-food) ─────────────
if (has_cols(df, "food", "condition",
             "n_pumps_scipy_active", "active_duration_s",
             "n_quiescent_bouts", "track_duration_s",
             "total_quiescent_time_s")) {

  d2           <- subset(df, food == "OP50")
  d2$condition <- relevel(factor(d2$condition), ref = "fed")
  fml2         <- "n_resp ~ genotype * condition + (1 | video_path)"
  specs2       <- "~ genotype * condition"

  d2r        <- subset(d2, active_duration_s >= 5)
  d2r$n_resp <- d2r$n_pumps_scipy_active
  r <- fit_count_model(fml2, d2r, "active_duration_s",       "feeding_rate",  specs2)
  if (!is.null(r)) { coef_list[["feeding_rate"]]  <- r$coefs; contrast_list[["feeding_rate"]]  <- r$contrasts }

  d2e        <- d2
  d2e$n_resp <- d2e$n_quiescent_bouts
  r <- fit_count_model(fml2, d2e, "track_duration_s",         "feeding_entry", specs2)
  if (!is.null(r)) { coef_list[["feeding_entry"]] <- r$coefs; contrast_list[["feeding_entry"]] <- r$contrasts }

  d2x        <- subset(d2, total_quiescent_time_s > 0)
  d2x$n_resp <- d2x$n_quiescent_bouts
  r <- fit_count_model(fml2, d2x, "total_quiescent_time_s",   "feeding_exit",  specs2)
  if (!is.null(r)) { coef_list[["feeding_exit"]]  <- r$coefs; contrast_list[["feeding_exit"]]  <- r$contrasts }
}

valid_coefs     <- Filter(Negate(is.null), coef_list)
valid_contrasts <- Filter(Negate(is.null), contrast_list)

if (length(valid_coefs) == 0) {
  cat("No models fitted.\n")
} else {
  write.csv(do.call(rbind, valid_coefs),     results_csv,   row.names = FALSE)
  cat("Fixed effects written to", results_csv, "\n")
}
if (length(valid_contrasts) > 0) {
  write.csv(do.call(rbind, valid_contrasts), contrasts_csv, row.names = FALSE)
  cat("Contrasts written to", contrasts_csv, "\n")
}
"""


# ── helpers ──────────────────────────────────────────────────────────────────

def run_one(video_path, config, log_dir):
    """Run track.py on a single file, streaming output to a log file."""
    log_path = Path(log_dir) / (Path(video_path).stem + ".log")
    with open(log_path, "w") as log:
        result = subprocess.run(
            [sys.executable, "track.py", "-f", video_path, "-c", config],
            stdout=log, stderr=subprocess.STDOUT, text=True
        )
    return video_path, result.returncode, str(log_path)


def compute_bout_stats(events_df, summary_df, ipi_states_df=None):
    """Compute per-particle bout statistics using HMM state assignments.

    When ipi_states_df is provided (from pumping_ipi_states.csv), quiescent
    bouts are defined as maximal runs of consecutive HMM-quiescent IPIs.
    This avoids counting single slow IPIs as state transitions.

    Particles absent from ipi_states_df (censored by the HMM) receive NaN
    for all bout columns.

    Parameters
    ----------
    events_df : pd.DataFrame
        pumping_events.csv — columns: particle, time_s, method
    summary_df : pd.DataFrame
        pumping_summary.csv — provides track_duration_s per particle
    ipi_states_df : pd.DataFrame or None
        pumping_ipi_states.csv — columns: particle, t_start_s, t_end_s, ipi_s, state

    Returns
    -------
    pd.DataFrame with columns:
        particle, n_quiescent_bouts, total_quiescent_time_s,
        active_duration_s, n_pumps_scipy_active, n_pumps_ampd_active
    """
    import pandas as pd

    scipy_ev = events_df[events_df["method"] == "scipy"].copy()
    ampd_ev  = events_df[events_df["method"] == "ampd"].copy()
    dur_map  = summary_df.set_index("particle")["track_duration_s"].to_dict()

    # Index HMM states by particle for fast lookup
    if ipi_states_df is not None and not ipi_states_df.empty:
        states_by_particle = {p: grp.sort_values("t_start_s")
                              for p, grp in ipi_states_df.groupby("particle")}
    else:
        states_by_particle = {}

    records = []
    all_particles = summary_df["particle"].unique()

    for particle in all_particles:
        particle = int(particle)
        track_dur = dur_map.get(particle, np.nan)

        if particle not in states_by_particle:
            # Censored or no HMM data — skip bout stats for this particle
            records.append({
                "particle":               particle,
                "n_quiescent_bouts":      np.nan,
                "total_quiescent_time_s": np.nan,
                "active_duration_s":      np.nan,
                "n_pumps_scipy_active":   np.nan,
                "n_pumps_ampd_active":    np.nan,
            })
            continue

        state_grp = states_by_particle[particle]
        states_arr = state_grp["state"].values
        t_starts   = state_grp["t_start_s"].values
        t_ends     = state_grp["t_end_s"].values
        ipis       = state_grp["ipi_s"].values
        is_q       = states_arr == "quiescent"

        # Quiescent bouts: maximal runs of consecutive HMM-quiescent IPIs
        n_bouts = 0
        total_q = 0.0
        in_q    = False
        for i, q in enumerate(is_q):
            if q:
                total_q += ipis[i]
                if not in_q:
                    n_bouts += 1
                    in_q = True
            else:
                in_q = False

        active_dur = max(track_dur - total_q, 0.0) if not np.isnan(track_dur) else np.nan

        # Count scipy and AMPD events in non-quiescent IPI windows
        def count_in_active(event_times):
            n = 0
            for t in event_times:
                # Find which IPI interval contains t
                idx = int(np.searchsorted(t_starts, t, side="right")) - 1
                if idx < 0 or idx >= len(is_q) or not is_q[idx]:
                    n += 1
            return n

        scipy_times = scipy_ev.loc[scipy_ev["particle"] == particle, "time_s"].values
        ampd_times  = ampd_ev.loc[ampd_ev["particle"]  == particle, "time_s"].values

        records.append({
            "particle":               particle,
            "n_quiescent_bouts":      n_bouts,
            "total_quiescent_time_s": round(total_q, 4),
            "active_duration_s":      round(active_dur, 4) if not np.isnan(active_dur) else np.nan,
            "n_pumps_scipy_active":   count_in_active(scipy_times),
            "n_pumps_ampd_active":    count_in_active(ampd_times),
        })

    return pd.DataFrame(records)


def run_mixed_models(combined_df, directory):
    """Run six Poisson GLMMs via Rscript.

    Returns a dict {"fixed_effects": DataFrame, "contrasts": DataFrame},
    or None if R is unavailable, required columns are absent, or all models fail.
    """
    import pandas as pd

    if subprocess.run(["Rscript", "--version"],
                      capture_output=True).returncode != 0:
        print("  [models] Rscript not found — skipping mixed models")
        return None

    required = {"genotype", "food", "condition", "video_path",
                "n_pumps_scipy_active", "active_duration_s",
                "n_quiescent_bouts", "track_duration_s",
                "total_quiescent_time_s"}
    missing = required - set(combined_df.columns)
    if missing:
        print(f"  [models] Missing columns {missing} — skipping mixed models")
        return None

    tmpdir        = Path(tempfile.mkdtemp())
    data_csv      = tmpdir / "model_data.csv"
    results_csv   = tmpdir / "model_results.csv"
    contrasts_csv = tmpdir / "model_contrasts.csv"
    r_path        = tmpdir / "models.R"

    model_cols = list(required | {"particle"})
    model_df   = combined_df[[c for c in model_cols if c in combined_df.columns]].copy()
    model_df   = model_df.dropna(subset=["genotype", "food", "condition"])
    model_df.to_csv(data_csv, index=False)
    r_path.write_text(R_SCRIPT)

    result = subprocess.run(
        ["Rscript", str(r_path), str(data_csv), str(results_csv), str(contrasts_csv)],
        capture_output=True, text=True, timeout=600
    )
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"  [R] {line}")
    if result.returncode != 0:
        print(f"  [models] R script failed:\n{result.stderr[-2000:]}")
        return None
    if not results_csv.exists():
        print("  [models] R produced no output")
        return None

    out = {"fixed_effects": pd.read_csv(results_csv), "contrasts": None}
    if contrasts_csv.exists():
        out["contrasts"] = pd.read_csv(contrasts_csv)
    return out


def format_model_section(fixed_df, contrasts_df=None):
    """Format fixed-effects and post-hoc contrast tables as a markdown string."""
    import pandas as pd

    MODEL_META = {
        "food_rate":     ("Food type comparison",     "active pump rate",       "scipy pumps / active_duration_s"),
        "food_entry":    ("Food type comparison",     "quiescence entry rate",  "quiescent bouts / track_duration_s"),
        "food_exit":     ("Food type comparison",     "quiescence exit rate",   "quiescent bouts / total_quiescent_time_s"),
        "feeding_rate":  ("Feeding state comparison", "active pump rate",       "scipy pumps / active_duration_s"),
        "feeding_entry": ("Feeding state comparison", "quiescence entry rate",  "quiescent bouts / track_duration_s"),
        "feeding_exit":  ("Feeding state comparison", "quiescence exit rate",   "quiescent bouts / total_quiescent_time_s"),
    }

    COMPARISON_HEADERS = {
        "Food type comparison":     "### Comparison 1: Food type (fed animals, OP50 vs JUb39)\n",
        "Feeding state comparison":  "### Comparison 2: Feeding state (OP50 animals, fed vs off-food)\n",
    }

    def p_fmt(v):
        try:
            return "<0.0001" if float(v) < 0.0001 else f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    lines = [
        "\n## Statistical models\n",
        "*Poisson GLMM (lme4/emmeans); negative binomial used if Pearson dispersion > 2.*  ",
        "*Random effect: (1 | video_path).  Reference levels: genotype = WT, food = OP50, condition = fed.*  ",
        "*Post-hoc contrasts: all pairwise comparisons on the link (log) scale, Tukey-adjusted.*\n",
    ]
    last_comp = None

    for model_id, (comparison, outcome, offset_desc) in MODEL_META.items():
        fe_sub = fixed_df[fixed_df["model_id"] == model_id]
        if fe_sub.empty:
            continue

        if comparison != last_comp:
            lines.append(COMPARISON_HEADERS[comparison])
            last_comp = comparison

        fam  = fe_sub["family"].iloc[0]
        disp = fe_sub["dispersion"].iloc[0]
        lines.append(f"#### {outcome.capitalize()}")
        lines.append(f"*offset: {offset_desc} | family: {fam} | dispersion: {disp}*\n")

        # Fixed effects
        lines.append("**Fixed effects**\n")
        lines.append("| term | estimate | SE | z | p |")
        lines.append("|------|----------|----|---|---|")
        for _, row in fe_sub.iterrows():
            term  = str(row["term"])
            est   = f"{row['estimate']:.4f}"
            se    = f"{row['std_error']:.4f}"
            z     = f"{row['z_value']:.3f}"
            p_str = p_fmt(row["p_value"])
            if ":" in term:
                lines.append(f"| **{term}** | **{est}** | **{se}** | **{z}** | **{p_str}** |")
            else:
                lines.append(f"| {term} | {est} | {se} | {z} | {p_str} |")

        # Post-hoc contrasts
        if contrasts_df is not None:
            ct_sub = contrasts_df[contrasts_df["model_id"] == model_id]
            if not ct_sub.empty:
                lines.append("\n**Post-hoc pairwise contrasts (Tukey)**\n")
                lines.append("| contrast | estimate | SE | z | p |")
                lines.append("|----------|----------|----|---|---|")
                for _, row in ct_sub.iterrows():
                    est   = f"{row['estimate']:.4f}"
                    se    = f"{row['std_error']:.4f}"  if "std_error" in row else "—"
                    z     = f"{row['z_value']:.3f}"    if "z_value"   in row else "—"
                    p_str = p_fmt(row["p_value"])      if "p_value"   in row else "—"
                    lines.append(f"| {row['contrast']} | {est} | {se} | {z} | {p_str} |")

        lines.append("")

    return "\n".join(lines)


def collect_pumping_results(directory, metadata_path=None):
    """Collect pumping results from all *_results/ subdirectories.

    For each video, reads pumping_summary.csv and pumping_events.csv.
    Computes per-particle bout statistics, merges metadata, runs six
    Poisson GLMMs via R, and writes a summary CSV, plots, and markdown report.
    """
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [collect] skipping — missing dependency: {e}")
        return

    directory      = Path(directory).resolve()
    summary_files  = sorted(directory.rglob("pumping_summary.csv"))

    if not summary_files:
        print("  [collect] No pumping_summary.csv files found — skipping.")
        return

    print(f"\n  [collect] Found {len(summary_files)} result(s) — building batch summary")

    frames = []
    for summary_path in summary_files:
        results_dir  = summary_path.parent
        rname        = results_dir.name
        video_stem   = rname[:-len("_results")] if rname.endswith("_results") else rname
        video_folder = results_dir.parent

        video_file = None
        for ext in (".avi", ".mp4", ".tif", ".tiff"):
            candidate = video_folder / (video_stem + ext)
            if candidate.exists():
                video_file = candidate
                break
        video_path_str = str(video_file or (video_folder / video_stem))

        summary_df = pd.read_csv(summary_path)
        summary_df.insert(0, "video_path",  video_path_str)
        summary_df.insert(1, "folder_name", video_folder.name)

        # Compute bout stats from events + HMM state CSVs if available
        events_path     = results_dir / "pumping_events.csv"
        ipi_states_path = results_dir / "pumping_ipi_states.csv"
        if events_path.exists():
            try:
                events_df     = pd.read_csv(events_path)
                ipi_states_df = pd.read_csv(ipi_states_path) if ipi_states_path.exists() else None
                bout_df       = compute_bout_stats(events_df, summary_df, ipi_states_df)
                if not bout_df.empty:
                    summary_df = summary_df.merge(bout_df, on="particle", how="left")
            except Exception as exc:
                print(f"  [collect] Warning: could not compute bout stats for "
                      f"{results_dir.name}: {exc}")

        frames.append(summary_df)

    combined = pd.concat(frames, ignore_index=True)

    # Merge metadata
    if metadata_path and Path(metadata_path).is_file():
        meta         = pd.read_csv(metadata_path)
        meta.columns = meta.columns.str.strip()
        meta         = meta.apply(lambda c: c.str.strip() if c.dtype == object else c)
        combined = combined.merge(meta, on="folder_name", how="left")
        print(f"  [collect] Metadata loaded from {metadata_path}")
    else:
        combined["genotype"]  = ""
        combined["food"]      = ""
        combined["condition"] = combined["folder_name"]

    # Derived rate column for plots.
    # Prefer rate_active_ampd_hz from the HMM classifier (already in pumping_summary.csv)
    # — it uses HMM-defined active windows and is well-validated against the training data.
    # Fall back to a scipy-based rate if HMM column is absent (e.g. non-pumping configs).
    if "rate_active_ampd_hz" not in combined.columns:
        combined["rate_active_ampd_hz"] = np.where(
            combined["active_duration_s"].fillna(0) >= MIN_ACTIVE_DURATION_S,
            combined["n_pumps_scipy_active"] / combined["active_duration_s"].replace(0, np.nan),
            np.nan
        )
    # Keep active_rate_hz as a diagnostic column (n_pumps_ampd_active / active_duration_s)
    combined["active_rate_hz"] = np.where(
        combined["active_duration_s"].fillna(0) >= MIN_ACTIVE_DURATION_S,
        combined["n_pumps_ampd_active"] / combined["active_duration_s"].replace(0, np.nan),
        np.nan
    )
    combined["entry_rate_per_min"] = (
        combined["n_quiescent_bouts"] / combined["track_duration_s"] * 60
    )
    combined["mean_bout_duration_s"] = np.where(
        combined.get("n_quiescent_bouts", pd.Series(0)) > 0,
        combined["total_quiescent_time_s"] / combined["n_quiescent_bouts"],
        np.nan
    )

    # Reorder columns
    key_cols = ["video_path", "particle", "track_duration_s",
                "n_pumps_ampd_active", "active_duration_s",
                "active_rate_hz", "n_quiescent_bouts",
                "total_quiescent_time_s", "entry_rate_per_min",
                "mean_bout_duration_s", "frac_quiescent",
                "genotype", "food", "condition", "folder_name"]
    ordered  = [c for c in key_cols if c in combined.columns]
    rest     = [c for c in combined.columns if c not in ordered]
    combined = combined[ordered + rest]

    csv_out = directory / "batch_pumping_summary.csv"
    combined.to_csv(csv_out, index=False)
    print(f"  [collect] Batch CSV   → {csv_out}")

    # ── Plots ────────────────────────────────────────────────────────────────
    # Sort folders by (genotype, condition, food) so related groups are adjacent.
    has_meta = all(c in combined.columns for c in ("genotype", "condition", "food"))
    if has_meta:
        sort_keys = ["genotype", "condition", "food", "folder_name"]
    else:
        sort_keys = ["folder_name"]

    folder_order = (
        combined[sort_keys]
        .drop_duplicates("folder_name")
        .sort_values(sort_keys)
        ["folder_name"]
        .tolist()
    )
    folder_idx = {f: i for i, f in enumerate(folder_order)}
    n_folders  = len(folder_order)
    rng        = np.random.default_rng(42)

    # Assign a colour to each (genotype, condition) group for visual separation.
    if has_meta:
        group_key  = combined[["folder_name", "genotype", "condition"]].drop_duplicates("folder_name")
        group_key  = group_key.set_index("folder_name")
        groups     = sorted({(r.genotype, r.condition)
                             for r in group_key.itertuples()
                             if pd.notna(r.genotype)})
        palette    = plt.cm.tab10.colors
        group_color = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
        def folder_color(fname):
            row = group_key.loc[fname] if fname in group_key.index else None
            if row is None or pd.isna(row.genotype):
                return "steelblue"
            return group_color.get((row.genotype, row.condition), "steelblue")
    else:
        def folder_color(_):
            return "steelblue"

    plot_metrics = [
        ("rate_active_ampd_hz",   "Active pump rate (Hz)"),
        ("entry_rate_per_min",    "Quiescence entry rate (bouts/min)"),
        ("mean_bout_duration_s",  "Mean quiescent bout duration (s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(12, n_folders * 0.9 + 3), 5))
    fig.suptitle("Batch pumping summary", fontsize=13)

    for ax, (col, ylabel) in zip(axes, plot_metrics):
        if col not in combined.columns:
            ax.set_visible(False)
            continue
        plot_df = combined[["folder_name", col]].dropna(subset=[col])
        for fname, grp in plot_df.groupby("folder_name"):
            xi   = folder_idx.get(fname, 0)
            vals = grp[col].values
            color = folder_color(fname)
            ax.scatter(xi + rng.uniform(-0.18, 0.18, len(vals)), vals,
                       alpha=0.5, s=16, color=color, linewidths=0, zorder=2)
            ax.plot([xi - 0.32, xi + 0.32], [np.median(vals)] * 2,
                    color="black", lw=2, zorder=3)

        # Draw faint vertical separators between condition groups.
        if has_meta:
            prev_group = None
            for fname in folder_order:
                row = group_key.loc[fname] if fname in group_key.index else None
                g = (row.genotype, row.condition) if (row is not None and pd.notna(row.genotype)) else None
                if prev_group is not None and g != prev_group:
                    xi = folder_idx[fname] - 0.5
                    ax.axvline(xi, color="0.75", lw=0.8, ls="--", zorder=1)
                prev_group = g

        ax.set_xticks(range(n_folders))
        ax.set_xticklabels(folder_order, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.7, n_folders - 0.3)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)

    # Legend for group colours.
    if has_meta and groups:
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=group_color[g], label=f"{g[0]} / {g[1]}")
                   for g in groups]
        fig.legend(handles=handles, title="genotype / condition",
                   loc="upper right", fontsize=7, title_fontsize=7,
                   framealpha=0.8, ncol=max(1, len(groups) // 4))

    plt.tight_layout()
    plot_out = directory / "batch_pumping_plots.png"
    fig.savefig(plot_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [collect] Plots       → {plot_out}")

    # ── Mixed models ─────────────────────────────────────────────────────────
    model_out      = run_mixed_models(combined, directory)
    fixed_df       = model_out["fixed_effects"] if model_out else None
    contrasts_df   = model_out["contrasts"]     if model_out else None

    # ── Markdown report ───────────────────────────────────────────────────────
    n_videos    = combined["video_path"].nunique()
    n_particles = len(combined)

    lines = [
        "# Batch pumping report\n",
        f"**Directory:** `{directory}`  ",
        f"**Videos:** {n_videos}  ",
        f"**Particles:** {n_particles}  ",
    ]
    if metadata_path:
        lines.append(f"**Metadata:** `{metadata_path}`  ")

    lines += ["", "![Pumping summary](batch_pumping_plots.png)\n",
              "## Per-condition summary\n",
              "| condition | n | active rate (Hz) | entry rate (bouts/min) | mean bout duration (s) |",
              "|-----------|---|-----------------|------------------------|------------------------|"]

    grp = combined.groupby("condition", sort=True).agg(
        n=("particle", "count"),
        med_rate=("rate_active_ampd_hz",  "median"),
        med_entry=("entry_rate_per_min",  "median"),
        med_bout=("mean_bout_duration_s", "median"),
    ).reset_index()

    for _, row in grp.iterrows():
        def fmt(v): return f"{v:.2f}" if __import__("pandas").notna(v) else "—"
        lines.append(
            f"| {row['condition']} | {int(row['n'])} | "
            f"{fmt(row['med_rate'])} | {fmt(row['med_entry'])} | {fmt(row['med_bout'])} |"
        )

    if fixed_df is not None:
        lines.append(format_model_section(fixed_df, contrasts_df))

    md_out = directory / "batch_pumping_report.md"
    md_out.write_text("\n".join(lines) + "\n")
    print(f"  [collect] Report      → {md_out}")


def collect_centroid_results(directory):
    """Collect tracks.csv files from centroid/postural runs and plot mean speed by subfolder."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"  [collect] skipping — missing dependency: {e}")
        return

    directory    = Path(directory).resolve()
    tracks_files = sorted(directory.rglob("tracks.csv"))

    if not tracks_files:
        print("  [collect] No tracks.csv files found — skipping.")
        return

    print(f"\n  [collect] Found {len(tracks_files)} result(s) — building speed summary")

    frames = []
    for tracks_path in tracks_files:
        # top-level subfolder relative to the search directory (e.g. N2, cest-2.1, tbh-1)
        rel        = tracks_path.relative_to(directory)
        subfolder  = rel.parts[0]
        video_stem = tracks_path.parent.name
        if video_stem.endswith("_results"):
            video_stem = video_stem[: -len("_results")]

        # Skip results from annotated MP4s or notAnalyzed folders
        if "_annotated" in video_stem or "notAnalyzed" in tracks_path.parts:
            continue

        try:
            df = pd.read_csv(tracks_path)
            if "speed" not in df.columns:
                print(f"  [collect] No speed column in {tracks_path.name} — skipping")
                continue

            per_particle = (
                df.groupby("particle")["speed"]
                .median()
                .reset_index()
                .rename(columns={"speed": "median_speed_mm_s"})
            )
            per_particle["video"]     = video_stem
            per_particle["subfolder"] = subfolder
            frames.append(per_particle)
        except Exception as exc:
            print(f"  [collect] Warning: could not read {tracks_path}: {exc}")

    if not frames:
        print("  [collect] No valid speed data found.")
        return

    combined = pd.concat(frames, ignore_index=True)

    csv_out = directory / "batch_speed_summary.csv"
    combined.to_csv(csv_out, index=False)
    print(f"  [collect] Batch CSV → {csv_out}")

    # ── Strip plot ────────────────────────────────────────────────────────────
    subfolders = sorted(combined["subfolder"].unique())
    n_groups   = len(subfolders)
    rng        = np.random.default_rng(42)
    palette    = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.8 + 2), 5))

    for i, sf in enumerate(subfolders):
        vals  = combined.loc[combined["subfolder"] == sf, "median_speed_mm_s"].dropna()
        color = palette[i % len(palette)]
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(i + jitter, vals, alpha=0.5, s=20, color=color,
                   linewidths=0, zorder=2)
        ax.plot([i - 0.32, i + 0.32], [np.median(vals)] * 2,
                color="black", lw=2.5, zorder=3)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(subfolders, rotation=30, ha="right")
    ax.set_ylabel("Median speed per particle (mm/s)")
    ax.set_title("Speed summary by condition")
    ax.set_xlim(-0.7, n_groups - 0.3)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plot_out = directory / "batch_speed_summary.png"
    fig.savefig(plot_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [collect] Plot → {plot_out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run track.py in parallel on all video files in a directory tree"
    )
    parser.add_argument("directory", nargs="?", default=".",
                        help="Directory to search for video files (default: current directory)")
    parser.add_argument("-c", "--config", required=False, default=None,
                        help="Config YAML file to pass to track.py (not needed with --summary-only)")
    parser.add_argument("-j", "--jobs", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--metadata", default=None,
                        help="CSV with columns folder_name, genotype, food, condition")
    parser.add_argument("--no-summary", dest="collect_summary",
                        action="store_false", default=True,
                        help="Skip batch collection and model fitting after run")
    parser.add_argument("--summary-only", action="store_true", default=False,
                        help="Skip video processing; only run batch collection on existing results")
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    if args.summary_only:
        if args.config:
            with open(args.config) as f:
                config_mode = yaml.safe_load(f).get("mode", "postural").strip().lower()
        else:
            config_mode = "pumping"
        if config_mode == "pumping":
            collect_pumping_results(directory, metadata_path=args.metadata)
        else:
            collect_centroid_results(directory)
        return

    if not args.config:
        print("ERROR: -c/--config is required unless --summary-only is set")
        sys.exit(1)

    config = str(Path(args.config).resolve())
    if not Path(config).is_file():
        print(f"ERROR: Config file not found: {config}")
        sys.exit(1)

    video_files = sorted(
        f for f in directory.rglob("*")
        if f.suffix.lower() in VIDEO_EXTENSIONS
        and not f.stem.endswith("_pumping")
        and not f.stem.endswith("_annotated")
        and "_results"     not in f.parts
        and "notAnalyzed"  not in f.parts
    )

    if not video_files:
        print(f"No video files found in {directory}")
        sys.exit(0)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    n_workers = min(args.jobs, len(video_files))
    print(f"Found {len(video_files)} file(s) — running {n_workers} at a time")
    print(f"Config:  {config}")
    print(f"Logs:    {log_dir.resolve()}/")
    print(f"Monitor: tail -f logs/<name>.log")
    print("-" * 60)

    n_ok, n_err = 0, 0
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for video in video_files:
            f = pool.submit(run_one, str(video), config, str(log_dir))
            futures[f] = video
        for future in as_completed(futures):
            video_path, returncode, log_path = future.result()
            name = Path(video_path).name
            if returncode == 0:
                n_ok += 1
                print(f"  OK      {name}")
            else:
                n_err += 1
                print(f"  FAILED  {name}  (see {log_path})")

    print("\n" + "=" * 60)
    print(f"Done: {n_ok} succeeded, {n_err} failed")
    if n_err:
        print("Check logs/ for details on failed jobs.")

    with open(config) as f:
        config_mode = yaml.safe_load(f).get("mode", "postural").strip().lower()

    if args.collect_summary:
        if config_mode == "pumping":
            collect_pumping_results(directory, metadata_path=args.metadata)
        else:
            collect_centroid_results(directory)


if __name__ == "__main__":
    main()
