"""
Batch postural comparison for multi-worm locomotion datasets.

Computes per-recording forward-run speed, reversal rate, and pirouette rate from
ODLabTracker tracks.csv output, then fits linear mixed-effects models (LME) via
R/lme4+lmerTest to estimate genotype effects relative to N2 controls.

Outputs:
    <out-dir>/postural_comparison.csv           per-recording summary table
    <out-dir>/postural_comparison_stats.csv     LME coefficients and q-values
    <out-dir>/postural_comparison.png           strip plot + speed distributions
    <out-dir>/supplemental_particle_data.csv    per-particle raw data (for reanalysis)

Requires: ODLabTracker, numpy, pandas, matplotlib, scipy, pyarrow, R with lme4+lmerTest.

Exclusion file (--exclude):
    CSV with columns genotype,date (YYYYMMDD). Matching recordings are dropped
    before normalization and plotting. Lines starting with # are ignored.

Usage:
    python dev/batch_postural_comparison.py --data-dir /path/to/dataset --out-dir results/
    python dev/batch_postural_comparison.py --out-dir results/  # replot from cached parquet
    python dev/batch_postural_comparison.py --out-dir results/ --refit   # re-run LME
    python dev/batch_postural_comparison.py --out-dir results/ --refresh  # rescan + refit
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats as sp_stats

OUT_DIR = "data/postural_results"
FRAME_RATE = 10   # fps, from IR_medium.yaml
N2_FOLDER = "N2"

METRICS = ["speed", "reversal_rate", "pirouette_rate"]
METRIC_LABELS = {
    "speed":          "Speed\n(fold-change vs N2)",
    "reversal_rate":  "Reversal rate\n(fold-change vs N2)",
    "pirouette_rate": "Pirouette rate\n(fold-change vs N2)",
}
# Map analysis metric name → column name in per-particle DataFrame
METRIC_COL = {
    "speed":          "fwd_speed",
    "reversal_rate":  "reversal_rate",
    "pirouette_rate": "pirouette_rate",
}


# ── per-recording loader ─────────────────────────────────────────────────────

def _parse_tracks(results_dir, frame_rate, min_speed=0.0):
    """
    Shared parsing logic.  Returns per-particle DataFrame with columns:
        fwd_speed, reversal_rate, pirouette_rate, all_frame_speed, n_frames
    or None if the file is missing / unusable.
    Particles with all_frame_speed < min_speed are excluded.
    """
    csv_path = os.path.join(results_dir, "tracks.csv")
    if not os.path.exists(csv_path):
        return None

    needed = {"frame", "particle", "speed", "movement_type",
              "reversal_start", "pirouette_start"}
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in needed)
    except Exception as e:
        print(f"  Warning: could not read {csv_path}: {e}", file=sys.stderr)
        return None

    if df.empty or "particle" not in df.columns:
        return None

    grp = df.groupby("particle")
    per_p = grp.agg(n_frames=("frame", "count"), all_frame_speed=("speed", "mean"))
    per_p["duration_min"] = per_p["n_frames"] / frame_rate / 60

    if "movement_type" in df.columns:
        fwd = df[df["movement_type"] == "forward_run"]
        per_p["fwd_speed"] = fwd.groupby("particle")["speed"].median()
    else:
        per_p["fwd_speed"] = per_p["all_frame_speed"]

    if "reversal_start" in df.columns:
        per_p["n_reversals"]   = grp["reversal_start"].sum()
        per_p["reversal_rate"] = per_p["n_reversals"] / per_p["duration_min"]
    else:
        per_p["n_reversals"]   = 0
        per_p["reversal_rate"] = np.nan

    if "pirouette_start" in df.columns:
        per_p["n_pirouettes"]   = grp["pirouette_start"].sum()
        per_p["pirouette_rate"] = per_p["n_pirouettes"] / per_p["duration_min"]
    else:
        per_p["n_pirouettes"]   = 0
        per_p["pirouette_rate"] = np.nan

    if min_speed > 0:
        per_p = per_p[per_p["all_frame_speed"] >= min_speed]

    return per_p.reset_index(drop=True) if not per_p.empty else None


def load_recording(results_dir, frame_rate, min_speed=0.0):
    """Return per-recording summary dict, or None if unusable."""
    per_p = _parse_tracks(results_dir, frame_rate, min_speed)
    if per_p is None or per_p.empty:
        return None
    return {
        "speed":          per_p["fwd_speed"].median(),
        "reversal_rate":  per_p["reversal_rate"].mean(),
        "pirouette_rate": per_p["pirouette_rate"].mean(),
        "n_reversals":    int(per_p["n_reversals"].sum()),
        "n_pirouettes":   int(per_p["n_pirouettes"].sum()),
        "n_particles":    len(per_p),
        "n_excluded":     0,
    }


def load_recording_particles(results_dir, frame_rate, min_speed=0.0):
    """Return per-particle DataFrame (fwd_speed, reversal_rate, pirouette_rate), or None."""
    per_p = _parse_tracks(results_dir, frame_rate, min_speed)
    if per_p is None or per_p.empty:
        return None
    return per_p[["fwd_speed", "reversal_rate", "pirouette_rate"]].copy()


def _parse_fwd_frames(results_dir, min_speed=0.0):
    """
    Return per-forward-run-frame DataFrame (frame, particle, speed), or None.

    Only particles passing the min_speed all-frame mean gate are included.
    """
    csv_path = os.path.join(results_dir, "tracks.csv")
    if not os.path.exists(csv_path):
        return None
    needed = {"frame", "particle", "speed", "movement_type"}
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in needed)
    except Exception:
        return None
    if df.empty or "movement_type" not in df.columns:
        return None
    if min_speed > 0:
        per_p  = df.groupby("particle")["speed"].mean()
        valid  = per_p[per_p >= min_speed].index
        df     = df[df["particle"].isin(valid)]
    fwd = df[df["movement_type"] == "forward_run"][["frame", "particle", "speed"]].copy()
    return fwd if not fwd.empty else None


# ── dataset scanners ─────────────────────────────────────────────────────────

def _iter_recordings(data_dir):
    """Yield (geno, date, fname, results_dir) for every AVI with a _results dir."""
    for geno in sorted(os.listdir(data_dir)):
        gdir = os.path.join(data_dir, geno)
        if not os.path.isdir(gdir) or geno == "placeholder" or geno.startswith("._"):
            continue
        for fname in sorted(os.listdir(gdir)):
            if not fname.endswith(".avi") or fname.startswith("._"):
                continue
            m = re.match(r"^(\d{8})", fname)
            if not m:
                continue
            date        = m.group(1)
            stem        = os.path.splitext(fname)[0]
            results_dir = os.path.join(gdir, stem + "_results")
            yield geno, date, fname, results_dir


def scan_dataset(data_dir, frame_rate, min_speed=0.0):
    """Return DataFrame with one row per successfully loaded recording."""
    rows = []
    for geno, date, fname, results_dir in _iter_recordings(data_dir):
        stats = load_recording(results_dir, frame_rate, min_speed=min_speed)
        if stats is None:
            print(f"  Skipping {geno}/{fname} (no tracks.csv or empty)")
            continue
        rows.append({"genotype": geno, "date": date,
                     "results_dir": results_dir, **stats})
    return pd.DataFrame(rows)


def scan_dataset_particles(data_dir, frame_rate, exclusions=None, min_speed=0.0):
    """Return per-particle DataFrame with genotype, date, recording_id columns."""
    if exclusions is None:
        exclusions = set()
    chunks = []
    for geno, date, _fname, results_dir in _iter_recordings(data_dir):
        if (geno, date) in exclusions:
            continue
        particles = load_recording_particles(results_dir, frame_rate, min_speed=min_speed)
        if particles is None:
            continue
        particles["genotype"]     = geno
        particles["date"]         = date
        particles["recording_id"] = f"{geno}##{date}"
        chunks.append(particles)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def scan_dataset_fwd_frames(data_dir, frame_rate, exclusions=None, min_speed=0.0):
    """
    Return per-forward-run-frame DataFrame for the speed LME.

    Columns: speed, genotype, date, recording_id, particle_uid.
    particle_uid is globally unique: '<geno>##<date>##p<particle_num>'.
    """
    if exclusions is None:
        exclusions = set()
    chunks = []
    for geno, date, _fname, results_dir in _iter_recordings(data_dir):
        if (geno, date) in exclusions:
            continue
        fwd = _parse_fwd_frames(results_dir, min_speed=min_speed)
        if fwd is None or fwd.empty:
            continue
        rec_id           = f"{geno}##{date}"
        fwd              = fwd.copy()
        fwd["genotype"]     = geno
        fwd["date"]         = date
        fwd["recording_id"] = rec_id
        fwd["particle_uid"] = rec_id + "##p" + fwd["particle"].astype(str)
        chunks.append(fwd[["speed", "genotype", "date", "recording_id", "particle_uid"]])
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ── normalization ────────────────────────────────────────────────────────────

def add_normalization(df):
    """
    For each metric add <metric>_norm (fold-change vs N2) and <metric>_date_matched.
    N2 recordings are normalized to the grand N2 mean so they scatter around 1.0.
    """
    n2_rows = df[df["genotype"] == N2_FOLDER]
    if n2_rows.empty:
        sys.exit(f"No recordings found for reference genotype '{N2_FOLDER}'.")

    n2_by_date = n2_rows.groupby("date")[METRICS].mean()
    n2_grand   = n2_rows[METRICS].mean()

    for metric in METRICS:
        date_ref_map = n2_by_date[metric].to_dict()
        grand_ref    = n2_grand[metric]

        same_date_ref = df["date"].map(date_ref_map)
        date_matched  = same_date_ref.notna()

        is_n2 = df["genotype"] == N2_FOLDER
        same_date_ref[is_n2] = grand_ref
        date_matched[is_n2]  = True

        ref = same_date_ref.fillna(grand_ref)
        df[f"{metric}_norm"]         = df[metric] / ref
        df[f"{metric}_date_matched"] = date_matched

    return df


# ── genotype sort order ──────────────────────────────────────────────────────

def genotype_order(df, stat_df=None):
    """Sort genotypes ascending (slowest/lowest FC at bottom).
    With stat_df: sort by LME speed fold-change; unmatched genotypes fall back to raw FC.
    Without stat_df: sort by mean raw speed."""
    if stat_df is not None:
        speed_lme = (stat_df[stat_df["metric"] == "speed"]
                     .set_index("genotype")["fold_change"])
        n2_speed = df[df["genotype"] == N2_FOLDER]["speed"].mean()
        raw_fc   = df.groupby("genotype")["speed"].mean() / n2_speed

        def _key(g):
            if g == N2_FOLDER:
                return 1.0
            fc = speed_lme.get(g, np.nan)
            return fc if not pd.isna(fc) else float(raw_fc.get(g, 1.0))

        return sorted(df["genotype"].unique(), key=_key)

    return (df.groupby("genotype")["speed"]
              .mean().sort_values(ascending=True)
              .index.tolist())


# ── exclusion loader ─────────────────────────────────────────────────────────

def load_exclusions(path):
    """Return set of (genotype, date_str) tuples to drop. Returns empty set if path is None."""
    if path is None or not os.path.exists(path):
        return set()
    excluded = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("genotype"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                excluded.add((parts[0].strip(), parts[1].strip()))
    return excluded


# ── statistics ──────────────────────────────────────────────────────────────

def _bh_correct(pvals):
    """Benjamini-Hochberg FDR correction; returns q-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    q = pvals * n / rank
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    return np.clip(q, 0, 1)


def _stars(q):
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def _safe_geno(name):
    """Make a genotype name safe for use in a patsy formula."""
    return "g_" + re.sub(r"[^a-zA-Z0-9]", "_", name)


def fit_lme_stats(frame_df, particle_df, order, metrics, out_dir):
    """
    Fit LME models via R/lme4 + lmerTest and return a tidy stats DataFrame.

    Speed (per-forward-run-frame):
        speed ~ genotype + date + (1|recording_id) + (1|particle_uid)

    Reversal rate / pirouette rate (per-particle):
        rate ~ genotype + date + (1|recording_id)

    date is a FIXED effect, not random.  A random date effect undergoes REML
    shrinkage and fails to fully absorb day-to-day N2 variability when σ²_date
    is small relative to residual variance, causing genotypes measured on
    atypically fast/slow dates to receive biased LME estimates.  Fixed date
    effects do not shrink — they are equivalent to within-date normalization
    (what the per-recording blue dots already show).

    Only recordings on dates where N2 was also measured are included in the
    model (keep_n2_dates filter).  Genotypes with no same-date N2 (gba-4,
    glo-1, cest-1.2, ugt-64) are excluded; their stats are reported as NaN.

    lmerTest Satterthwaite df approximation gives correct small-sample p-values
    whose df reflect the number of recordings, not the number of particles.

    Fold-change is expressed relative to the LME model intercept (the N2
    baseline estimated by the model), which is consistent with the model scale.

    Returns DataFrame: genotype, metric, n_recordings, n_particles,
                       fold_change, fc_lo, fc_hi, p_raw, q, stars.
    """
    import subprocess

    # ── write input CSVs ──────────────────────────────────────────────────────
    frame_csv    = os.path.join(out_dir, "_lme_fwd_frames.csv")
    particle_csv = os.path.join(out_dir, "_lme_particles.csv")
    results_csv  = os.path.join(out_dir, "_lme_results.csv")
    r_script_path = os.path.join(out_dir, "_lme_fit.R")

    frame_df.to_csv(frame_csv, index=False)
    particle_df[["reversal_rate", "pirouette_rate",
                 "genotype", "date", "recording_id"]].to_csv(particle_csv, index=False)

    # ── R script ──────────────────────────────────────────────────────────────
    r_script = f"""
suppressMessages(library(lme4))
suppressMessages(library(lmerTest))

n2_ref <- "{N2_FOLDER}"

extract_coefs <- function(fit, metric) {{
  s        <- as.data.frame(coef(summary(fit)))
  s$term   <- rownames(s)
  s$metric <- metric
  rownames(s) <- NULL
  colnames(s) <- c("estimate", "se", "df", "t_value", "p_value", "term", "metric")
  s
}}

# ── helper: filter to dates where N2 was recorded ───────────────────────────
keep_n2_dates <- function(df) {{
  n2_dates <- unique(df$date[df$genotype == n2_ref])
  df[df$date %in% n2_dates, ]
}}

# ── speed: per-forward-run-frame ────────────────────────────────────────────
cat("Fitting speed model...\\n")
fdf        <- read.csv("{frame_csv}", stringsAsFactors = FALSE)
fdf_match  <- keep_n2_dates(fdf)
fdf_match$genotype <- relevel(droplevels(as.factor(fdf_match$genotype)), ref = n2_ref)
fdf_match$date     <- factor(fdf_match$date)
cat(sprintf("  Speed: %d frames, %d genotypes, %d dates\\n",
            nrow(fdf_match), nlevels(fdf_match$genotype), nlevels(fdf_match$date)))
speed_fit  <- lmer(speed ~ genotype + date + (1|recording_id) + (1|particle_uid),
                   data = fdf_match, REML = TRUE,
                   control = lmerControl(optimizer = "bobyqa"))
cat("Speed model done.\\n")

# ── reversal rate: per-particle ─────────────────────────────────────────────
cat("Fitting reversal rate model...\\n")
pdf        <- read.csv("{particle_csv}", stringsAsFactors = FALSE)
pdf_rev    <- keep_n2_dates(pdf[!is.na(pdf$reversal_rate), ])
pdf_rev$genotype <- relevel(droplevels(as.factor(pdf_rev$genotype)), ref = n2_ref)
pdf_rev$date     <- factor(pdf_rev$date)
rev_fit    <- lmer(reversal_rate ~ genotype + date + (1|recording_id),
                   data = pdf_rev, REML = TRUE,
                   control = lmerControl(optimizer = "bobyqa"))
cat("Reversal rate model done.\\n")

# ── pirouette rate: per-particle ─────────────────────────────────────────────
cat("Fitting pirouette rate model...\\n")
pdf_pir    <- keep_n2_dates(pdf[!is.na(pdf$pirouette_rate), ])
pdf_pir$genotype <- relevel(droplevels(as.factor(pdf_pir$genotype)), ref = n2_ref)
pdf_pir$date     <- factor(pdf_pir$date)
pir_fit    <- lmer(pirouette_rate ~ genotype + date + (1|recording_id),
                   data = pdf_pir, REML = TRUE,
                   control = lmerControl(optimizer = "bobyqa"))
cat("Pirouette rate model done.\\n")

# ── collect results ──────────────────────────────────────────────────────────
results <- rbind(
  extract_coefs(speed_fit,  "speed"),
  extract_coefs(rev_fit,    "reversal_rate"),
  extract_coefs(pir_fit,    "pirouette_rate")
)
# strip "genotype" prefix to recover factor level name
results$genotype <- sub("^genotype", "", results$term)

write.csv(results, "{results_csv}", row.names = FALSE)
cat("Results written to {results_csv}\\n")
"""

    with open(r_script_path, "w") as f:
        f.write(r_script)

    print("  Running lme4 in R (speed model may take a few minutes)…")
    proc = subprocess.run(
        ["R", "--no-save", "--quiet", "-f", r_script_path],
        capture_output=True, text=True, timeout=900
    )
    # Print R stdout (progress messages)
    for line in proc.stdout.strip().splitlines():
        print(f"    [R] {line}")
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("lme4 fitting failed — see R stderr above")

    # ── parse lme4 output ─────────────────────────────────────────────────────
    lme4_df = pd.read_csv(results_csv)

    # Build per-genotype, per-metric rows using the LME intercept as N2 baseline
    n_particles = particle_df.groupby("genotype")["recording_id"].agg(
        n_recs="nunique", n_part="count").reset_index()
    n_particles.columns = ["genotype", "n_recordings", "n_particles"]

    all_rows = []
    for metric in metrics:
        m_rows        = lme4_df[lme4_df["metric"] == metric]
        intercept_row = m_rows[m_rows["term"] == "(Intercept)"]
        if intercept_row.empty:
            continue
        n2_baseline = float(intercept_row["estimate"].iloc[0])
        geno_rows   = m_rows[m_rows["genotype"] != "(Intercept)"].copy()

        tests = []
        for geno in order:
            if geno == N2_FOLDER:
                continue
            row    = geno_rows[geno_rows["genotype"] == geno]
            counts = n_particles[n_particles["genotype"] == geno]
            n_rec  = int(counts["n_recordings"].iloc[0]) if not counts.empty else 0
            n_part = int(counts["n_particles"].iloc[0])  if not counts.empty else 0
            if row.empty or pd.isna(row["estimate"].iloc[0]):
                tests.append({"genotype": geno, "metric": metric,
                              "n_recordings": n_rec, "n_particles": n_part,
                              "fold_change": np.nan, "fc_lo": np.nan,
                              "fc_hi": np.nan, "p_raw": np.nan})
                continue
            b  = float(row["estimate"].iloc[0])
            se = float(row["se"].iloc[0])
            p  = float(row["p_value"].iloc[0])
            fc    = 1 + b  / n2_baseline
            fc_lo = 1 + (b - 1.96 * se) / n2_baseline
            fc_hi = 1 + (b + 1.96 * se) / n2_baseline
            tests.append({"genotype": geno, "metric": metric,
                          "n_recordings": n_rec, "n_particles": n_part,
                          "fold_change": fc, "fc_lo": fc_lo, "fc_hi": fc_hi,
                          "p_raw": p})

        pvals    = np.array([t["p_raw"] for t in tests], dtype=float)
        testable = ~np.isnan(pvals)
        qvals    = np.full(len(pvals), np.nan)
        if testable.sum() > 0:
            qvals[testable] = _bh_correct(pvals[testable])
        for t, q in zip(tests, qvals):
            t["q"]     = q
            t["stars"] = _stars(q)
        all_rows.extend(tests)

    return pd.DataFrame(all_rows)


# ── figure colours ────────────────────────────────────────────────────────────

DOT_MATCHED   = "#4393c3"   # steel blue  — same-date N2, filled
DOT_UNMATCHED = "#d6604d"   # coral       — grand-mean N2, open
DOT_N2        = "#999999"   # grey        — N2 recordings
DIAMOND_MUT   = "#b2182b"   # dark red    — mutant LME estimate
DIAMOND_N2    = "#111111"   # near-black  — N2 mean


# ── main comparison figure ───────────────────────────────────────────────────

def make_plot(df, order, stat_df, particle_df, n2_by_date_speed, n2_grand_speed, out_path,
              title="Forward-run speed (fold-change vs N2)"):
    from scipy.stats import gaussian_kde

    ytick  = {g: i for i, g in enumerate(order)}
    n_geno = len(order)
    fig_h  = max(8, n_geno * 0.45)

    fig, (ax, ax_dist) = plt.subplots(
        1, 2, sharey=True, figsize=(13, fig_h),
        gridspec_kw={"width_ratios": [2, 1]})
    fig.subplots_adjust(wspace=0.04)
    fig.suptitle(title, fontsize=11, y=1.01)

    metric      = "speed"
    norm_col    = f"{metric}_norm"
    matched_col = f"{metric}_date_matched"

    # ── left panel: strip plot ────────────────────────────────────────────────
    for _, row in df.iterrows():
        y = ytick[row["genotype"]]
        x = row[norm_col]
        if pd.isna(x):
            continue
        if row["genotype"] == N2_FOLDER:
            ax.plot(x, y, "o", mfc=DOT_N2, mec=DOT_N2, ms=5, alpha=0.65, lw=0, zorder=2)
        else:
            matched = row[matched_col]
            ec, fc = (DOT_MATCHED, DOT_MATCHED) if matched else (DOT_UNMATCHED, "none")
            ax.plot(x, y, "o", mfc=fc, mec=ec, ms=5, alpha=0.65, lw=0, zorder=2)

    for geno in order:
        y  = ytick[geno]
        dc = DIAMOND_N2 if geno == N2_FOLDER else DIAMOND_MUT

        if geno == N2_FOLDER:
            vals = df[(df["genotype"] == N2_FOLDER) & df[matched_col]][norm_col].dropna()
            if len(vals) == 0:
                continue
            center = vals.mean()
            sem    = vals.sem() if len(vals) > 1 else 0.0
            lo, hi = center - sem, center + sem
            raw_n2 = df[df["genotype"] == N2_FOLDER][metric].mean()
            ax.text(center, y + 0.19, f"{raw_n2 * 1000:.0f} µm/s", fontsize=6.5, va="bottom",
                    ha="center", color=dc, alpha=0.85, zorder=7)
            ax.plot([lo, hi], [y, y], color=dc, lw=2.5, solid_capstyle="round", zorder=4)
            ax.plot(center, y, "D", color=dc, ms=7, zorder=5, mec="white", mew=0.5)
            continue

        row = stat_df[(stat_df["genotype"] == geno) & (stat_df["metric"] == metric)]
        if row.empty or pd.isna(row["fold_change"].iloc[0]):
            continue
        r      = row.iloc[0]
        center = r["fold_change"]
        lo     = r["fc_lo"]
        hi     = r["fc_hi"]
        stars  = _stars(r["q"])
        if stars:
            ax.text(hi + 0.06, y, stars, fontsize=11, va="center",
                    ha="left", color="#111111", fontweight="bold", zorder=6)

        ax.plot([lo, hi], [y, y], color=dc, lw=2.5, solid_capstyle="round", zorder=4)
        ax.plot(center, y, "D", color=dc, ms=7, zorder=5, mec="white", mew=0.5)
        ax.text(center, y + 0.19, f"{center:.2f}", fontsize=6.5, va="bottom",
                ha="center", color=dc, alpha=0.85, zorder=7)

    ax.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=0)
    ax.set_xlabel(METRIC_LABELS[metric], fontsize=9)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.8, n_geno - 0.2)
    ax.set_yticks(list(ytick.values()))
    ax.set_yticklabels(list(ytick.keys()), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    leg_handles = [
        mlines.Line2D([], [], color=DOT_MATCHED, marker="o", ls="none",
                      label="Recording —\nsame-date N2"),
        mlines.Line2D([], [], color=DOT_UNMATCHED, marker="o", mfc="none", ls="none",
                      label="Recording —\ngrand-mean N2\n(no same-date N2)"),
        mlines.Line2D([], [], color=DIAMOND_MUT, marker="D", ls="none",
                      mec="white", mew=0.5, label="LME estimate\n± 95% CI"),
        mlines.Line2D([], [], color=DIAMOND_N2, marker="D", ls="none",
                      mec="white", mew=0.5, label="N2 mean ± SEM"),
    ]
    xlim = ax.get_xlim()
    max_dot = df[norm_col].max()
    legend_x = (max_dot - xlim[0]) / (xlim[1] - xlim[0]) + 0.01
    ax.legend(handles=leg_handles, fontsize=7.5, framealpha=0.9,
              loc="center left", bbox_to_anchor=(legend_x, 0.5))

    # ── right panel: per-recording speed distributions ────────────────────────
    VIOLIN_HW   = 0.09   # half-height per recording violin
    N2_VHW      = 0.18   # N2 pooled violin (wider — many particles)
    DENS_THRESH = 0.03   # fraction of peak below which outline is suppressed
    x_range     = np.linspace(0, 2.5, 500)

    def _draw_violin(ax_d, x_r, raw_density, vhw, y_row, color, vals_n):
        density = raw_density / raw_density.max() * vhw
        mask    = raw_density >= raw_density.max() * DENS_THRESH
        idx     = np.where(mask)[0]
        if len(idx) == 0:
            return
        xs = x_r[idx[0]:idx[-1] + 1]
        ds = density[idx[0]:idx[-1] + 1]
        ax_d.fill_between(xs, y_row - ds, y_row + ds, alpha=0.45, color=color, lw=0)
        ax_d.plot(xs, y_row + ds, color=color, lw=0.4, alpha=0.6)
        ax_d.plot(xs, y_row - ds, color=color, lw=0.4, alpha=0.6)
        med = float(np.median(vals_n))
        ax_d.plot([med, med], [y_row - vhw * 0.85, y_row + vhw * 0.85],
                  color=color, lw=1.2, solid_capstyle="round", zorder=3)

    for geno in order:
        y_ctr = ytick[geno]
        ax_dist.axhline(y_ctr - 0.5, color="lightgray", lw=0.3, zorder=0)

        if geno == N2_FOLDER:
            vals = particle_df[particle_df["genotype"] == N2_FOLDER]["fwd_speed"].dropna()
            ref  = n2_grand_speed
            if len(vals) >= 3 and ref > 0:
                vals_norm = vals / ref
                kde       = gaussian_kde(vals_norm, bw_method=0.30)
                _draw_violin(ax_dist, x_range, kde(x_range), N2_VHW, y_ctr, DOT_N2, vals_norm)
        else:
            dates   = sorted(particle_df[particle_df["genotype"] == geno]["date"].unique())
            n       = len(dates)
            if n == 0:
                continue
            spacing = min(0.16, 0.40 / max(1, n - 1))
            offsets = [i * spacing - spacing * (n - 1) / 2 for i in range(n)]

            for date, offset in zip(dates, offsets):
                vals  = particle_df[(particle_df["genotype"] == geno) &
                                    (particle_df["date"] == date)]["fwd_speed"].dropna()
                ref   = n2_by_date_speed.get(date, n2_grand_speed)
                color = DOT_MATCHED if date in n2_by_date_speed else DOT_UNMATCHED
                if len(vals) < 3 or ref <= 0:
                    continue
                vals_norm = vals / ref
                try:
                    kde = gaussian_kde(vals_norm, bw_method=0.30)
                except Exception:
                    continue
                _draw_violin(ax_dist, x_range, kde(x_range), VIOLIN_HW,
                             y_ctr + offset, color, vals_norm)

    ax_dist.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=0)
    ax_dist.set_xlim(0, 2.5)
    ax_dist.set_xlabel("Speed distribution\n(fold-change vs same-date N2)", fontsize=9)
    ax_dist.spines[["top", "right", "left"]].set_visible(False)
    ax_dist.tick_params(axis="x", labelsize=8)
    ax_dist.tick_params(axis="y", left=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch postural comparison for multi-worm locomotion datasets")
    parser.add_argument("--data-dir",   default=None,
                        help="Path to dataset root directory (required with --refresh)")
    parser.add_argument("--out-dir",    default=OUT_DIR)
    parser.add_argument("--frame-rate", type=float, default=FRAME_RATE)
    parser.add_argument("--title",      default="Forward-run speed (fold-change vs N2)",
                        help="Figure title")
    parser.add_argument("--exclude",    default=None,
                        help="CSV file with genotype,date rows to censor")
    parser.add_argument("--min-speed",  type=float, default=0.03,
                        help="Exclude particles with mean speed below this (mm/s). "
                             "Default: 0.03")
    parser.add_argument("--refresh",    action="store_true",
                        help="Force re-scan from data-dir even if local parquet cache exists")
    parser.add_argument("--refit",      action="store_true",
                        help="Re-run R/lme4 models even if stats CSV cache exists. "
                             "--refresh implies --refit.")
    args = parser.parse_args()
    if args.refresh:
        args.refit = True
    if args.refresh and not args.data_dir:
        parser.error("--refresh requires --data-dir")

    os.makedirs(args.out_dir, exist_ok=True)

    exclusions = load_exclusions(args.exclude)
    if exclusions:
        print(f"Exclusions loaded ({len(exclusions)}):")
        for g, d in sorted(exclusions):
            print(f"  {g} / {d}")

    # ── parquet cache ─────────────────────────────────────────────────────────
    particle_cache = os.path.join(args.out_dir, "particle_data.parquet")
    frame_cache    = os.path.join(args.out_dir, "fwd_frame_data.parquet")
    has_cache = (os.path.exists(particle_cache) and os.path.exists(frame_cache)
                 and not args.refresh)

    if has_cache:
        print("Loading cached data from local parquet files…")
        particle_df = pd.read_parquet(particle_cache)
        frame_df    = pd.read_parquet(frame_cache)
        print(f"  {len(particle_df)} particles, {len(frame_df):,} fwd frames")
    else:
        if args.refresh:
            print("--refresh: re-scanning NAS…")
        else:
            print("No local cache found — scanning NAS…")
        print(f"  (min_speed filter: {args.min_speed:.3f} mm/s)")
        particle_df = scan_dataset_particles(
            args.data_dir, args.frame_rate,
            exclusions=exclusions, min_speed=args.min_speed)
        frame_df = scan_dataset_fwd_frames(
            args.data_dir, args.frame_rate,
            exclusions=exclusions, min_speed=args.min_speed)
        particle_df.to_parquet(particle_cache, index=False)
        frame_df.to_parquet(frame_cache, index=False)
        print(f"  Cached to {particle_cache}")
        print(f"  Cached to {frame_cache}")

    # ── per-recording summary ─────────────────────────────────────────────────
    recording_cache = os.path.join(args.out_dir, "recording_data.parquet")
    if os.path.exists(recording_cache) and not args.refresh:
        print("Loading cached recording summary…")
        df = pd.read_parquet(recording_cache)
        print(f"  {len(df)} recordings, {df['genotype'].nunique()} genotypes")
    else:
        print(f"\nScanning dataset… (min_speed filter: {args.min_speed:.3f} mm/s)")
        df = scan_dataset(args.data_dir, args.frame_rate, min_speed=args.min_speed)
        if exclusions:
            before = len(df)
            df = df[~df.apply(lambda r: (r["genotype"], r["date"]) in exclusions, axis=1)]
            print(f"Dropped {before - len(df)} excluded recording(s)")
        print(f"Loaded {len(df)} recordings, {df['genotype'].nunique()} genotypes, "
              f"{df['n_particles'].sum():.0f} total particles")
        df = add_normalization(df)
        df.to_parquet(recording_cache, index=False)
        print(f"  Cached to {recording_cache}")

    n2_recs  = df[df["genotype"] == N2_FOLDER]
    n2_grand = {m: n2_recs[m].mean() for m in METRICS}
    n2_by_date = {m: n2_recs.groupby("date")[m].mean().to_dict() for m in METRICS}

    print("\nN2 grand means (raw):")
    for m in METRICS:
        cv = n2_recs[m].std() / n2_grand[m] * 100
        print(f"  {m}: {n2_grand[m]:.3f}  (CV = {cv:.0f}%)")

    # Save per-recording CSV
    csv_out   = os.path.join(args.out_dir, "postural_comparison.csv")
    col_order = (["genotype", "date", "n_particles",
                  "n_reversals", "n_pirouettes", "n_excluded"]
                 + METRICS
                 + [f"{m}_norm" for m in METRICS]
                 + [f"{m}_date_matched" for m in METRICS]
                 + ["results_dir"])
    df[col_order].to_csv(csv_out, index=False, float_format="%.4f")
    print(f"\nSaved summary CSV: {csv_out}")

    # ── LME stats ─────────────────────────────────────────────────────────────
    stats_out = os.path.join(args.out_dir, "postural_comparison_stats.csv")
    if os.path.exists(stats_out) and not args.refit:
        print("\nLoading cached LME stats (use --refit to rerun models)…")
        stat_df = pd.read_csv(stats_out)
    else:
        print("\nFitting LME models via R/lme4…")
        prelim_order = genotype_order(df)
        stat_df = fit_lme_stats(frame_df, particle_df, prelim_order, METRICS, args.out_dir)
        stat_df.to_csv(stats_out, index=False, float_format="%.4f")
        print(f"Saved stats CSV: {stats_out}")

    order = genotype_order(df, stat_df)

    # ── supplemental table ────────────────────────────────────────────────────
    supp_out = os.path.join(args.out_dir, "supplemental_particle_data.csv")
    supp_df = particle_df[["genotype", "date", "recording_id",
                            "fwd_speed", "reversal_rate", "pirouette_rate"]].copy()
    supp_df = supp_df.rename(columns={
        "fwd_speed":       "forward_run_speed_mm_s",
        "reversal_rate":   "reversal_rate_per_min",
        "pirouette_rate":  "pirouette_rate_per_min",
    })
    supp_df.to_csv(supp_out, index=False, float_format="%.4f")
    print(f"Saved supplemental table: {supp_out}")

    fig_out = os.path.join(args.out_dir, "postural_comparison.png")
    make_plot(df, order, stat_df, particle_df,
              n2_by_date_speed=n2_by_date["speed"],
              n2_grand_speed=n2_grand["speed"],
              out_path=fig_out,
              title=args.title)

    # Print summary
    print("\nGenotype summary (sorted by LME speed FC):")
    summary = []
    for geno in order:
        rows_m = df[(df["genotype"] == geno) & df["speed_date_matched"]]
        rows_a = df[df["genotype"] == geno]
        rows   = rows_m if len(rows_m) > 0 else rows_a
        note   = "" if len(rows_m) > 0 else "no same-date N2"
        stat_r = stat_df[(stat_df["genotype"] == geno) & (stat_df["metric"] == "speed")]
        lme_fc = stat_r["fold_change"].iloc[0] if not stat_r.empty else np.nan
        lme_q  = stat_r["q"].iloc[0] if not stat_r.empty else np.nan
        summary.append({
            "genotype": geno, "n": len(rows), "note": note,
            "speed": rows["speed"].mean(),
            "rev_rate": rows["reversal_rate"].mean(),
            "speed_fc (LME)": lme_fc,
            "speed_q": lme_q,
        })
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(
        index=False,
        columns=["genotype", "n", "note", "speed", "rev_rate", "speed_fc (LME)", "speed_q"],
        float_format=lambda x: f"{x:.3f}",
        max_colwidth=22,
    ))


if __name__ == "__main__":
    main()
