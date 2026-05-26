"""
Locomotion analysis utilities for ODLabTracker postural tracking output.

Provides functions to load per-recording and per-particle data from tracks.csv
files produced by ODLabTracker, normalize to N2 controls, and fit linear
mixed-effects models via R/lme4 to estimate genotype effects.

Typical usage
-------------
    from ODLabTracker.locomotion import (
        scan_dataset_particles,
        scan_dataset_fwd_frames,
        add_normalization,
        load_exclusions,
        fit_lme_stats,
        genotype_order,
    )
"""

import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd


# ── per-recording loaders ─────────────────────────────────────────────────────

def _parse_tracks(results_dir, frame_rate, min_speed=0.0):
    """
    Parse a tracks.csv file and return per-particle summary DataFrame, or None.

    Columns returned: fwd_speed, reversal_rate, pirouette_rate,
                      all_frame_speed, n_frames, n_reversals, n_pirouettes,
                      duration_min.

    Particles with mean all-frame speed < min_speed are excluded.
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

    grp   = df.groupby("particle")
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
    """Return per-recording summary dict (speed, rates, counts), or None if unusable."""
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

    Only particles whose mean all-frame speed >= min_speed are included.
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
        per_p = df.groupby("particle")["speed"].mean()
        valid = per_p[per_p >= min_speed].index
        df    = df[df["particle"].isin(valid)]
    fwd = df[df["movement_type"] == "forward_run"][["frame", "particle", "speed"]].copy()
    return fwd if not fwd.empty else None


# ── dataset scanners ──────────────────────────────────────────────────────────

def _iter_recordings(data_dir):
    """Yield (genotype, date, filename, results_dir) for every AVI with a _results dir."""
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
    """
    Scan a dataset directory and return a DataFrame with one row per recording.

    Expected layout::

        data_dir/
            <genotype>/
                <YYYYMMDD>*.avi
                <YYYYMMDD>*_results/
                    tracks.csv
    """
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
    """
    Return per-particle DataFrame across the full dataset.

    Columns: fwd_speed, reversal_rate, pirouette_rate, genotype, date, recording_id.

    Parameters
    ----------
    exclusions : set of (genotype, date) tuples, optional
        Recordings to skip (e.g. from load_exclusions()).
    """
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
    Return per-forward-run-frame DataFrame for use as LME input.

    Columns: speed, genotype, date, recording_id, particle_uid.
    particle_uid is globally unique: ``<genotype>##<date>##p<particle_id>``.
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
        rec_id              = f"{geno}##{date}"
        fwd                 = fwd.copy()
        fwd["genotype"]     = geno
        fwd["date"]         = date
        fwd["recording_id"] = rec_id
        fwd["particle_uid"] = rec_id + "##p" + fwd["particle"].astype(str)
        chunks.append(fwd[["speed", "genotype", "date", "recording_id", "particle_uid"]])
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ── normalization ─────────────────────────────────────────────────────────────

_DEFAULT_METRICS = ("speed", "reversal_rate", "pirouette_rate")


def add_normalization(df, n2_ref="N2", metrics=_DEFAULT_METRICS):
    """
    Add fold-change columns to a per-recording DataFrame in place and return it.

    For each metric, adds:
        ``<metric>_norm``         — fold-change vs N2 (same-date or grand mean)
        ``<metric>_date_matched`` — True if a same-date N2 reference exists

    N2 recordings are normalized to the grand N2 mean so they scatter around 1.0.
    """
    n2_rows = df[df["genotype"] == n2_ref]
    if n2_rows.empty:
        sys.exit(f"No recordings found for reference genotype '{n2_ref}'.")

    n2_by_date = n2_rows.groupby("date")[list(metrics)].mean()
    n2_grand   = n2_rows[list(metrics)].mean()

    for metric in metrics:
        date_ref_map  = n2_by_date[metric].to_dict()
        grand_ref     = n2_grand[metric]
        same_date_ref = df["date"].map(date_ref_map)
        date_matched  = same_date_ref.notna()

        is_n2 = df["genotype"] == n2_ref
        same_date_ref[is_n2] = grand_ref
        date_matched[is_n2]  = True

        ref = same_date_ref.fillna(grand_ref)
        df[f"{metric}_norm"]         = df[metric] / ref
        df[f"{metric}_date_matched"] = date_matched

    return df


# ── exclusion loader ──────────────────────────────────────────────────────────

def load_exclusions(path):
    """
    Parse an exclusion CSV and return a set of (genotype, date) tuples.

    The file should have columns ``genotype,date`` (date as YYYYMMDD).
    Lines starting with ``#`` and a header row are ignored.
    Returns an empty set if path is None or does not exist.
    """
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


# ── statistics ────────────────────────────────────────────────────────────────

def _bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values as a numpy array."""
    pvals = np.asarray(pvals, dtype=float)
    n     = len(pvals)
    order = np.argsort(pvals)
    rank  = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    q      = pvals * n / rank
    q_sort = q[order]
    for i in range(n - 2, -1, -1):
        q_sort[i] = min(q_sort[i], q_sort[i + 1])
    q[order] = q_sort
    return np.clip(q, 0, 1)


def fit_lme_stats(frame_df, particle_df, order, metrics, out_dir, n2_ref="N2"):
    """
    Fit LME models via R/lme4 + lmerTest and return a tidy stats DataFrame.

    Models fitted
    -------------
    Speed (per-forward-run-frame)::

        speed ~ genotype + date + (1|recording_id) + (1|particle_uid)

    Reversal rate / pirouette rate (per-particle)::

        rate ~ genotype + date + (1|recording_id)

    ``date`` is a fixed effect to avoid REML shrinkage bias (see methods).
    Only recordings on dates where ``n2_ref`` was also measured are included
    (``keep_n2_dates`` filter in the R script). Genotypes without any same-date
    N2 receive NaN stats.

    P-values use Satterthwaite df (lmerTest); multiple comparisons corrected by
    Benjamini-Hochberg FDR per metric.

    Parameters
    ----------
    frame_df : DataFrame
        Per-forward-run-frame data with columns speed, genotype, date,
        recording_id, particle_uid. From scan_dataset_fwd_frames().
    particle_df : DataFrame
        Per-particle data with columns reversal_rate, pirouette_rate, genotype,
        date, recording_id. From scan_dataset_particles().
    order : list of str
        Genotype names in display order. Determines which genotypes are tested.
    metrics : list of str
        Subset of ["speed", "reversal_rate", "pirouette_rate"] to model.
    out_dir : str
        Directory for temporary R input/output files and the saved R script.
    n2_ref : str
        Name of the N2 reference genotype folder (default "N2").

    Returns
    -------
    DataFrame with columns: genotype, metric, n_recordings, n_particles,
        fold_change, fc_lo, fc_hi, p_raw, q, stars.
    """
    frame_csv     = os.path.join(out_dir, "_lme_fwd_frames.csv")
    particle_csv  = os.path.join(out_dir, "_lme_particles.csv")
    results_csv   = os.path.join(out_dir, "_lme_results.csv")
    r_script_path = os.path.join(out_dir, "_lme_fit.R")

    frame_df.to_csv(frame_csv, index=False)
    particle_df[["reversal_rate", "pirouette_rate",
                 "genotype", "date", "recording_id"]].to_csv(particle_csv, index=False)

    r_script = f"""
suppressMessages(library(lme4))
suppressMessages(library(lmerTest))

n2_ref <- "{n2_ref}"

extract_coefs <- function(fit, metric) {{
  s        <- as.data.frame(coef(summary(fit)))
  s$term   <- rownames(s)
  s$metric <- metric
  rownames(s) <- NULL
  colnames(s) <- c("estimate", "se", "df", "t_value", "p_value", "term", "metric")
  s
}}

# filter to dates where N2 was recorded (avoids REML shrinkage on fixed date)
keep_n2_dates <- function(df) {{
  n2_dates <- unique(df$date[df$genotype == n2_ref])
  df[df$date %in% n2_dates, ]
}}

# speed: per-forward-run-frame
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

# reversal rate: per-particle
cat("Fitting reversal rate model...\\n")
pdf        <- read.csv("{particle_csv}", stringsAsFactors = FALSE)
pdf_rev    <- keep_n2_dates(pdf[!is.na(pdf$reversal_rate), ])
pdf_rev$genotype <- relevel(droplevels(as.factor(pdf_rev$genotype)), ref = n2_ref)
pdf_rev$date     <- factor(pdf_rev$date)
rev_fit    <- lmer(reversal_rate ~ genotype + date + (1|recording_id),
                   data = pdf_rev, REML = TRUE,
                   control = lmerControl(optimizer = "bobyqa"))
cat("Reversal rate model done.\\n")

# pirouette rate: per-particle
cat("Fitting pirouette rate model...\\n")
pdf_pir    <- keep_n2_dates(pdf[!is.na(pdf$pirouette_rate), ])
pdf_pir$genotype <- relevel(droplevels(as.factor(pdf_pir$genotype)), ref = n2_ref)
pdf_pir$date     <- factor(pdf_pir$date)
pir_fit    <- lmer(pirouette_rate ~ genotype + date + (1|recording_id),
                   data = pdf_pir, REML = TRUE,
                   control = lmerControl(optimizer = "bobyqa"))
cat("Pirouette rate model done.\\n")

results <- rbind(
  extract_coefs(speed_fit,  "speed"),
  extract_coefs(rev_fit,    "reversal_rate"),
  extract_coefs(pir_fit,    "pirouette_rate")
)
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
    for line in proc.stdout.strip().splitlines():
        print(f"    [R] {line}")
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError("lme4 fitting failed — see R stderr above")

    lme4_df = pd.read_csv(results_csv)

    n_counts = particle_df.groupby("genotype")["recording_id"].agg(
        n_recs="nunique", n_part="count").reset_index()
    n_counts.columns = ["genotype", "n_recordings", "n_particles"]

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
            if geno == n2_ref:
                continue
            row    = geno_rows[geno_rows["genotype"] == geno]
            counts = n_counts[n_counts["genotype"] == geno]
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
            fc    = 1 + b        / n2_baseline
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
            t["stars"] = _sig_stars(q)
        all_rows.extend(tests)

    return pd.DataFrame(all_rows)


def _sig_stars(q):
    """Return significance asterisks string for a BH-corrected q-value."""
    if pd.isna(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


# ── genotype ordering ─────────────────────────────────────────────────────────

def genotype_order(df, stat_df=None, n2_ref="N2"):
    """
    Return genotype names sorted ascending by LME speed fold-change.

    With ``stat_df``: sorts by LME speed fold-change; genotypes without an LME
    estimate fall back to raw speed fold-change relative to N2.
    Without ``stat_df``: sorts by mean raw speed.
    """
    if stat_df is not None:
        speed_lme = (stat_df[stat_df["metric"] == "speed"]
                     .set_index("genotype")["fold_change"])
        n2_speed  = df[df["genotype"] == n2_ref]["speed"].mean()
        raw_fc    = df.groupby("genotype")["speed"].mean() / n2_speed

        def _key(g):
            if g == n2_ref:
                return 1.0
            fc = speed_lme.get(g, np.nan)
            return fc if not pd.isna(fc) else float(raw_fc.get(g, 1.0))

        return sorted(df["genotype"].unique(), key=_key)

    return (df.groupby("genotype")["speed"]
              .mean().sort_values(ascending=True)
              .index.tolist())
