"""
Batch postural comparison for Nawaphat's 0_COMPLETE!! locomotion dataset.

For each _results/tracks.csv, computes per-particle rates then averages to a
single per-recording mean. Normalizes to same-date N2 (fold-change). Recordings
without a same-date N2 are normalized to the grand-mean N2 and plotted with open
symbols — they are included visually but excluded from genotype mean ± SEM.

Outputs:
    <out-dir>/postural_comparison.csv   per-recording summary table
    <out-dir>/postural_comparison.png   horizontal strip plot (3 panels)

Exclusion file (--exclude):
    CSV with columns genotype,date (YYYYMMDD). Matching recordings are dropped
    before normalization and plotting. Lines starting with # are ignored.
    Example:
        genotype,date
        bas-1,20260207
        bas-1,20260307

Usage:
    python dev/batch_postural_comparison.py
    python dev/batch_postural_comparison.py --exclude data/nawaphat_postural_results/exclude.csv
    python dev/batch_postural_comparison.py --out-dir results/nawaphat
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DATA_DIR = (
    "/Volumes/User Homes/ODlab-user/UserFolders/Nawaphat"
    "/6 CEST-2.1/Locomotion_Off food/0_COMPLETE!!"
)
OUT_DIR = "data/nawaphat_postural_results"
FRAME_RATE = 10   # fps, from IR_medium.yaml
N2_FOLDER = "N2"

METRICS = ["speed", "reversal_rate", "pirouette_rate"]
METRIC_LABELS = {
    "speed":          "Speed\n(fold-change vs N2)",
    "reversal_rate":  "Reversal rate\n(fold-change vs N2)",
    "pirouette_rate": "Pirouette rate\n(fold-change vs N2)",
}


# ── per-recording loader ─────────────────────────────────────────────────────

def load_recording(results_dir, frame_rate, min_speed=0.0):
    """
    Read tracks.csv and return per-recording stats, or None if unusable.

    Rates are computed per particle (events / track-duration-in-minutes) then
    averaged across particles, so short tracks don't dilute long ones.

    Particles with mean_speed < min_speed are excluded before aggregation to
    remove injured/stationary worms that bias the mean and inflate reversal
    rates through centroid jitter.  Speed is reported as the median across
    particles (robust to the remaining speed distribution tail).
    """
    csv_path = os.path.join(results_dir, "tracks.csv")
    if not os.path.exists(csv_path):
        return None

    needed = {"frame", "particle", "speed", "movement_type", "reversal_start", "pirouette_start"}
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

    # Forward-run speed: median of speed during forward_run frames per particle.
    # Using all-frame mean for the slow-worm filter so injured/stationary
    # particles are still caught even if they have no forward_run frames.
    if "movement_type" in df.columns:
        fwd = df[df["movement_type"] == "forward_run"]
        fwd_speed = fwd.groupby("particle")["speed"].median()
        per_p["fwd_speed"] = fwd_speed
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

    n_total = len(per_p)

    # Drop barely-moving particles (injured/stationary — jitter inflates reversal rate)
    if min_speed > 0:
        per_p = per_p[per_p["all_frame_speed"] >= min_speed]

    n = len(per_p)
    if n == 0:
        return None

    return {
        "speed":              per_p["fwd_speed"].median(),
        "reversal_rate":      per_p["reversal_rate"].mean(),
        "pirouette_rate":     per_p["pirouette_rate"].mean(),
        "n_reversals":        int(per_p["n_reversals"].sum()),
        "n_pirouettes":       int(per_p["n_pirouettes"].sum()),
        "n_particles":        n,
        "n_excluded":         n_total - n,
    }


# ── dataset scanner ──────────────────────────────────────────────────────────

def scan_dataset(data_dir, frame_rate, min_speed=0.0):
    """Return DataFrame with one row per successfully loaded recording."""
    rows = []
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
            date = m.group(1)
            stem = os.path.splitext(fname)[0]
            results_dir = os.path.join(gdir, stem + "_results")
            stats = load_recording(results_dir, frame_rate, min_speed=min_speed)
            if stats is None:
                print(f"  Skipping {geno}/{fname} (no tracks.csv or empty)")
                continue
            rows.append({"genotype": geno, "date": date,
                         "results_dir": results_dir, **stats})

    return pd.DataFrame(rows)


# ── normalization ────────────────────────────────────────────────────────────

def add_normalization(df):
    """
    For each metric add <metric>_norm (fold-change vs N2) and <metric>_date_matched
    (True = same-date N2 used; False = grand-mean N2 used).

    N2 recordings are normalized to the grand N2 mean so their scatter reflects
    actual day-to-day variability rather than collapsing to 1.0.
    """
    n2_rows = df[df["genotype"] == N2_FOLDER]
    if n2_rows.empty:
        sys.exit(f"No recordings found for reference genotype '{N2_FOLDER}'.")

    n2_by_date = n2_rows.groupby("date")[METRICS].mean()
    n2_grand   = n2_rows[METRICS].mean()

    for metric in METRICS:
        date_ref_map = n2_by_date[metric].to_dict()
        grand_ref    = n2_grand[metric]

        # Map each recording's date to its N2 reference
        same_date_ref = df["date"].map(date_ref_map)           # NaN where no same-date N2
        date_matched  = same_date_ref.notna()

        # N2 rows: always use grand mean so they scatter around 1.0
        is_n2 = df["genotype"] == N2_FOLDER
        same_date_ref[is_n2]  = grand_ref
        date_matched[is_n2]   = True   # treat as "matched" for mean/SEM inclusion

        # Fill remaining NaN (non-N2 without same-date N2) with grand mean
        ref = same_date_ref.fillna(grand_ref)

        df[f"{metric}_norm"]         = df[metric] / ref
        df[f"{metric}_date_matched"] = date_matched

    return df


# ── genotype sort order ──────────────────────────────────────────────────────

def genotype_order(df):
    """
    Sort mutants by mean normalized speed (date-matched recordings only), slowest first.
    N2 placed at top. Genotypes with no date-matched recordings sorted by raw
    speed and placed at the bottom of the mutant list.
    """
    mutants = df[df["genotype"] != N2_FOLDER]

    matched = mutants[mutants["speed_date_matched"]]
    unmatched_only = mutants[~mutants["genotype"].isin(matched["genotype"].unique())]

    matched_order = (matched.groupby("genotype")["speed_norm"]
                     .mean().sort_values().index.tolist())

    unmatched_order = (unmatched_only.groupby("genotype")["speed"]
                       .mean().sort_values().index.tolist())

    return unmatched_order + matched_order + [N2_FOLDER]


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


# ── figure ───────────────────────────────────────────────────────────────────

# Individual recording dots
DOT_MATCHED   = "#4393c3"   # steel blue  — same-date N2, filled
DOT_UNMATCHED = "#d6604d"   # coral       — grand-mean N2, open
DOT_N2        = "#999999"   # grey        — N2 recordings, filled

# Genotype summary diamonds + SEM bars (visually distinct from dots)
DIAMOND_MUT   = "#b2182b"   # dark red
DIAMOND_N2    = "#111111"   # near-black


def make_plot(df, order, out_path):
    ytick = {g: i for i, g in enumerate(order)}
    n_geno = len(order)
    fig_h  = max(8, n_geno * 0.45)

    fig, axes = plt.subplots(1, 3, figsize=(15, fig_h), sharey=True)
    fig.suptitle("Nawaphat locomotion off food — postural comparison (fold-change vs N2)",
                 fontsize=12, y=1.01)

    # Precompute per-genotype event totals for annotation
    event_cols = {"reversal_rate": ("n_reversals", "n_particles"),
                  "pirouette_rate": ("n_pirouettes", "n_particles")}

    for ax, metric in zip(axes, METRICS):
        norm_col    = f"{metric}_norm"
        matched_col = f"{metric}_date_matched"

        # Individual recording dots
        for _, row in df.iterrows():
            y = ytick[row["genotype"]]
            x = row[norm_col]
            if pd.isna(x):
                continue
            is_n2   = row["genotype"] == N2_FOLDER
            matched = row[matched_col]
            if is_n2:
                ec, fc = DOT_N2, DOT_N2
            elif matched:
                ec, fc = DOT_MATCHED, DOT_MATCHED
            else:
                ec, fc = DOT_UNMATCHED, "none"
            ax.plot(x, y, "o", mfc=fc, mec=ec, ms=5, alpha=0.65, lw=0, zorder=2)

        # Genotype mean ± SEM (date-matched only) — diamonds in a contrasting colour
        for geno in order:
            y    = ytick[geno]
            rows = df[(df["genotype"] == geno) & df[matched_col]]
            vals = rows[norm_col].dropna()
            if len(vals) == 0:
                continue
            mean = vals.mean()
            sem  = vals.sem() if len(vals) > 1 else 0.0
            dc   = DIAMOND_N2 if geno == N2_FOLDER else DIAMOND_MUT
            ax.plot([mean - sem, mean + sem], [y, y],
                    color=dc, lw=2.5, solid_capstyle="round", zorder=4)
            ax.plot(mean, y, "D", color=dc, ms=7, zorder=5,
                    mec="white", mew=0.5)

        # Annotate reversal and pirouette panels with n_events / n_worms per genotype
        if metric in event_cols:
            ev_col, n_col = event_cols[metric]
            for geno in order:
                y      = ytick[geno]
                g_rows = df[df["genotype"] == geno]
                n_ev   = int(g_rows[ev_col].sum())
                n_worm = int(g_rows[n_col].sum())
                ax.text(1.01, y, f"{n_ev}/{n_worm}",
                        transform=ax.get_yaxis_transform(),
                        fontsize=5.5, va="center", ha="left", color="#555555")

        ax.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=0)
        ax.set_xlabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_ylim(-0.8, n_geno - 0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

    axes[0].set_yticks(list(ytick.values()))
    axes[0].set_yticklabels(list(ytick.keys()), fontsize=8)

    leg_handles = [
        mlines.Line2D([], [], color=DOT_MATCHED, marker="o", ls="none",
                      label="Recording — same-date N2"),
        mlines.Line2D([], [], color=DOT_UNMATCHED, marker="o", mfc="none", ls="none",
                      label="Recording — grand-mean N2 (no same-date N2)"),
        mlines.Line2D([], [], color=DOT_N2, marker="o", ls="none",
                      label="N2 recording (vs own grand mean)"),
        mlines.Line2D([], [], color=DIAMOND_MUT, marker="D", ls="none",
                      mec="white", mew=0.5, label="Genotype mean ± SEM"),
        mlines.Line2D([], [], color=DIAMOND_N2, marker="D", ls="none",
                      mec="white", mew=0.5, label="N2 mean ± SEM"),
    ]
    axes[-1].legend(handles=leg_handles, fontsize=7.5, loc="lower right",
                    framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch postural comparison for Nawaphat locomotion dataset")
    parser.add_argument("--data-dir",   default=DATA_DIR)
    parser.add_argument("--out-dir",    default=OUT_DIR)
    parser.add_argument("--frame-rate", type=float, default=FRAME_RATE)
    parser.add_argument("--exclude",    default=None,
                        help="CSV file with genotype,date rows to censor")
    parser.add_argument("--min-speed",  type=float, default=0.03,
                        help="Exclude particles with mean speed below this (mm/s). "
                             "Removes injured/stationary worms. Default: 0.03")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    exclusions = load_exclusions(args.exclude)
    if exclusions:
        print(f"Exclusions loaded ({len(exclusions)}):")
        for g, d in sorted(exclusions):
            print(f"  {g} / {d}")

    print(f"Scanning dataset… (min_speed filter: {args.min_speed:.3f} mm/s)")
    df = scan_dataset(args.data_dir, args.frame_rate, min_speed=args.min_speed)

    if exclusions:
        before = len(df)
        df = df[~df.apply(lambda r: (r["genotype"], r["date"]) in exclusions, axis=1)]
        print(f"Dropped {before - len(df)} excluded recording(s)")
    print(f"Loaded {len(df)} recordings, {df['genotype'].nunique()} genotypes, "
          f"{df['n_particles'].sum():.0f} total particles")

    df = add_normalization(df)

    # Print N2 date-to-date variability as a sanity check
    n2 = df[df["genotype"] == N2_FOLDER]
    print("\nN2 grand means (raw):")
    for m in METRICS:
        print(f"  {m}: {n2[m].mean():.3f}  (CV = {n2[m].std()/n2[m].mean()*100:.0f}%)")

    # Save CSV
    csv_out = os.path.join(args.out_dir, "postural_comparison.csv")
    col_order = (["genotype", "date", "n_particles",
                  "n_reversals", "n_pirouettes", "n_excluded"]
                 + METRICS
                 + [f"{m}_norm" for m in METRICS]
                 + [f"{m}_date_matched" for m in METRICS]
                 + ["results_dir"])
    df[col_order].to_csv(csv_out, index=False, float_format="%.4f")
    print(f"\nSaved summary CSV: {csv_out}")

    order = genotype_order(df)
    fig_out = os.path.join(args.out_dir, "postural_comparison.png")
    make_plot(df, order, fig_out)

    # Print quick text summary sorted by normalized reversal rate
    print("\nGenotype summary (date-matched recordings only, sorted by reversal rate):")
    summary = []
    for geno in order:
        rows = df[(df["genotype"] == geno) & df["reversal_rate_date_matched"]]
        n = len(rows)
        if n == 0:
            rows_all = df[df["genotype"] == geno]
            summary.append({
                "genotype": geno, "n": len(rows_all), "note": "no same-date N2",
                "speed": rows_all["speed"].mean(),
                "rev_rate": rows_all["reversal_rate"].mean(),
                "pir_rate": rows_all["pirouette_rate"].mean(),
                "speed_fc": rows_all["speed_norm"].mean(),
                "rev_fc": rows_all["reversal_rate_norm"].mean(),
                "pir_fc": rows_all["pirouette_rate_norm"].mean(),
            })
        else:
            summary.append({
                "genotype": geno, "n": n, "note": "",
                "speed": rows["speed"].mean(),
                "rev_rate": rows["reversal_rate"].mean(),
                "pir_rate": rows["pirouette_rate"].mean(),
                "speed_fc": rows["speed_norm"].mean(),
                "rev_fc": rows["reversal_rate_norm"].mean(),
                "pir_fc": rows["pirouette_rate_norm"].mean(),
            })
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(
        index=False,
        columns=["genotype", "n", "note", "speed", "rev_rate", "pir_rate",
                 "speed_fc", "rev_fc", "pir_fc"],
        float_format=lambda x: f"{x:.3f}",
        max_colwidth=22,
    ))


if __name__ == "__main__":
    main()
