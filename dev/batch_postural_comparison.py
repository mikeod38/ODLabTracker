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


from ODLabTracker.locomotion import (
    load_recording,
    load_recording_particles,
    scan_dataset,
    scan_dataset_particles,
    scan_dataset_fwd_frames,
    add_normalization,
    load_exclusions,
    fit_lme_stats,
    genotype_order,
)


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
        1, 2, sharey=True, figsize=(8.5, 11),
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
