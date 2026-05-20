"""
Compare per-particle-mean vs frame-pooled speed estimates by date,
and plot track length vs speed per video to check for length-speed bias.

Figure 1: per-date comparison of the two metrics
Figure 2: grid of track-length vs per-particle-mean-speed scatterplots

Usage:
    python dev/qc_speed_weighting.py
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = (
    "/Volumes/User Homes/ODlab-user/UserFolders/Nawaphat"
    "/6 CEST-2.1/Locomotion_Off food/0_COMPLETE!!/N2"
)
OUT_COMPARISON = "data/nawaphat_postural_results/35_speed_weighting_comparison.png"
OUT_SCATTER    = "data/nawaphat_postural_results/36_track_length_vs_speed.png"

EXCLUDE_DATES = {"20260120"}
NCOLS = 5


def date_from_path(path):
    m = re.search(r"(\d{8})", path)
    return m.group(1) if m else None


def date_label(date_str):
    return pd.to_datetime(date_str, format="%Y%m%d").strftime("%b %-d")


def load_all(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "**", "tracks.csv"), recursive=True))
    records = []
    for csv in csvs:
        d = date_from_path(csv)
        if d in EXCLUDE_DATES:
            continue
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        if not {"frame", "speed", "particle"}.issubset(df.columns):
            continue
        records.append((d, df))
    return records


# ── Figure 1: per-date metric comparison ─────────────────────────────────────

def plot_comparison(records, out_path):
    rows = []
    for d, df in records:
        grp = df.groupby("particle")
        track_lengths    = grp["frame"].count()
        per_part_med     = grp["speed"].median()
        per_part_max     = grp["speed"].max()
        per_part_min     = grp["speed"].min()
        rows.append({
            "date":          d,
            "label":         date_label(d),
            "unweighted":    per_part_med.median(),
            "weighted":      df["speed"].median(),
            "med_of_max":    per_part_max.median(),
            "med_of_min":    per_part_min.median(),
            "n_particles":   len(track_lengths),
            "track_len_med": track_lengths.median(),
        })
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Speed by date: per-particle median vs frame-pooled median\n"
                 "Shaded band = median of per-particle min/max (spread across particles)",
                 fontsize=11)

    x = np.arange(len(summary))
    labels = summary["label"].tolist()

    ax = axes[0]
    ax.fill_between(x, summary["med_of_min"] * 1000, summary["med_of_max"] * 1000,
                    alpha=0.15, color="steelblue", label="median of per-particle min/max")
    ax.plot(x, summary["unweighted"] * 1000, "o-", color="steelblue", linewidth=1.5,
            markersize=5, label="per-particle median → video median")
    ax.plot(x, summary["weighted"] * 1000, "s--", color="darkorange", linewidth=1.5,
            markersize=5, label="frame-pooled median (duration-weighted)")
    ax.set_ylabel("Speed (µm/s)")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Difference as % of unweighted
    diff_pct = (summary["weighted"] - summary["unweighted"]) / summary["unweighted"] * 100
    ax2 = axes[1]
    colors = ["tomato" if v > 0 else "steelblue" for v in diff_pct]
    ax2.bar(x, diff_pct, color=colors, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Difference\n(weighted − unweighted, %)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.grid(True, linewidth=0.3, alpha=0.4, axis="y")

    # Annotate n_particles on top panel
    for xi, (n, tl) in enumerate(zip(summary["n_particles"], summary["track_len_med"])):
        axes[0].annotate(f"n={n}\ntl={tl:.0f}", (xi, summary["unweighted"].iloc[xi] * 1000),
                         textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=5.5, color="gray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


# ── Figure 2: per-video track length vs speed scatter ────────────────────────

def plot_length_vs_speed(records, out_path):
    n = len(records)
    nrows = int(np.ceil(n / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 3, nrows * 2.6), sharey=False)
    fig.suptitle("Track length vs per-particle speed (median, with min/max range) — per video\n"
                 "Dots = median; bars = min–max per particle. Binned median trend in black.",
                 fontsize=11)
    axes = np.array(axes).flatten()
    cmap = plt.get_cmap("turbo")

    for i, (d, df) in enumerate(records):
        ax = axes[i]
        col = cmap(i / max(n - 1, 1))
        grp = df.groupby("particle")
        lengths  = grp["frame"].count()
        sp_med   = grp["speed"].median()
        sp_max   = grp["speed"].max()
        sp_min   = grp["speed"].min()

        lvals = lengths.values
        svals = sp_med.values * 1000
        err_lo = (sp_med - sp_min).values * 1000
        err_hi = (sp_max - sp_med).values * 1000

        ax.errorbar(lvals, svals, yerr=[err_lo, err_hi],
                    fmt="none", ecolor=col, alpha=0.25, linewidth=0.5)
        ax.scatter(lvals, svals, s=8, alpha=0.7, color=col, linewidths=0, zorder=3)

        # Binned median of per-particle medians
        bins = np.percentile(lvals, np.linspace(0, 100, 7))
        bins = np.unique(bins)
        if len(bins) > 2:
            centers, meds = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (lvals >= lo) & (lvals < hi)
                if mask.sum() >= 3:
                    centers.append((lo + hi) / 2)
                    meds.append(np.median(svals[mask]))
            if len(centers) > 1:
                ax.plot(centers, meds, color="black", linewidth=1.2, alpha=0.8)

        r = np.corrcoef(lvals, svals)[0, 1] if len(lvals) > 2 else np.nan
        ax.set_title(f"{date_label(d)}  n={len(lengths)}\nr={r:.2f}", fontsize=7.5)
        ax.set_xlabel("Track length (frames)", fontsize=7)
        ax.set_ylabel("Median speed (µm/s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    for j in range(len(records), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    records = load_all(DATA_DIR)
    print(f"Loaded {len(records)} videos")
    plot_comparison(records, OUT_COMPARISON)
    plot_length_vs_speed(records, OUT_SCATTER)


if __name__ == "__main__":
    main()
