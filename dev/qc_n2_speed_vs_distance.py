"""
Speed vs distance from frame center for N2 off-food dataset.

Per particle: median centroid distance from frame center (mm) vs mean speed (mm/s).
Scatter of individual particles + binned median with IQR ribbon.
All N2 videos pooled; Jan 20 excluded (ring artifact).

Usage:
    python dev/qc_n2_speed_vs_distance.py
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
OUT_PATH  = "data/nawaphat_postural_results/34_speed_vs_distance_from_center.png"

FRAME_CX     = 1280 / 2   # pixels
FRAME_CY     = 1024 / 2
PIXEL_PER_MM = 60.0

EXCLUDE_DATES = {"20260120"}   # ring artifact


def date_from_path(path):
    m = re.search(r"(\d{8})", path)
    return m.group(1) if m else None


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
        if not {"x", "y", "particle", "mean_speed"}.issubset(df.columns):
            continue

        # Per-particle: median centroid → distance from center in mm
        grp = df.groupby("particle")
        med_x = grp["x"].median()
        med_y = grp["y"].median()
        dist_px = np.sqrt((med_x - FRAME_CX) ** 2 + (med_y - FRAME_CY) ** 2)
        dist_mm = dist_px / PIXEL_PER_MM
        speed   = grp["speed"].median()   # median instantaneous speed across all frames

        part_df = pd.DataFrame({"dist_mm": dist_mm, "speed": speed})
        part_df["date"] = d
        records.append(part_df)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def binned_median_iqr(dist, speed, n_bins=20):
    bins = np.linspace(dist.min(), dist.max(), n_bins + 1)
    centers, medians, q25s, q75s = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (dist >= lo) & (dist < hi)
        if mask.sum() < 5:
            continue
        s = speed[mask]
        centers.append((lo + hi) / 2)
        medians.append(np.median(s))
        q25s.append(np.percentile(s, 25))
        q75s.append(np.percentile(s, 75))
    return np.array(centers), np.array(medians), np.array(q25s), np.array(q75s)


def main():
    df = load_all(DATA_DIR)
    print(f"Loaded {len(df)} particles from {df['date'].nunique()} videos")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["dist_mm"], df["speed"], s=4, alpha=0.15, color="steelblue", linewidths=0)

    centers, meds, q25s, q75s = binned_median_iqr(df["dist_mm"].values, df["speed"].values)
    ax.plot(centers, meds, color="black", linewidth=2, label="binned median")
    ax.fill_between(centers, q25s, q75s, color="black", alpha=0.15, label="IQR")

    # Max inscribed radius (distance to nearest edge)
    max_r = min(FRAME_CX, FRAME_CY) / PIXEL_PER_MM
    ax.axvline(max_r, color="red", linewidth=1, linestyle="--", alpha=0.7,
               label=f"nearest edge ({max_r:.1f} mm)")

    ax.set_xlabel("Distance from frame center (mm)")
    ax.set_ylabel("Median instantaneous speed (mm/s)")
    ax.set_title(f"N2 off-food: speed vs distance from center\n"
                 f"({df['date'].nunique()} videos, {len(df)} particles, Jan 20 excluded)")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
