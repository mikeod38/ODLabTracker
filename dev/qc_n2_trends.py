"""
QC plot: per-video speed, area, and track-length trends for N2 dataset.

Usage:
    python dev/qc_n2_trends.py
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

DATA_DIR = (
    "/Volumes/User Homes/ODlab-user/UserFolders/Nawaphat"
    "/6 CEST-2.1/Locomotion_Off food/0_COMPLETE!!/N2"
)
OUT_PATH = "data/nawaphat_postural_results/31_qc_n2_trends.png"


def date_from_path(path):
    # Search the full path — the results dir name carries the date, not tracks.csv itself.
    m = re.search(r'(\d{8})', path)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    return pd.NaT


def load_summaries(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "**", "tracks.csv"), recursive=True))
    rows = []
    for csv in csvs:
        dt = date_from_path(csv)
        df = pd.read_csv(csv)

        # per-particle summary
        grp = df.groupby("particle")
        track_lengths = grp["frame"].count()
        mean_speeds   = grp["speed"].mean()
        mean_areas    = grp["area"].mean()

        rows.append({
            "date":           dt,
            "n_particles":    len(track_lengths),
            "track_len_med":  track_lengths.median(),
            "track_len_mean": track_lengths.mean(),
            "speed_med":      mean_speeds.median(),
            "speed_mean":     mean_speeds.mean(),
            "area_med":       mean_areas.median(),
            "area_mean":      mean_areas.mean(),
            "area_cv_med":    df.groupby("particle")["area_cv"].first().median()
                              if "area_cv" in df.columns else np.nan,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main():
    summary = load_summaries(DATA_DIR)
    print(summary[["date", "n_particles", "track_len_med", "speed_med", "area_med"]].to_string(index=False))

    dates = summary["date"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("N2 off-food — per-video QC trends", fontsize=13)

    def plot_panel(ax, y, ylabel, color):
        ax.plot(dates, y, "o-", color=color, markersize=5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.axhline(y.median(), color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    plot_panel(axes[0], summary["track_len_med"],  "Median track length\n(frames)",     "steelblue")
    plot_panel(axes[1], summary["speed_med"],       "Median worm speed\n(mm/s)",          "darkorange")
    plot_panel(axes[2], summary["area_med"],        "Median worm area\n(px²)",            "seagreen")
    plot_panel(axes[3], summary["n_particles"],     "N worms detected",                   "mediumpurple")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    fig.autofmt_xdate(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
