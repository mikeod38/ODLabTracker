"""
Per-video speed and area distribution plots for N2 off-food dataset.
Matches the style of dev/speed_distributions_post_normalization.png and
dev/area_distributions.png (plots 23 and 24).

Usage:
    python dev/qc_n2_distributions.py
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
SPEED_THRESHOLD = 0.03   # mm/s — from config
MIN_AREA        = 150    # px²  — from config
MAX_AREA        = 700    # px²  — from config
NCOLS           = 5


def date_label(path):
    m = re.search(r'(\d{8})', path)
    if m:
        dt = pd.to_datetime(m.group(1), format="%Y%m%d")
        return dt.strftime("%b %-d"), dt
    return os.path.basename(path), pd.NaT


def load_all(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "**", "tracks.csv"), recursive=True))
    records = []
    for csv in csvs:
        label, dt = date_label(csv)
        df = pd.read_csv(csv, low_memory=False)
        records.append((dt, label, df))
    records.sort(key=lambda r: r[0])
    return records


def make_grid(n, ncols):
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def color_for(i, n, cmap="turbo"):
    return plt.get_cmap(cmap)(i / max(n - 1, 1))


# ── Speed distributions ────────────────────────────────────────────────────

def plot_speed_distributions(records, out_path):
    n = len(records)
    nrows, ncols = make_grid(n, NCOLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8),
                             sharex=True, sharey=False)
    axes = np.array(axes).flatten()
    fig.suptitle("Per-particle speed distributions — N2 off-food", fontsize=13)

    for i, (dt, label, df) in enumerate(records):
        ax  = axes[i]
        col = color_for(i, n)

        # per-particle mean speed
        p_speed = df.groupby("particle")["speed"].mean()
        p_fwd   = p_speed[p_speed > SPEED_THRESHOLD]
        med     = p_speed.median()

        bins = np.linspace(0, 0.55, 30)
        ax.hist(p_speed, bins=bins, density=True, alpha=0.4, color=col)
        ax.hist(p_fwd,   bins=bins, density=True, alpha=0.8, color=col)
        ax.axvline(med,             color="black", linewidth=1.2, linestyle="--")
        ax.axvline(SPEED_THRESHOLD, color="red",   linewidth=0.8, linestyle=":")

        ax.set_title(label, fontsize=9)
        ax.legend(
            [f"all (n={len(p_speed)})", f"fwd (n={len(p_fwd)})", f"med={med:.3f}"],
            fontsize=6, loc="upper right", handlelength=0
        )
        ax.set_xlabel("Speed (mm/s)", fontsize=7)
        ax.set_ylabel("Density",      fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


# ── Area distributions ─────────────────────────────────────────────────────

def plot_area_distributions(records, out_path):
    n = len(records)
    nrows, ncols = make_grid(n, NCOLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8),
                             sharex=True, sharey=False)
    axes = np.array(axes).flatten()
    fig.suptitle(
        f"Per-detection area distributions — N2 off-food "
        f"(config bounds: {MIN_AREA}–{MAX_AREA} px²)",
        fontsize=13
    )

    for i, (dt, label, df) in enumerate(records):
        ax  = axes[i]
        col = color_for(i, n)
        med = df["area"].median()

        bins = np.linspace(0, 1200, 50)
        ax.hist(df["area"], bins=bins, density=True, color=col, alpha=0.7)
        ax.axvline(med,      color="black", linewidth=1.2, linestyle="--")
        ax.axvline(MIN_AREA, color="red",   linewidth=0.8, linestyle=":")
        ax.axvline(MAX_AREA, color="blue",  linewidth=0.8, linestyle=":")

        ax.set_title(label, fontsize=9)
        ax.legend(
            [f"med={med:.0f}", f"min_area={MIN_AREA}", f"max_area={MAX_AREA}"],
            fontsize=6, loc="upper right", handlelength=0
        )
        ax.set_xlabel("Detection area (px²)", fontsize=7)
        ax.set_ylabel("Density",              fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    records = load_all(DATA_DIR)
    print(f"Loaded {len(records)} videos")
    plot_speed_distributions(records, "dev/qc_n2_speed_distributions.png")
    plot_area_distributions(records,  "dev/qc_n2_area_distributions.png")


if __name__ == "__main__":
    main()
