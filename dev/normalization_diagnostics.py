"""
Verify two-pass illumination normalization across all N2 videos.

Shows the RAW median and p99 signals as the primary trace so oscillations
are visible, with corrected versions overlaid:

  Figure 1 — Median (background brightness):
    raw signal (fast flicker visible) + fast-corrected flat reference

  Figure 2 — p99 (worm-pixel brightness):
    raw p99 (faint) + slow envelope (dashed) + fully corrected (both passes,
    should be flat if normalization is working)

CV (std/mean) is shown before and after each correction stage.

Usage:
    python dev/normalization_diagnostics.py
"""

import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter1d

DATA_DIR = (
    "/Volumes/User Homes/ODlab-user/UserFolders/Nawaphat"
    "/6 CEST-2.1/Locomotion_Off food/0_COMPLETE!!/N2"
)
CONFIG  = "configs/IR_medium.yaml"
OUT_MED = "data/nawaphat_postural_results/29_normalization_median_timeseries.png"
OUT_P99 = "data/nawaphat_postural_results/30_normalization_p99_timeseries.png"
NCOLS   = 5


def date_label(path):
    m = re.search(r"(\d{8})", path)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d").strftime("%b %-d")
    return os.path.basename(path)


def stream_stats(video_path):
    cap = cv2.VideoCapture(video_path)
    medians, p99s, p90s = [], [], []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        medians.append(float(np.median(gray)))
        p99s.append(float(np.percentile(gray, 99)))
        p90s.append(float(np.percentile(gray, 90)))
    cap.release()
    return np.array(medians), np.array(p99s), np.array(p90s)


def compute_normalization(medians, p99s, p90s, frame_rate):
    n = len(medians)

    # Pass 1: per-frame median → scale_fast
    ref_fast   = float(np.median(medians))
    scale_fast = np.ones(n)
    if ref_fast > 1:
        for i in range(n):
            if medians[i] > 1:
                s = ref_fast / medians[i]
                if abs(s - 1.0) > 0.01:
                    scale_fast[i] = s

    med_corrected = medians * scale_fast

    # Pass 2: slow p99 drift → scale_slow
    p99s_after_fast = p99s * scale_fast
    use_p90 = np.std(p99s_after_fast) < 2.0
    bright_after_fast = (p90s if use_p90 else p99s) * scale_fast

    win    = max(3, int(frame_rate * 10))
    p_slow = uniform_filter1d(bright_after_fast, size=win, mode="reflect")
    ref_slow   = float(np.mean(p_slow))
    scale_slow = ref_slow / p_slow

    bright_final = bright_after_fast * scale_slow
    pct_label    = "p90" if use_p90 else "p99"

    return {
        "ref_fast":          ref_fast,
        "scale_fast":        scale_fast,
        "med_corrected":     med_corrected,
        "bright_raw":        p90s if use_p90 else p99s,          # raw (same pct used throughout)
        "bright_after_fast": bright_after_fast,
        "p_slow":            p_slow,
        "bright_final":      bright_final,
        "pct_label":         pct_label,
        "win":               win,
    }


def cv(x):
    m = np.mean(x)
    return np.std(x) / m if m > 0 else np.nan


def color_for(i, n):
    return plt.get_cmap("turbo")(i / max(n - 1, 1))


def make_grid(n, ncols):
    return int(np.ceil(n / ncols)), ncols


def _hide_extra(axes, used):
    for j in range(used, len(axes)):
        axes[j].set_visible(False)


# ── Figure 1: median timeseries ───────────────────────────────────────────

def plot_median_figure(records, out_path):
    n = len(records)
    nrows, ncols = make_grid(n, NCOLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.6), sharey=False)
    fig.suptitle(
        "Per-frame median (background brightness) — raw signal + fast-corrected\n"
        "High-frequency spikes = LED flicker; fast pass should flatten to dashed ref",
        fontsize=11
    )
    axes = np.array(axes).flatten()

    for i, (label, t, medians, stats) in enumerate(records):
        ax  = axes[i]
        col = color_for(i, n)

        cv_raw  = cv(medians)
        cv_corr = cv(stats["med_corrected"])

        # raw signal prominent; corrected as thin overlay
        ax.plot(t, medians,               color=col,     linewidth=0.6, alpha=0.9, label="raw")
        ax.plot(t, stats["med_corrected"], color="black", linewidth=0.6, alpha=0.5, label="fast-corrected")
        ax.axhline(stats["ref_fast"],      color="black", linewidth=0.8, linestyle="--", alpha=0.7)

        n_fast = int(np.sum(stats["scale_fast"] != 1.0))
        ax.set_title(
            f"{label}  ({n_fast}/{len(t)} corrected)\n"
            f"CV raw={cv_raw:.4f} → corrected={cv_corr:.4f}",
            fontsize=7.5
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("Median px", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right", handlelength=1)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    _hide_extra(axes, len(records))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


# ── Figure 2: p99/p90 timeseries ─────────────────────────────────────────

def plot_p99_figure(records, out_path):
    n = len(records)
    nrows, ncols = make_grid(n, NCOLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.6), sharey=False)
    fig.suptitle(
        "Per-frame bright-pixel percentile — raw vs fully corrected (both passes)\n"
        "Slow envelope shown as dashed; corrected should be flat if normalization works",
        fontsize=11
    )
    axes = np.array(axes).flatten()

    for i, (label, t, medians, stats) in enumerate(records):
        ax   = axes[i]
        col  = color_for(i, n)
        pct  = stats["pct_label"]

        cv_raw   = cv(stats["bright_raw"])
        cv_final = cv(stats["bright_final"])

        ax.plot(t, stats["bright_raw"],    color=col,     linewidth=0.5, alpha=0.4, label=f"{pct} raw")
        ax.plot(t, stats["p_slow"],        color="black", linewidth=1.0, linestyle="--", alpha=0.7, label=f"slow env (win={stats['win']}fr)")
        ax.plot(t, stats["bright_final"],  color=col,     linewidth=0.8, alpha=0.95, label="corrected")

        ax.set_title(
            f"{label}  [{pct}]\n"
            f"CV: raw={cv_raw:.4f} → final={cv_final:.4f}",
            fontsize=7.5
        )
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel(f"{pct} px", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, loc="upper right", handlelength=1)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    _hide_extra(axes, len(records))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    with open(CONFIG) as f:
        config = yaml.safe_load(f)
    frame_rate = config["frame_rate"]

    avis = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.avi"), recursive=True))
    avis = [a for a in avis if not os.path.basename(a).startswith("._")]
    print(f"Found {len(avis)} videos")

    records = []
    for avi in avis:
        label = date_label(avi)
        print(f"  {label} ...", end=" ", flush=True)
        medians, p99s, p90s = stream_stats(avi)
        t = np.arange(len(medians)) / frame_rate
        stats = compute_normalization(medians, p99s, p90s, frame_rate)
        print(f"{len(medians)} frames  {stats['pct_label']}  "
              f"CV: raw={cv(stats['bright_raw']):.4f} "
              f"fast={cv(stats['bright_after_fast']):.4f} "
              f"final={cv(stats['bright_final']):.4f}")
        records.append((label, t, medians, stats))

    plot_median_figure(records, OUT_MED)
    plot_p99_figure(records, OUT_P99)


if __name__ == "__main__":
    main()
