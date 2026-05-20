"""
Zoom calibration for Nawaphat locomotion dataset.

Measures the effective zoom level for each recording session using N2 worm
major axis as an internal standard (C. elegans body length is constant under
controlled conditions).  Outputs a calibration table with the corrected
pixel_length and search_range for each date, ready to use when re-tracking.

Approach
--------
1. For every N2 recording, compute median major axis across all particles.
2. Define the reference pixel_length as the value that minimises variance in
   N2 major axis across sessions (i.e. the median-of-medians day).
3. For each date (N2 and mutant), compute:
       zoom_factor = N2_major_px / reference_N2_major_px
       pixel_length_cal = pixel_length_config × zoom_factor
       search_range_cal = round(search_range_config × zoom_factor)
4. Dates without a same-day N2 get the nearest-date calibration.

Outputs
-------
  <out-dir>/zoom_calibration.csv   per-date calibration table
  <out-dir>/zoom_calibration.png   N2 major axis vs date (zoom timeline)

Usage
-----
  python dev/zoom_calibration.py
  python dev/zoom_calibration.py --pixel-length 60 --search-range 30
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = (
    "/Volumes/User Homes/ODlab-user/UserFolders/Nawaphat"
    "/6 CEST-2.1/Locomotion_Off food/0_COMPLETE!!"
)
OUT_DIR   = "data/nawaphat_postural_results"
N2_FOLDER = "N2"

PIXEL_LENGTH_CONFIG  = 60    # px/mm from IR_medium.yaml
SEARCH_RANGE_CONFIG  = 30    # px from IR_medium.yaml
MIN_PARTICLES        = 10    # minimum N2 particles to trust a date's calibration


def n2_major_by_date(data_dir):
    """Return dict date_str → median N2 major axis (px) across all particles."""
    gdir = os.path.join(data_dir, N2_FOLDER)
    result = {}
    for fname in sorted(os.listdir(gdir)):
        if not fname.endswith(".avi") or fname.startswith("._"):
            continue
        m = re.match(r"^(\d{8})", fname)
        if not m:
            continue
        date = m.group(1)
        rdir = os.path.join(gdir, os.path.splitext(fname)[0] + "_results")
        csv  = os.path.join(rdir, "tracks.csv")
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv, usecols=["particle", "major_axis"])
        except Exception:
            continue
        if df.empty:
            continue
        per_p = df.groupby("particle")["major_axis"].mean()
        if len(per_p) < MIN_PARTICLES:
            continue
        result[date] = float(per_p.median())
    return result


def nearest_date_cal(date_str, cal_map):
    """Return the calibration factor for the nearest available date."""
    if date_str in cal_map:
        return cal_map[date_str]
    date_int = int(date_str)
    nearest = min(cal_map.keys(), key=lambda d: abs(int(d) - date_int))
    return cal_map[nearest]


def main():
    parser = argparse.ArgumentParser(description="Zoom calibration from N2 worm major axis")
    parser.add_argument("--data-dir",      default=DATA_DIR)
    parser.add_argument("--out-dir",       default=OUT_DIR)
    parser.add_argument("--pixel-length",  type=float, default=PIXEL_LENGTH_CONFIG,
                        help="Nominal pixel_length in config (px/mm)")
    parser.add_argument("--search-range",  type=int,   default=SEARCH_RANGE_CONFIG,
                        help="Nominal search_range in config (px)")
    parser.add_argument("--min-particles", type=int,   default=MIN_PARTICLES)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── N2 major axis per date ────────────────────────────────────────────────
    n2_major = n2_major_by_date(args.data_dir)
    if not n2_major:
        sys.exit("No N2 tracks.csv found — has the dataset been tracked?")

    dates = sorted(n2_major.keys())
    majors = [n2_major[d] for d in dates]

    # Reference: median of all N2 major axes (robust to outlier sessions)
    ref_major = float(np.median(majors))
    print(f"N2 major axis reference (median of {len(dates)} sessions): {ref_major:.2f} px")

    # ── Per-date calibration ──────────────────────────────────────────────────
    cal_map = {}   # date_str → zoom_factor
    for date in dates:
        zoom = n2_major[date] / ref_major
        cal_map[date] = zoom

    # ── All recording dates in the dataset ───────────────────────────────────
    all_dates = set()
    for geno in sorted(os.listdir(args.data_dir)):
        gdir = os.path.join(args.data_dir, geno)
        if not os.path.isdir(gdir) or geno.startswith("._"):
            continue
        for fname in sorted(os.listdir(gdir)):
            if not fname.endswith(".avi") or fname.startswith("._"):
                continue
            m = re.match(r"^(\d{8})", fname)
            if m:
                all_dates.add(m.group(1))

    # ── Build calibration table ───────────────────────────────────────────────
    rows = []
    for date in sorted(all_dates):
        zoom    = nearest_date_cal(date, cal_map)
        has_n2  = date in cal_map
        pl_cal  = args.pixel_length * zoom
        sr_cal  = max(1, round(args.search_range * zoom))
        rows.append({
            "date":              date,
            "has_n2":            has_n2,
            "n2_major_px":       n2_major.get(date, np.nan),
            "zoom_factor":       zoom,
            "pixel_length_cal":  round(pl_cal, 2),
            "search_range_cal":  sr_cal,
            "deviation_pct":     round((zoom - 1.0) * 100, 1),
        })

    cal_df = pd.DataFrame(rows)

    csv_out = os.path.join(args.out_dir, "zoom_calibration.csv")
    cal_df.to_csv(csv_out, index=False, float_format="%.3f")
    print(f"\nCalibration table saved: {csv_out}")
    print(f"\n{'Date':>10}  {'N2 maj':>8}  {'Zoom':>6}  {'px/mm':>7}  {'srng':>5}  {'dev%':>6}  {'src':>8}")
    for _, r in cal_df.iterrows():
        src = "N2 direct" if r["has_n2"] else "nearest N2"
        maj = f"{r['n2_major_px']:.1f}" if not np.isnan(r['n2_major_px']) else "  —  "
        print(f"{r['date']:>10}  {maj:>8}  {r['zoom_factor']:>6.3f}  "
              f"{r['pixel_length_cal']:>7.1f}  {r['search_range_cal']:>5d}  "
              f"{r['deviation_pct']:>+6.1f}%  {src:>10}")

    # ── Sessions needing re-tracking (>5% zoom deviation) ────────────────────
    needs_retrack = cal_df[cal_df["deviation_pct"].abs() > 5]
    print(f"\nSessions with >5% zoom deviation (need re-tracking): {len(needs_retrack)}")
    for _, r in needs_retrack.iterrows():
        print(f"  {r['date']}  zoom={r['zoom_factor']:.3f}  "
              f"pixel_length={r['pixel_length_cal']:.1f}  search_range={r['search_range_cal']}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    n2_dates = cal_df[cal_df["has_n2"]]
    ax.scatter(n2_dates["date"], n2_dates["n2_major_px"],
               color="black", s=60, zorder=5, label="N2 (direct)")
    ax.axhline(ref_major, color="gray", ls="--", lw=1.2, label=f"reference={ref_major:.1f} px")
    ax.axhline(ref_major * 1.05, color="red", ls=":", lw=0.8, label="±5% (re-track threshold)")
    ax.axhline(ref_major * 0.95, color="red", ls=":", lw=0.8)
    ax.set_xlabel("Recording date", fontsize=9)
    ax.set_ylabel("N2 median major axis (px)", fontsize=9)
    ax.set_title("Zoom calibration — N2 worm major axis by session", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "zoom_calibration.png"), dpi=130, bbox_inches="tight")
    print(f"Plot saved: {os.path.join(args.out_dir, 'zoom_calibration.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
