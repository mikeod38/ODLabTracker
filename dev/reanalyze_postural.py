"""
Re-run postural analysis (speed params + reversal/pirouette detection) on an
existing tracks.csv without re-tracking from video.

Usage:
    python dev/reanalyze_postural.py <results_dir> [--config <yaml>]

The raw positional columns (frame, x, y, area, major_axis, minor_axis,
orientation, eccentricity, …, particle) are preserved; all computed columns
are rebuilt with the current tracking.py code.

Writes an updated tracks.csv in-place after printing a summary.
"""

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ODLabTracker import tracking

RAW_COLS = [
    'frame', 'x', 'y', 'area', 'major_axis', 'minor_axis', 'orientation',
    'eccentricity', 'euler_num', 'solidity', 'area_convex',
    'mean_intensity', 'median_intensity', 'max_intensity',
    'integrated_intensity', 'particle',
]


def main():
    parser = argparse.ArgumentParser(description="Re-run postural analysis on existing tracks.csv")
    parser.add_argument('results_dir', help="Path to *_results directory containing tracks.csv")
    parser.add_argument('--config', default='configs/IR_medium.yaml',
                        help="YAML config path (default: configs/IR_medium.yaml)")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print summary but do not overwrite tracks.csv")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, 'tracks.csv')
    if not os.path.exists(csv_path):
        sys.exit(f"tracks.csv not found in {args.results_dir}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(csv_path)
    available_raw = [c for c in RAW_COLS if c in df.columns]
    df = df[available_raw].copy()
    print(f"Loaded {len(df)} rows, {df['particle'].nunique()} particles from {csv_path}")

    df = tracking.calculate_speed_parameters(
        df,
        pixel_length=cfg['pixel_length'],
        frame_rate=cfg['frame_rate'],
        window_size=cfg['window_size'],
        speed_threshold=cfg['speed_threshold'],
        smooth_window=cfg.get('smooth_window', 5),
        min_displacement_for_angle=cfg.get('min_displacement_for_angle', 0.001),
        max_instantaneous_speed=cfg.get('max_instantaneous_speed', 0.6),
        stability_threshold=cfg.get('stability_threshold', 0.88),
    )

    df = tracking.calculate_postural_states(
        df,
        frame_rate=cfg['frame_rate'],
        direction_threshold=cfg['direction_threshold'],
        speed_threshold=cfg['speed_threshold'],
        min_run_length=cfg.get('min_run_length', 5),
        reversal_persistence=cfg.get('reversal_persistence', 2),
        pirouette_speed_threshold=cfg.get('pirouette_speed_threshold', 0.1),
        pirouette_eccentricity_threshold=cfg.get('pirouette_eccentricity_threshold', 0.9),
        min_pirouette_duration=cfg.get('min_pirouette_duration', 4),
        area_reliability_threshold=cfg.get('area_reliability_threshold', 0.70),
        merge_reversal_gap=cfg.get('merge_reversal_gap', 5),
    )

    # Summary
    n_reversals = df['reversal_start'].sum()
    n_with_rev = (df.groupby('particle')['reversal_start'].sum() > 0).sum()
    n_pirouettes = df['pirouette_start'].sum()
    print(f"Reversals: {n_reversals} events across {n_with_rev}/{df['particle'].nunique()} particles")
    print(f"Pirouettes: {n_pirouettes} events")

    per_particle = df.groupby('particle').agg(
        n_frames=('frame', 'count'),
        x_mean=('x', 'mean'),
        y_mean=('y', 'mean'),
        mean_speed=('speed', 'mean'),
        n_reversals=('reversal_start', 'sum'),
        n_pirouettes=('pirouette_start', 'sum'),
    ).reset_index()
    print("\nPer-particle (particles with reversals or pirouettes):")
    mask = (per_particle['n_reversals'] > 0) | (per_particle['n_pirouettes'] > 0)
    print(per_particle[mask].sort_values('n_reversals', ascending=False).to_string(index=False))

    if args.dry_run:
        print("\n[dry-run] tracks.csv not overwritten.")
        return

    df.to_csv(csv_path, index=False)
    print(f"\nWrote updated tracks.csv to {csv_path}")


if __name__ == '__main__':
    main()
