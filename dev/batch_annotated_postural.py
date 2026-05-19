"""
Generate annotated postural videos for a list of particles from a single video.

Usage:
    python dev/batch_annotated_postural.py <video_path> <results_dir> \
        --particles 23 66 44 122 111 [--config configs/IR_medium.yaml]

Output videos go into <results_dir>/postural_validation/
"""

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ODLabTracker import tracking


def main():
    parser = argparse.ArgumentParser(description="Batch postural annotated video generator")
    parser.add_argument('video_path')
    parser.add_argument('results_dir')
    parser.add_argument('--particles', nargs='+', type=int, required=True,
                        help="Particle IDs to annotate")
    parser.add_argument('--config', default='configs/IR_medium.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    csv_path = os.path.join(args.results_dir, 'tracks.csv')
    df = pd.read_csv(csv_path)
    if 'is_reversal' not in df.columns:
        sys.exit("tracks.csv has no reversal data — run reanalyze_postural.py first")

    out_dir = os.path.join(args.results_dir, 'postural_validation')
    os.makedirs(out_dir, exist_ok=True)

    available = set(df['particle'].unique())
    for pid in args.particles:
        if pid not in available:
            print(f"particle {pid} not in tracks.csv, skipping")
            continue

        pdf = df[df['particle'] == pid]
        n_rev = pdf['reversal_start'].sum()
        n_pir = pdf['pirouette_start'].sum()
        loc = f"x={pdf['x'].mean():.0f} y={pdf['y'].mean():.0f}"
        print(f"\n--- particle {pid}: {len(pdf)} frames, {n_rev} reversals, {n_pir} pirouettes @ {loc}")

        tracking.create_annotated_video(
            video_path=args.video_path,
            df=df,
            particle_id=pid,
            output_folder=out_dir,
            pixel_length=cfg['pixel_length'],
            frame_rate=cfg['frame_rate'],
            min_area=cfg['min_area'],
            max_area=cfg['max_area'],
            illumination=cfg['illumination'],
        )

    print(f"\nDone. Videos written to {out_dir}")


if __name__ == '__main__':
    main()
