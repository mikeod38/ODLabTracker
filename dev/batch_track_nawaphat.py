"""
Batch tracking / reanalysis script for Nawaphat's 0_COMPLETE!! dataset.

Modes (--mode):

  track-missing   (default) — FastTrack.py on videos without tracks.csv.
  retrack-all     — FastTrack.py on ALL videos, overwriting existing tracks.csv.
                    Use when detection/linking code has changed.
  reanalyze-all   — reanalyze_postural.py on all existing tracks.csv (no video
                    re-read). Use when only speed/postural analysis changed.
  all             — track-missing, then reanalyze-all.

Usage:
    python dev/batch_track_nawaphat.py                        # track missing only
    python dev/batch_track_nawaphat.py --mode retrack-all     # retrack everything
    python dev/batch_track_nawaphat.py --mode reanalyze-all   # refresh postural only
    python dev/batch_track_nawaphat.py --dry-run              # list work, exit
    python dev/batch_track_nawaphat.py --workers 6            # set parallelism
"""

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTTRACK = os.path.join(REPO_ROOT, 'FastTrack.py')
REANALYZE = os.path.join(REPO_ROOT, 'dev', 'reanalyze_postural.py')
PYTHON = os.path.expanduser('~/.pyenv/shims/python')

DEFAULT_DATA_DIR = (
    "/Volumes/User Homes/ODlab-user/UserFolders/Nawaphat"
    "/6 CEST-2.1/Locomotion_Off food/0_COMPLETE!!"
)


def find_all_videos(data_dir):
    """Return [(genotype, avi_path)] for every .avi in the dataset."""
    videos = []
    for geno in sorted(os.listdir(data_dir)):
        gdir = os.path.join(data_dir, geno)
        if not os.path.isdir(gdir) or geno == 'placeholder' or geno.startswith('._'):
            continue
        for fname in sorted(os.listdir(gdir)):
            if not fname.endswith('.avi') or fname.startswith('._'):
                continue
            videos.append((geno, os.path.join(gdir, fname)))
    return videos


def find_untracked(data_dir):
    """Return [(genotype, avi_path)] for videos without tracks.csv."""
    untracked = []
    for geno in sorted(os.listdir(data_dir)):
        gdir = os.path.join(data_dir, geno)
        if not os.path.isdir(gdir) or geno == 'placeholder' or geno.startswith('._'):
            continue
        for fname in sorted(os.listdir(gdir)):
            if not fname.endswith('.avi') or fname.startswith('._'):
                continue
            stem = os.path.splitext(fname)[0]
            csv = os.path.join(gdir, stem + '_results', 'tracks.csv')
            if not os.path.exists(csv):
                untracked.append((geno, os.path.join(gdir, fname)))
    return untracked


def find_existing_results(data_dir):
    """Return [(genotype, results_dir)] for all results dirs with tracks.csv."""
    existing = []
    for geno in sorted(os.listdir(data_dir)):
        gdir = os.path.join(data_dir, geno)
        if not os.path.isdir(gdir) or geno == 'placeholder' or geno.startswith('._'):
            continue
        for dname in sorted(os.listdir(gdir)):
            if not dname.endswith('_results') or dname.startswith('._'):
                continue
            results_dir = os.path.join(gdir, dname)
            if os.path.exists(os.path.join(results_dir, 'tracks.csv')):
                existing.append((geno, results_dir))
    return existing


# ── zoom calibration helpers ────────────────────────────────────────────────

def load_calibration(csv_path):
    """Return DataFrame indexed by date string, or None if path is not given."""
    if csv_path is None or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, dtype={"date": str})
    df = df.set_index("date")
    return df


def make_calibrated_config(base_config_path, calibration, date_str, config_cache_dir):
    """
    Write (and cache) a date-specific YAML with calibrated pixel_length /
    search_range overriding the base config.  Returns the path to that YAML.
    If calibration is None or the date has no entry, returns base_config_path.
    """
    if calibration is None or date_str not in calibration.index:
        return base_config_path

    cache_path = os.path.join(config_cache_dir, f"config_{date_str}.yaml")
    if os.path.exists(cache_path):
        return cache_path

    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    row = calibration.loc[date_str]
    cfg["pixel_length"]  = round(float(row["pixel_length_cal"]), 2)
    cfg["search_range"]  = int(row["search_range_cal"])

    os.makedirs(config_cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return cache_path


def date_from_avi(avi_path):
    """Extract YYYYMMDD from filename, or '' if not present."""
    m = re.match(r"^(\d{8})", os.path.basename(avi_path))
    return m.group(1) if m else ""


# ── subprocess runners ──────────────────────────────────────────────────────

def track_one(args):
    """Run FastTrack.py on a video. Returns (path, returncode, elapsed, log_path)."""
    avi_path, config_path, log_dir = args
    stem = os.path.splitext(os.path.basename(avi_path))[0]
    log_path = os.path.join(log_dir, 'track_' + stem + '.log')
    env = {**os.environ, 'MPLBACKEND': 'Agg'}  # suppress GUI windows
    t0 = time.time()
    with open(log_path, 'w') as lf:
        rc = subprocess.run(
            [PYTHON, FASTTRACK, '-f', avi_path, '-c', config_path],
            stdout=lf, stderr=subprocess.STDOUT, text=True, env=env,
        ).returncode
    return avi_path, rc, time.time() - t0, log_path


def reanalyze_one(args):
    """Run reanalyze_postural.py on a results dir. Returns (path, rc, elapsed, log_path)."""
    results_dir, config_path, log_dir = args
    stem = os.path.basename(results_dir)
    log_path = os.path.join(log_dir, 'reanalyze_' + stem + '.log')
    t0 = time.time()
    with open(log_path, 'w') as lf:
        rc = subprocess.run(
            [PYTHON, REANALYZE, results_dir, '--config', config_path],
            stdout=lf, stderr=subprocess.STDOUT, text=True,
        ).returncode
    return results_dir, rc, time.time() - t0, log_path


# ── dispatch helpers ────────────────────────────────────────────────────────

def run_batch(jobs, worker_fn, n_workers, label, total_label):
    """Submit jobs to a pool; print progress; return list of failed paths."""
    failures = []
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(worker_fn, j): j[0] for j in jobs}
        for fut in as_completed(futures):
            path, rc, elapsed, log_path = fut.result()
            done += 1
            status = 'OK' if rc == 0 else f'FAILED (rc={rc})'
            name = os.path.basename(path)
            print(f"  [{done}/{total_label}] {status}  {elapsed/60:.1f} min  {name}")
            if rc != 0:
                failures.append(path)
                print(f"    Log: {log_path}")
                with open(log_path) as f:
                    for line in f.readlines()[-15:]:
                        print(f"    | {line}", end="")
    elapsed_total = time.time() - t0
    ok = total_label - len(failures)
    print(f"\n{label}: {ok}/{total_label} succeeded in {elapsed_total/60:.1f} min")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Batch track/reanalyze Nawaphat locomotion dataset")
    parser.add_argument('--mode', choices=['track-missing', 'retrack-all', 'reanalyze-all', 'all'],
                        default='track-missing',
                        help="track-missing: FastTrack only missing; "
                             "retrack-all: FastTrack every video (overwrites); "
                             "reanalyze-all: re-apply postural analysis on existing tracks.csv; "
                             "all: track-missing then reanalyze-all (default: track-missing)")
    parser.add_argument('--config', default=os.path.join(REPO_ROOT, 'configs', 'IR_medium.yaml'),
                        help="YAML config (default: configs/IR_medium.yaml)")
    parser.add_argument('--workers', type=int, default=2,
                        help="Parallel subprocesses (default: 2; CPU-bound so >2 gives diminishing returns)")
    parser.add_argument('--dry-run', action='store_true',
                        help="List work to be done and exit")
    parser.add_argument('--limit', type=int, default=None,
                        help="Process at most N videos/dirs (useful for profiling)")
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help="Root folder containing genotype subfolders")
    parser.add_argument('--calibration', default=None,
                        help="zoom_calibration.csv (from dev/zoom_calibration.py). "
                             "When given, per-date pixel_length and search_range "
                             "override the base config for each recording.")
    parser.add_argument('--genotype', default=None,
                        help="Only process this genotype folder (e.g. 'N2'). "
                             "Matches the subfolder name exactly.")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        sys.exit(f"Config not found: {config_path}")

    calibration = load_calibration(args.calibration)
    if calibration is not None:
        print(f"Zoom calibration loaded: {len(calibration)} dates from {args.calibration}")
    config_cache_dir = os.path.join(REPO_ROOT, 'logs', 'batch_track_nawaphat', 'configs')

    log_dir = os.path.join(REPO_ROOT, 'logs', 'batch_track_nawaphat')
    os.makedirs(log_dir, exist_ok=True)

    do_retrack_all = args.mode == 'retrack-all'
    do_track = args.mode in ('track-missing', 'all')
    do_reanalyze = args.mode in ('reanalyze-all', 'all')

    if args.genotype:
        # Point data_dir directly at the single genotype subfolder so that
        # find_all_videos / find_existing_results treat it as the only entry.
        # We wrap it in a temp structure by overriding data_dir to parent and
        # filtering after collection.
        _geno_filter = args.genotype
    else:
        _geno_filter = None

    if do_retrack_all:
        videos_to_track = find_all_videos(args.data_dir)
    elif do_track:
        videos_to_track = find_untracked(args.data_dir)
    else:
        videos_to_track = []
    existing = find_existing_results(args.data_dir) if do_reanalyze else []

    if _geno_filter:
        videos_to_track = [(g, p) for g, p in videos_to_track if g == _geno_filter]
        existing        = [(g, p) for g, p in existing        if g == _geno_filter]

    if args.limit is not None:
        videos_to_track = videos_to_track[:args.limit]
        existing = existing[:args.limit]

    # Summary
    if do_retrack_all or do_track:
        label = "Videos to retrack" if do_retrack_all else "Videos to track"
        print(f"{label} ({len(videos_to_track)}):")
        for geno, path in videos_to_track:
            sz = os.path.getsize(path) / 1e6
            print(f"  [{geno:22s}] {os.path.basename(path)}  ({sz:.0f} MB)")
        if not videos_to_track:
            print("  (none)")

    if do_reanalyze:
        print(f"\nResults dirs to reanalyze ({len(existing)}):")
        for geno, rdir in existing[:5]:
            print(f"  [{geno:22s}] {os.path.basename(rdir)}")
        if len(existing) > 5:
            print(f"  ... and {len(existing) - 5} more")

    if args.dry_run:
        return

    print(f"\nConfig:  {config_path}")
    print(f"Workers: {args.workers}")
    print(f"Logs:    {log_dir}")
    print(f"Start:   {time.strftime('%H:%M:%S')}\n")

    all_failures = []

    if (do_retrack_all or do_track) and videos_to_track:
        label = "Retracking" if do_retrack_all else "Tracking"
        print(f"=== {label} {len(videos_to_track)} videos ===")
        track_jobs = []
        for _, path in videos_to_track:
            date = date_from_avi(path)
            cfg = make_calibrated_config(config_path, calibration, date, config_cache_dir)
            if cfg != config_path:
                row = calibration.loc[date]
                print(f"  Calibrated {date}: pixel_length={row['pixel_length_cal']:.1f}  "
                      f"search_range={row['search_range_cal']}")
            track_jobs.append((path, cfg, log_dir))
        all_failures += run_batch(track_jobs, track_one, args.workers,
                                  label, len(videos_to_track))

    if do_reanalyze and existing:
        print(f"\n=== Reanalyzing {len(existing)} existing results dirs ===")
        reanalyze_jobs = [(rdir, config_path, log_dir) for _, rdir in existing]
        all_failures += run_batch(reanalyze_jobs, reanalyze_one, args.workers,
                                  "Reanalysis", len(existing))

    if all_failures:
        print(f"\nFailed ({len(all_failures)}):")
        for f in all_failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("\nAll jobs completed successfully.")


if __name__ == '__main__':
    main()
