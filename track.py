#!/usr/bin/env python
# coding: utf-8
"""
ODLabTracker entry point.

Reads 'mode' from the config file and dispatches to the appropriate analysis:
  mode: postural  → FastTrack.py       (body movement states)
  mode: pumping   → FastTrackPumping.py (pharyngeal pumping)

Usage:
  python track.py -f <video> -c <config.yaml>
  python track.py -c <config.yaml>          # interactive file picker
"""

import argparse
import os
import sys
import yaml


def get_file_path(filename_arg):
    if filename_arg:
        return os.path.abspath(filename_arg)
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select a Video File")
    root.destroy()
    if not path:
        print("No file selected. Exiting.")
        sys.exit(1)
    return path


def main():
    parser = argparse.ArgumentParser(description="ODLabTracker — mode-dispatching entry point")
    parser.add_argument("-f", "--filename", help="Path to input video file")
    parser.add_argument("-c", "--config",   help="Configuration (yaml) file", required=True)
    parser.add_argument("-v", "--verbose",  action="store_true")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    mode = config_data.get('mode', 'postural').strip().lower()
    print(f"Mode: {mode}")

    file_path = get_file_path(args.filename)
    print(f"Processing file: {file_path}")

    if mode in ('postural', 'centroid'):
        from FastTrack import main as run_postural
        run_postural(file_path, config_path, verbose=args.verbose)

    elif mode == 'pumping':
        from FastTrackPumping import main as run_pumping
        run_pumping(file_path, config_path, verbose=args.verbose)

    else:
        print(f"ERROR: Unknown mode '{mode}' in config. Choose 'centroid', 'postural', or 'pumping'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
