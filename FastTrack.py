#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import tifffile
from skimage.color import rgb2gray
import yaml
import time

import matplotlib as mpl
import cv2
import matplotlib.pyplot as plt

from skimage import filters, morphology, measure

import trackpy as tp

import ODLabTracker
from ODLabTracker import tracking

import sys
import argparse


class Colors:
    """ANSI color codes"""
    RED    = '\033[91m'
    GREEN  = '\033[92m'
    BLUE   = '\033[94m'
    WARNING = '\033[93m'
    PURPLE = '\033[0;35m'
    ENDC   = '\033[0m'


def main(file_path, config_path, verbose=False):

    result_path = os.path.join(f"{os.path.splitext(file_path)[0]}_results")
    print(f"results will be saved to {result_path}")
    os.makedirs(result_path, exist_ok=True)

    if verbose:
        print("Verbose mode enabled.")

    ####### 2. Config file setup ########
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    min_area      = config_data['min_area']
    max_area      = config_data['max_area']
    gap_range     = config_data['gap_range']
    thresh        = None if config_data['thresh'] == 'None' else config_data['thresh']
    thresh_method = config_data.get('thresh_method', 'otsu')
    search_range  = config_data['search_range']
    min_length    = config_data['min_length']
    frame_rate    = config_data['frame_rate']
    illumination  = config_data['illumination']
    subsample     = config_data['subsample']
    backsub       = config_data['backsub']
    backsub_frames = config_data['backsub_frames']
    pixel_length  = config_data['pixel_length']
    speed_threshold = config_data['speed_threshold']
    window_size   = config_data['window_size']
    direction_threshold = config_data['direction_threshold']
    min_run_length = config_data['min_run_length']
    smooth_positions = config_data['smooth_positions']
    smooth_window = config_data['smooth_window']
    min_displacement_for_angle = config_data['min_displacement_for_angle']
    reversal_persistence = config_data['reversal_persistence']
    pirouette_speed_threshold = config_data['pirouette_speed_threshold']
    pirouette_eccentricity_threshold = config_data['pirouette_eccentricity_threshold']
    min_pirouette_duration = config_data['min_pirouette_duration']
    max_instantaneous_speed = config_data['max_instantaneous_speed']
    stability_threshold = config_data['stability_threshold']
    area_reliability_threshold = config_data.get('area_reliability_threshold', 0.70)
    merge_reversal_gap = config_data.get('merge_reversal_gap', 5)
    max_objects   = config_data.get('max_objects', None)
    min_thresh    = config_data.get('min_thresh', None)
    mode          = config_data.get('mode', 'postural').strip().lower()

    print(f'{Colors.PURPLE}PARAMETER SETTINGS:{Colors.ENDC}')
    print(f'minimum area of worm in pixels: {Colors.GREEN}{min_area}{Colors.ENDC}')
    print(f'maximum area of worm in pixels: {Colors.GREEN}{max_area}{Colors.ENDC}')
    print(f'gap range of worms in frames: {Colors.GREEN}{gap_range}{Colors.ENDC}')
    if backsub:
        print(f'{Colors.GREEN}Subtracting background{Colors.ENDC}')
    if thresh is None:
        print(f'{Colors.GREEN}automatically calculating threshold{Colors.ENDC}')
    else:
        print(f'manual threshold: {Colors.GREEN}{thresh}{Colors.ENDC}')
    print(f'maximum pixel distance to link worms: {Colors.GREEN}{search_range}{Colors.ENDC}')
    print(f'minimum length of worm track to keep in frames: {Colors.GREEN}{min_length}{Colors.ENDC}')
    print(f'frame rate to use for speed analysis: {Colors.GREEN}{frame_rate}{Colors.ENDC}')
    if illumination == 0:
        print(f'{Colors.GREEN}analyzing light worms on dark background{Colors.ENDC}')
    else:
        print(f'{Colors.GREEN}analyzing dark worms on light background{Colors.ENDC}')
    if subsample > 1:
        print(f"keeping one out of every {Colors.GREEN}{subsample}{Colors.ENDC} frames")

    print(f'selected image path: {file_path}')
    filename = os.path.splitext(file_path)[0]
    print(f'selected filename: {filename}')

    ####### 3. Load video ########
    TIFF_EXTENSIONS = {".tif", ".tiff"}
    file_ext = os.path.splitext(file_path)[1].lower()
    is_tiff  = file_ext in TIFF_EXTENSIONS

    if is_tiff:
        tif_file    = tifffile.TiffFile(file_path)
        num_frames  = len(tif_file.pages)
        first_frame = tif_file.pages[0].asarray()
        imiter_vid  = (page.asarray() for page in tif_file.pages)
        print(f"TIFF stack: {num_frames} frames, shape {first_frame.shape}, dtype {first_frame.dtype}")
    else:
        # cv2.VideoCapture reads MJPEG frames as BGR and extracts the Y channel
        # directly via COLOR_BGR2GRAY — immune to chroma channel values (Cb/Cr),
        # unlike rgb2gray which weights channels and is sensitive to non-neutral chroma.
        cap         = cv2.VideoCapture(file_path)
        num_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, _bgr   = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        first_frame = cv2.cvtColor(_bgr, cv2.COLOR_BGR2GRAY) if ret else None
        imiter_vid  = cap
        print(f"Video: {num_frames} frames, shape {first_frame.shape}, dtype {first_frame.dtype}")

    frames = []

    if subsample > 1:
        print(f'Running tracking on {num_frames / subsample:.0f} frames from the original {num_frames}')
    else:
        print(f'Running full tracking on {num_frames} frames')

    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        start_time = time.time()

        if is_tiff:
            for i, frame in enumerate(imiter_vid):
                if i % subsample == 0:
                    print(f"\rKeeping frame: {i}", end="", flush=True)
                    time.sleep(0.001)
                    if frame.ndim == 3 and frame.shape[-1] == 3:
                        frame = rgb2gray(frame)
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                    frames.append(frame)
        else:
            i = 0
            while True:
                ret, bgr = imiter_vid.read()
                if not ret:
                    break
                if i % subsample == 0:
                    print(f"\rKeeping frame: {i}", end="", flush=True)
                    time.sleep(0.001)
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
                i += 1
            imiter_vid.release()

        end_time = time.time()
        print(f"  Reading in {len(frames)} frames took {end_time - start_time:.1f} seconds")

    first_frame = frames[0]
    print(f"  [diag] raw frame[0]: dtype={first_frame.dtype}  "
          f"min={first_frame.min()}  max={first_frame.max()}  "
          f"mean={first_frame.mean():.1f}")

    ####### 4. Background subtraction ########
    if backsub:
        print("Subtracting background")
        backsub_frame = np.zeros_like(first_frame, dtype=np.float32)
        frame_list = np.linspace(1, num_frames - 1, backsub_frames, dtype=int)
        print("Frames to average for background:", frame_list)
        for i in frame_list:
            frame_to_add = tracking.convert_8bit(frames[i])
            backsub_frame += frame_to_add.astype(np.float32)
        average_frame = (backsub_frame / backsub_frames).astype(np.uint8)
        print(f"  [diag] average_frame: min={average_frame.min()}  "
              f"max={average_frame.max()}  mean={average_frame.mean():.1f}")
        plt.figure()
        plt.imshow(average_frame, cmap="gray")
        plt.show(block=False)

        subtracted = [tracking.subtract_background(f, average_frame=average_frame, normalize=False)
                      for f in frames]
        frames = subtracted
        first_frame = subtracted[0]
        print(f"  [diag] subtracted frame[0]: min={first_frame.min()}  "
              f"max={first_frame.max()}  mean={first_frame.mean():.1f}")

        if min_thresh is None:
            noise_est  = float(np.std(average_frame))
            min_thresh = int(np.ceil(2 * noise_est))
            print(f"  [diag] background noise σ={noise_est:.2f}  "
                  f"→ auto min_thresh={min_thresh}")
        else:
            print(f"  [diag] using config min_thresh={min_thresh} "
                  f"(background σ={np.std(average_frame):.2f})")

    ####### 5. Tracking ########
    print(f"Tracking and linking objects from {len(frames)} frames")

    if thresh is None:
        _, _, global_thresh = tracking.preprocess_frame(
            first_frame, min_area, max_area, thresh,
            illumination=illumination, thresh_method=thresh_method,
            max_objects=max_objects, min_thresh=min_thresh)
    else:
        print("tracking video using manual threshold")
        global_thresh = thresh

    detections = tracking.collect_detections(
        frames, global_thresh=global_thresh,
        min_area=min_area, max_area=max_area, illumination=illumination)

    tracks = tracking.link_tracks(detections, search_range=search_range,
                                  memory=gap_range, quiet=True)

    print(f'removing tracks shorter than {min_length} frames')
    tracks = tracking.filter_short_tracks(tracks, min_length=min_length)

    ####### 6. Phase 1: speed parameters (centroid + postural modes) ########
    print(f"calculating speed parameters ({mode} mode)")
    tracks = tracking.calculate_speed_parameters(
        tracks,
        pixel_length=pixel_length,
        frame_rate=frame_rate,
        window_size=window_size,
        speed_threshold=speed_threshold,
        smooth_window=smooth_window,
        min_displacement_for_angle=min_displacement_for_angle,
        max_instantaneous_speed=max_instantaneous_speed,
        stability_threshold=stability_threshold
    )

    ####### 7. Summary, trajectory plot, and CSV ########
    counts = tracks.groupby("particle")["frame"].count()
    print('Mean track length is ', np.ceil(np.mean(counts) / 2), ' frames')
    print('Minimum track length is ', int(min(counts)))
    print('Maximum track length is ', int(max(counts)))
    plt.figure(figsize=(10, 8))
    binwidth = 25
    plt.hist(counts, bins=range(int(min(counts)), int(max(counts)) + binwidth, binwidth))
    plt.xlabel("length of track in frames\nif too many short tracks, try increasing gap_range,\nor increase threshold if there are too many small objects")
    plt.ylabel("number of worm tracks")
    plt.title("histogram of worm track lengths")
    plt.show(block=False)

    print('plotting linked and filtered worm tracks')
    tracking.plot_trajectories(stack=first_frame, tracks=tracks, output_path=result_path)

    print(f'saving tracked centroids to {os.path.join(result_path, "tracks.csv")}')
    tracks.to_csv(os.path.join(result_path, "tracks.csv"), index=False)

    ####### 8. Phase 2: postural states (postural mode only) ########
    if mode == 'postural':
        print("calculating postural states (reversals, pirouettes)")
        tracks = tracking.calculate_postural_states(
            tracks,
            frame_rate=frame_rate,
            direction_threshold=direction_threshold,
            speed_threshold=speed_threshold,
            min_run_length=min_run_length,
            reversal_persistence=reversal_persistence,
            pirouette_speed_threshold=pirouette_speed_threshold,
            pirouette_eccentricity_threshold=pirouette_eccentricity_threshold,
            min_pirouette_duration=min_pirouette_duration,
            area_reliability_threshold=area_reliability_threshold,
            merge_reversal_gap=merge_reversal_gap,
        )
        tracks.to_csv(os.path.join(result_path, "tracks.csv"), index=False)

        print("Finding particle with all behaviors for demonstration video")
        best_particle = tracking.find_particle_with_all_behaviors(tracks)
        if best_particle is not None:
            print(f"Creating annotated video for particle {best_particle}")
            output_video = tracking.create_annotated_video(
                video_path=file_path,
                df=tracks,
                particle_id=best_particle,
                output_folder=result_path,
                pixel_length=pixel_length,
                frame_rate=frame_rate,
                global_thresh=global_thresh,
                min_area=min_area,
                max_area=max_area,
                illumination=illumination,
                crop_size=300,
                show_mask=True
            )
            print(f"Annotated video saved to {output_video}")
        else:
            print("No particle found with all three behaviors — skipping annotated video")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ODLabTracker — postural mode")
    parser.add_argument("-f", "--filename", help="Path to input video file")
    parser.add_argument("-c", "--config",   help="Configuration (yaml) file", required=True)
    parser.add_argument("-v", "--verbose",  action="store_true")
    args = parser.parse_args()

    if args.filename:
        print("batch (non-interactive) mode")
        file_path = os.path.abspath(args.filename)
        print(f"Processing file: {file_path}")
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select a Video File")
        root.destroy()

    config_path = os.path.abspath(args.config)
    main(file_path, config_path, verbose=args.verbose)
