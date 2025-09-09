#!/usr/bin/env python
# coding: utf-8

# importing, data handling
import numpy as np
import pandas as pd
import ipyfilechooser
import os
import imageio
import imageio.v3 as iio
import tifffile
from PIL import Image
from skimage.color import rgb2gray
import yaml

# plotting
import matplotlib  as mpl
import cv2
import matplotlib.pyplot as plt

# thresholding
from skimage import io, color, filters, morphology, measure
from skimage.draw import rectangle_perimeter

# tracking
import trackpy as tp

#from IPython.display import Video, Markdown, display

import ODLabTracker
from ODLabTracker import tracking

####### 1. Setup #############
import sys
import argparse

## Using argparse (more robust for complex arguments)
parser = argparse.ArgumentParser(description="Process a file.")
#parser.add_argument("-i", "--interactive", action="store_true", help="interactive file selection")
parser.add_argument("-f", "--filename", help="The path to the input file, if specified")
parser.add_argument("-c", "--config", help="Configuration (yaml) file, if specfified")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
args = parser.parse_args()

if args.filename:
    print("batch mode")
    print(f"Processing file: {os.path.join(os.getcwd(),args.filename)}")
    file_path = os.path.join(os.getcwd(),args.filename)
else:
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(title="Select a Video File")
    
    root.destroy()

result_path = os.path.join(f"{os.path.splitext(file_path)[0]}_results")
print(f"results will be saved to {result_path}")   

if args.verbose:
    print("Verbose mode enabled.")

#### config file setup ####
if args.config:
    config_path = os.path.join(os.getcwd(),args.config)
    import yaml

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    min_area = config_data['min_area'] #min area of worm in pixels
    max_area = config_data['max_area'] #max area of worm in pixels
    gap_range = config_data['gap_range'] # max number of frame gap to link worms
    if config_data['thresh'] == 'None':
        thresh = None
    else:
        thresh = config_data['thresh'] # manual threshold - use if too few worms detected or too many short tracks
    search_range = config_data['search_range'] # max pixel distance to link tracks across frames
    min_length = config_data['min_length'] # minimum length of track in frames to keep
    frame_rate = config_data['frame_rate'] # FPS - only necessary for speed analysis
    illumination = config_data['illumination'] # illumination source, 0 = white worms on dark (e.g. IR), 1 = dark worms on light
    
else:
    #default values small plate on IR light:
    min_area = 200 #min area of worm in pixels
    max_area = 2000 #max area of worm in pixels
    gap_range = 8 # max number of frame gap to link worms
    thresh = None # manual threshold - use if too few worms detected or too many short tracks
    search_range = 60 # max pixel distance to link tracks across frames
    min_length = 25 # minimum length of track in frames to keep
    frame_rate = 10 # FPS - only necessary for speed analysis
    illumination = 0 # illumination source, 0 = white worms on dark (e.g. IR), 1 = dark worms on light

print('PARAMETER SETTINGS:')
print(f'minimum area of worm in pixels: {min_area}')
print(f'maximum area of worm in pixels: {max_area}')
print(f'gap range of worms in frames: {gap_range}')
if thresh is None:
    print(f'automatically calculating threshold')
else:
    print(f'manual threshold: {thresh}')
print(f'maximum pixel distance to link worms: {search_range}')
print(f'minimum length of worm track to keep in frames: {min_length}')
print(f'frame rate to use for speed analysis: {frame_rate}')
if illumination == 0:
    print(f'analyzing light worms on dark background')
else:
    print(f'analyzing dark worms on light background')

print(f'selected image path: {file_path}')
filename = os.path.splitext(file_path)[0]
print(f'selected filename: {filename}')


###### 2. warnings for non-optimal video ######
import time

start_time = time.time()
im = iio.imread(file_path)
end_time = time.time()
print(f"Reading in {im.shape[0]} frames of video took {end_time - start_time} seconds")
print(im.shape)

if im.shape[-1] == 3:
    print("!!!Video is RGB, need to convert to grayscale 8-bit - consider \n changing video output to this type ahead of time!!!")

##### 3. tracking #######

print(f'Running full tracking on {im.shape[0]} frames')

# if that looks good, run for the whole video after converting to 8-bit grayscale
frames = []

# not sure why matmul errors happen for the first rgb2gray call but suppress this block:
with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
    start_time = time.time()
    for i, frame in enumerate(im):
        # Convert to grayscale only if RGB
        if frame.ndim == 3 and frame.shape[-1] == 3: 
            #print("converting to grayscale images")
            frame = rgb2gray(frame)
            frame = (frame * 255).astype(np.uint8)
        elif frame.ndim == 2:  # already grayscale
            #print("converting to 8-bit")
            frame = frame.astype(np.uint8)
            # Append the processed frame to the list
        frames.append(frame)
    end_time = time.time()
print(f"Converting {len(frames)} frames to 8-bit grayscale took {end_time - start_time} seconds")

# now simple track for the whole video and link tracks together
first_frame = frames[0]

# Compute global threshold
_, _, global_thresh = tracking.preprocess_frame(first_frame, 
                                                min_area, 
                                                max_area, 
                                                thresh,
                                                illumination=illumination)

# Collect detections
if thresh is None:
    detections = tracking.collect_detections(frames, 
                                             global_thresh, 
                                             min_area, 
                                             max_area,
                                            illumination-illumination)
else:    
    print("tracking full video using manual threshold")
    detections = tracking.collect_detections(frames, 
                                             thresh, 
                                             min_area, 
                                             max_area,
                                             illumination=illumination)

# Link tracks
# Suppress all trackpy logging messages
tp.ignore_logging()

tracks = tracking.link_tracks(detections, search_range=search_range, memory=gap_range)

# Now tracks['particle'] is a persistent worm ID

# visualize the track length of each "worm"
# if there are background particles, this will lead to a ton of short tracks

counts = tracks.groupby("particle")["frame"].count()
print('Mean track length is ',np.ceil(np.mean(counts)/2), ' frames')
print('Minimum track length is ',int(min(counts)))
print('Maximum track length is ',int(max(counts)))
#plt.figure(figsize=(10,8))
binwidth = 25
plt.hist(counts, bins=range(int(min(counts)), int(max(counts)) + binwidth, binwidth))
plt.xlabel("length of track in frames")
plt.title("histogram of worm track lengths")
plt.xlabel("length of track in frames \nif too many short tracks, try increasing gap_range, \nif your real worms are disconnected, \nor increase threshold if there are too many small objects")
plt.ylabel("number of worm tracks")
plt.show()

# Remove short tracks
print(f'removing tracks shorter than {min_length} frames')
tracks = tracking.filter_short_tracks(tracks, min_length=min_length)

# Plot over first frame
print('plotting linked and filtered worm tracks')
tracking.plot_trajectories(stack=first_frame, tracks=tracks, output_path=result_path)

# Save the track centroids to a csv file
print(f'saving tracked centroids to {os.path.join(result_path,"tracks.csv")}')
tracks.to_csv(os.path.join(result_path,"tracks.csv"), index=False)
