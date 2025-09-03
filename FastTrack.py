#!/usr/bin/env python
# coding: utf-8


# importing, data handling
import numpy as np
import pandas as pd
import ipyfilechooser
import os
import imageio
import tifffile
from PIL import Image

# plotting
import matplotlib  as mpl
import cv2
import matplotlib.pyplot as plt

# thresholding
from skimage import io, color, filters, morphology, measure
from skimage.draw import rectangle_perimeter

# tracking
import trackpy as tp

from IPython.display import Video

import ODLabTracker
from ODLabTracker import tracking

# ## 1. Setup
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
# interactive = True;
# if(interactive == True):
#     workingDir = os.getcwd()
#     #baseDir = os.path.dirname(workingDir)
#
#     # need to install ipyfilechooser - use pip
#     from ipyfilechooser import FileChooser
#
#     # Create and display a FileChooser widget
#     fc = FileChooser(workingDir)
#     print("select the first .tiff file you want to analyze")
#     display(fc)
#
# else:
#     fc.selected_filename = filepath
#     fc.selected_path = os.path.splitext(filepath)[0]
#     workingDir = os.getdwc()


# ### Input parameters

# In[3]:


filename = os.path.basename(file_path)
result_path = os.path.join(f"{os.path.splitext(file_path)[0]}_results")
print(f"results will be saved to {result_path}")
workingDir = os.getcwd()

min_area = 200 #min area of worm in pixels
max_area = 2000 #max area of worm in pixels
gap_range = 4 # max number of frames to link worms
thresh=None # manual threshold - use if too few worms detected or too many short tracks


# In[4]:


print("There should be white worms on black background, \nin the video located in the folder, there should be a box around all worms.\nIf not, you may need to adjust the threshold value manually")
tracking.process_video(file_path, min_area, max_area, thresh, output_path=result_path, save_as="mp4", max_frames=50)
#Video.from_file(os.path.join(result_path,"worms_annotated.mp4"), width=480, height=320)


# In[5]:


import tifffile

# Load stack
stack = tifffile.imread(file_path)

# Compute global threshold
_, _, global_thresh = tracking.preprocess_frame(stack[0], min_area, max_area, thresh)

# Collect detections
detections = tracking.collect_detections(stack, global_thresh, min_area, max_area)

# Link tracks
tracks = tracking.link_tracks(detections, search_range=60, memory=4)

# Now tracks['particle'] is a persistent worm ID


# In[6]:


counts = tracks.groupby("particle")["frame"].count()
print(np.ceil(np.mean(counts)/2))
print(int(min(counts)))
print(int(max(counts)))
#plt.figure(figsize=(10,8))
binwidth = 50
plt.hist(counts, bins=range(int(min(counts)), int(max(counts)) + binwidth, binwidth))
plt.xlabel("length of track in frames")
plt.title("histogram of worm track lengths")
plt.xlabel("length of track in frames \n if too many short tracks, try increasing gap_range or decrease threshold")
plt.ylabel("number of worm tracks")
plt.show()


# Remove short tracks
tracks = tracking.filter_short_tracks(tracks, min_length=20)

for i, frame in enumerate(stack):
    mask, props, _ = tracking.preprocess_frame(frame, min_area, max_area, thresh=global_thresh)
    overlay = tracking.draw_tracks_gray(frame, props, tracks, i)
    # save overlay into MP4 or TIFF


# Plot over first frame
tracking.plot_trajectories(stack, tracks, result_path, background="first")

# save tracking data


# In[10]:


tracks.to_csv(os.path.join(result_path,"tracks.csv"), index=False)


# # In[ ]:
#
#
# ##### below is slow method for posture #####
#
#
# # In[169]:
#
#
# f = tp.locate(stack[0], 101, threshold = 25)
#
#
# # In[170]:
#
#
# f.head()
#
#
# # In[171]:
#
#
# tp.annotate(f, stack[0]);
#
#
# # In[173]:
#
#
# fig, ax = plt.subplots()
# ax.hist(f['mass'], bins=10)
# ax.set(xlabel='mass', ylabel='count')
#
#
# # In[181]:
#
#
# f = tp.batch(stack[:100], 51, threshold = 15);
#
#
# # In[182]:
#
#
# t = tp.link(f, 20, memory=4)
#
#
# # In[183]:
#
#
# t.head()
#
#
# # In[184]:
#
#
# plt.figure()
# tp.mass_size(t.groupby('particle').mean());
#
#
# # In[185]:
#
#
# tp.plot_traj(t)
#
#
# # In[ ]:
