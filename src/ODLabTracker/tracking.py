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



def preprocess_frame(img, min_area, max_area, thresh):
        # Convert PIL.Image or other inputs to numpy
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # if img.ndim == 3:
    #     gray = color.rgb2gray(img)
    # else:
    #     gray = img

    gray = img

    # Use provided threshold or compute new one
    if thresh is None:
        thresh = filters.threshold_otsu(gray)

    bw = gray > thresh   # worms lighter

    bw = morphology.remove_small_objects(bw, min_size=min_area)

    labeled = measure.label(bw)
    props = measure.regionprops(labeled)

    mask = np.zeros_like(bw, dtype=bool)
    for prop in props:
        if min_area <= prop.area <= max_area:
            mask[labeled == prop.label] = True

    return mask, props, thresh

def draw_boxes_labels(frame, props):
    overlay = np.array(frame).copy()

    # Convert RGB to grayscale [0-255]
    if overlay.ndim == 3:
        overlay = color.rgb2gray(overlay)
        overlay = (overlay * 255).astype(np.uint8)

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        rr, cc = rectangle_perimeter((minr, minc), end=(maxr, maxc), shape=overlay.shape[:2])
        overlay[rr, cc] = 255
    return overlay

def process_video(min_area, 
    max_area, 
    thresh, 
    input_path=None, 
    output_path=None, 
    max_frames=None,
    fps=None, 
    save_as="mp4"):
    """
    Process TIFF stack or video, annotate worms, save as MP4 or TIFF stack.

    Parameters
    ----------
    input_path : str
        Path to input video or TIFF stack
    output_path : str or None
        Path to save annotated output (mp4 or tif). If None, nothing saved.
    max_frames : int or None
        Process only first N frames (for testing).
    save_as : str
        "mp4" or "tif"
    """
    os.makedirs(output_path,exist_ok=True)

    #reader = imageio.get_reader(input_path)
    # cap = cv2.VideoCapture(input_path)
    # nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print("The video has Frames:", nframes, "FPS:", fps)
    # cap.release()



    import time
    # update after fixing manual import

    start_time = time.time()
    im = iio.imread(input_path)
    end_time = time.time()
    print(f"Reading in {im.shape[0]} frames of video took {end_time - start_time} seconds")

    if im.shape[-1] == 3:
        RGB=1
        print("Video is RGB, need to convert to grayscale 8-bit - consider \n changing video output to this type ahead of time")


    frames = []
    # ----- convert to grayscale 8-bit ---------
    #if im.ndim[] == 3 and im[0].shape[-1] == 3:
    #    print("converting to grayscale images")
    start_time = time.time()
    for i, frame in enumerate(im):
        if i >= 50:
            break
    # Convert to grayscale only if RGB
        # if frame.ndim == 3 and frame.shape[-1] == 3: 
        if RGB==1: 
            frame = rgb2gray(frame)
            frame = (frame * 255).astype(np.uint8)
        elif frame.ndim == 2:  # already grayscale
            #print("converting to 8-bit")
            frame = frame.astype(np.uint8)
            # Append the processed frame to the list
        frames.append(frame)
    end_time = time.time()
    print(f"Converting {len(frames)} frames to 8-bit grayscale took {end_time - start_time} seconds")
    
    result_path = os.path.join(f"{os.path.splitext(input_path)[0]}_results")
    print(f"results will be saved to {result_path}")
    workingDir = os.getcwd()

    for i, frame in enumerate(frames):
        if i == 0:
            first_frame = frame
        else:
            break

    if save_as == "mp4" and output_path:
        writer = imageio.get_writer(os.path.join(output_path,"worms_annotated.mp4"), fps=fps)
    else:
        writer = None

    overlays = []  # store frames if saving as TIFF

    # --- Get threshold from first frame ---
    if thresh is None:
        mask1, props1, global_thresh = preprocess_frame(first_frame, min_area, max_area, thresh)
        print("Auto calculated threshold value is",global_thresh)
        #print(mean(props1.area))
        plt.figure(figsize=(8,6))
        plt.imshow(mask1, cmap="gray")
        plt.title(f"Frame {i}, worms: {len(props1)}")
        plt.axis("off")
        plt.show()
    else:
        global_thresh = thresh
        print("Manual threshold value is",global_thresh)
        mask1, props1, _ = preprocess_frame(first_frame, min_area, max_area, thresh=global_thresh)
        areas = []
        for prop in props1:
            area = prop.area
            areas.append(area)
        print('Mean area of objects in pixels is', np.mean(areas))
        if (np.mean(areas) < min_area) | (np.mean(areas) > max_area):
            print("which is outside of your min and max area estimate")
        plt.figure(figsize=(8,6))
        plt.imshow(mask1, cmap="gray")
        plt.title(f"Frame {i}, worms: {len(props1)}")
        plt.axis("off")
        plt.show()

    for i, frame in enumerate(im):
        if max_frames and i >= max_frames:
            break

        mask, props, _ = preprocess_frame(frame, min_area, max_area, thresh=global_thresh)
                                    # Debug: show first frame
        # if i < 1:
        #     plt.figure(figsize=(8,6))
        #     plt.imshow(mask, cmap="gray")
        #     plt.title(f"Frame {i}, worms: {len(props)}")
        #     plt.axis("off")
        #     plt.show()

        overlay = draw_boxes_labels_gray(frame, props)

        if save_as == "mp4" and writer:
            writer.append_data(overlay)
        elif save_as == "tif":
            overlays.append(overlay)


    if save_as == "mp4" and writer:
        writer.close()
    elif save_as == "tif" and output_path:
        tifffile.imwrite(os.path.join(output_path,"worms_annotated.tif"), np.array(overlays), photometric="minisblack")

def draw_boxes_labels_gray(frame, props):
    """
    Draw bounding boxes + worm labels on grayscale frame.
    """
    overlay = np.array(frame).copy()
    # Convert RGB to grayscale [0-255] - already done in process_video
    # if overlay.ndim == 3:
    #     from skimage import color
    #     overlay = color.rgb2gray(overlay)
    #     overlay = (overlay * 255).astype(np.uint8)

    for prop in props:
        # if no objects are detected generate error:
        if (len(prop.bbox)) == 4:
            minr, minc, maxr, maxc = prop.bbox
            rr, cc = rectangle_perimeter(
                (minr, minc), end=(maxr, maxc), shape=overlay.shape[:2]
            )
            overlay[rr, cc] = 255  # white box

            # Add label text (use regionprops label)
            y, x = prop.centroid
            cv2.putText(
                overlay, str(prop.label),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,), 1, cv2.LINE_AA
            )
        else:
            print("No object detected in this frame")
    return overlay

def collect_detections(stack, global_thresh, min_area, max_area):
    records = []
    for frame_no, frame in enumerate(stack):
        mask, props, _ = preprocess_frame(frame, min_area=min_area, max_area=max_area, thresh=global_thresh,)
        for prop in props:
            y, x = prop.centroid  # note (row, col) = (y, x)
            records.append({
                "frame": frame_no,
                "x": x,
                "y": y,
                "area": prop.area
            })
    return pd.DataFrame(records)

def link_tracks(detections, search_range=50, memory=3):
    """
    Link worm detections into tracks.
    search_range: max distance a worm can move between frames
    memory: how many frames a worm can vanish and still be linked
    """
    linked = tp.link_df(detections, search_range=search_range, memory=memory)
    return linked

def draw_tracks_gray(frame, props, tracks, frame_no):
    overlay = np.array(frame).copy()
    if overlay.ndim == 3:
        overlay = color.rgb2gray(overlay)
        overlay = (overlay * 255).astype(np.uint8)

    # Get track IDs for this frame
    frame_tracks = tracks[tracks["frame"] == frame_no]

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        rr, cc = rectangle_perimeter((minr, minc), end=(maxr, maxc), shape=overlay.shape[:2])
        overlay[rr, cc] = 255

        # Match centroid to track
        y, x = prop.centroid
        match = frame_tracks[(frame_tracks["x"].sub(x).abs() < 1) &
                             (frame_tracks["y"].sub(y).abs() < 1)]
        if not match.empty:
            track_id = int(match["particle"].iloc[0])
            cv2.putText(
                overlay, str(track_id),
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,), 1, cv2.LINE_AA
            )
    return overlay

def filter_short_tracks(tracks, min_length=10):
    """
    Remove worms (particles) with fewer than `min_length` frames.
    """
    counts = tracks.groupby("particle")["frame"].count()
    keep_ids = counts[counts >= min_length].index
    return tracks[tracks["particle"].isin(keep_ids)].copy()

def plot_trajectories(stack, tracks, output_path, background="first"):
    """
    Plot worm trajectories on top of an image.

    Parameters
    ----------
    stack : ndarray
        TIFF stack (frames, h, w)
    tracks : DataFrame
        trackpy-linked detections with columns [frame, x, y, particle]
    background : str
        "first" = use first frame
        "mean" = use mean intensity projection
    """
    bg = stack

    #    bg = np.zeros_like(stack[0])

    plt.figure(figsize=(10,8))
    plt.imshow(bg, cmap="gray")


    # Plot trajectories for each worm
    for pid, worm in tracks.groupby("particle"):
        plt.plot(worm["x"], worm["y"], marker=".", markersize=1, label=f"Worm {pid}")
    plt.legend(loc="upper right", fontsize=6)
    plt.title("Worm Trajectories")
    plt.savefig(os.path.join(output_path,"trackPlot.png"))
    plt.show()

def load_video_frames(path, max_frames=None):
    """
    Load frames from AVI or TIFF.
    Returns a numpy array: (n_frames, h, w), grayscale.
    """
    frames = []

    # imageio can open both .avi and .tif
    reader = iio.imiter(path)
    for i, frame in enumerate(reader):
        if max_frames and i >= max_frames:
            break
        # If RGB, convert to grayscale
        if frame.ndim == 3:
            frame = rgb2gray(frame)
            frame = (frame * 255).astype(np.uint8)
        frames.append(frame)

    return np.stack(frames, axis=0)
