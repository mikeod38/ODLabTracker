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
import time

# plotting
import matplotlib  as mpl
import cv2
import matplotlib.pyplot as plt

# thresholding
from skimage import io, color, filters, morphology, measure
from skimage.draw import rectangle_perimeter

# tracking
import trackpy as tp
from scipy.ndimage import median_filter

from IPython.display import Video



THRESH_METHODS = {
    'otsu':     filters.threshold_otsu,
    'triangle': filters.threshold_triangle,
    'yen':      filters.threshold_yen,
    'li':       filters.threshold_li,
}

def _count_inrange(gray, thresh, min_area, max_area, illumination):
    """Return (props_filtered, labeled) at a given threshold (no mask built)."""
    if illumination == 0:
        bw = gray > thresh
    else:
        bw = gray < thresh
    bw = morphology.remove_small_objects(bw, min_size=min_area)
    labeled = measure.label(bw)
    props = measure.regionprops(labeled)
    props_filtered = [p for p in props if min_area <= p.area_convex <= max_area]
    return props_filtered, labeled


def preprocess_frame(img, min_area, max_area, thresh, illumination, thresh_method='otsu', max_objects=None, min_thresh=None):
    """Threshold a frame and return (mask, props_filtered, thresh_used).

    Parameters
    ----------
    max_objects : int or None
        When thresh=None and max_objects is set, the auto threshold is raised
        via binary search until the number of in-range objects is <= max_objects.
        This prevents trackpy subnet explosions from noise frames.
    min_thresh : int or None
        When thresh=None (auto mode), the computed threshold is floored to this
        value.  Useful for backsub frames with L-shaped histograms where
        auto-threshold methods can land at very low pixel values (2–4) and
        admit noise as detections.  Typical value: 5.
    """
    # Convert PIL.Image or other inputs to numpy
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    gray = img

    # Use provided threshold or compute new one
    if thresh is None:
        method_fn = THRESH_METHODS.get(thresh_method, filters.threshold_otsu)
        thresh = float(method_fn(gray))
        print(f'auto threshold ({thresh_method}): {thresh:.1f}', end='')

        if min_thresh is not None and thresh < min_thresh:
            thresh = float(min_thresh)
            print(f'  → raised to min_thresh={min_thresh}', end='')

        # Binary search upward if too many objects detected
        if max_objects is not None:
            props_check, _ = _count_inrange(gray, thresh, min_area, max_area, illumination)
            if len(props_check) > max_objects:
                lo, hi = int(thresh), 255
                while lo < hi - 1:
                    mid = (lo + hi) // 2
                    p, _ = _count_inrange(gray, mid, min_area, max_area, illumination)
                    if len(p) <= max_objects:
                        hi = mid
                    else:
                        lo = mid
                thresh = hi
                print(f'  → raised to {thresh} (max_objects={max_objects})', end='')
        print()

    if illumination == 0:
        bw = gray > thresh   # worms lighter
    else:
        bw = gray < thresh   # worms darker

    bw = morphology.remove_small_objects(bw, min_size=min_area)

    labeled = measure.label(bw)
    props = measure.regionprops(labeled)

    # Filter props by area and build mask only from in-range objects
    props_filtered = [p for p in props if min_area <= p.area_convex <= max_area]
    keep_labels = np.array([p.label for p in props_filtered])
    mask = np.isin(labeled, keep_labels)

    return mask, props_filtered, thresh

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

def show_background(input_video=None, input_file=None, first_frame=None, avg_frame=None, num_frames=None, frames_to_avg=10):

    # make sure frames are uint8:
    if first_frame.ndim == 3 and first_frame.shape[-1] == 3:
        first_frame = rgb2gray(first_frame)
        first_frame = (first_frame * 255).astype(np.uint8)
    elif first_frame.ndim == 2:
        first_frame = first_frame.astype(np.uint8)

    if avg_frame.ndim == 3 and vg_frame.shape[-1] == 3:
        avg_frame = rgb2gray(first_frame)
        avg_frame = (avg_frame * 255).astype(np.uint8)
    elif avg_frame.ndim == 2:
        avg_frame = avg_frame.astype(np.uint8)
        
    avg_background = avg_frame

    # this avoids wraparound values due to negatives after subtraction:
    # backsubbed = cv2.absdiff(first_frame, avg_background) 

    # clip negative values to zero
    backsubbed = first_frame.astype(np.int16) - avg_background.astype(np.int16)
    backsubbed = np.clip(backsubbed, 0, 255).astype(np.uint8)

    # # convert background subtracted image back to uint8 (0-255 integers):
    min_val = np.min(backsubbed)
    max_val = np.max(backsubbed)
    if max_val - min_val == 0:
        # Handle case of a flat image to avoid division by zero
        backsubbed = np.zeros_like(image_float, dtype=np.uint8)
    else:
        # normalize across the range of values in the subtracted image:
        backsubbed = (backsubbed - min_val) / (max_val - min_val)
        
        # Scale to 0-255 and convert to uint8
        backsubbed = np.round(backsubbed * 255).astype(np.uint8)
        # backsubbed = filters.gaussian(backsubbed, sigma=5, preserve_range=True)
    
    ### check for uneven luminance ####
    background_diag = np.diag(avg_background)
    background_invdiag = np.diag(np.fliplr(avg_background))
    backsubbed_diag = np.diag(backsubbed)
    first_frame_diag = np.diag(first_frame)
    x = np.arange(len(background_diag))

    coefficients_diag = np.polyfit(x, background_diag, 1)
    coefficients_invdiag = np.polyfit(x, background_invdiag, 1)

    print(f"Slope of background top-left to bottom right: {coefficients_diag[0]}")
    print(f"Slope of background bottom-left to top right: {coefficients_invdiag[0]}")
    

    if np.abs(coefficients_diag[0]) or np.abs(coefficients_invdiag[0]) > 0.02:
        print("Significantly uneven background luminance, need to subtract background")
        backsub = True
        f = plt.figure(figsize = (6,12))
        f.add_subplot(3,1,1)
        plt.imshow(avg_background, cmap="gray")
        plt.title(f"Background luminance original and corrected")
        f.add_subplot(3,1,2)
        plt.imshow(first_frame, cmap="gray")
        f.add_subplot(3,1,3)
        plt.imshow(backsubbed, cmap="gray")

        plt.figure(figsize=(10, 6))
        plt.plot(x,background_diag, label='background')
        # plt.plot(x,background_invdiag)
        plt.plot(x,backsubbed_diag, label='subtracted')
        plt.plot(x,first_frame_diag, label='first frame')
        plt.legend()  
        # imiter_vid = input_video # should be in imiter format
        print(f"returning background image of type {type(avg_background)}")
        print(f"background subtracted image is type {type(backsubbed)}")

    return(avg_background, backsub)

def convert_8bit(frame):
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = rgb2gray(frame)
        frame = (frame * 255).astype(np.uint8)
    elif frame.ndim == 2:
        frame = frame.astype(np.uint8)
    return(frame)

def float_to8bit(frame):
    # convert background subtracted or averaged image back to uint8 (0-255 integers):
    min_val = np.min(frame)
    max_val = np.max(frame)
    if max_val - min_val == 0:
        # Handle case of a flat image to avoid division by zero
        frame = np.zeros_like(frame, dtype=np.uint8)
    else:
        # normalize across the range of values in the subtracted image:
        frame = (frame - min_val) / (max_val - min_val)
        
        # Scale to 0-255 and convert to uint8
        frame = np.round(frame * 255).astype(np.uint8)
        # print(np.max(frame), np.min(frame))
    return(frame)

def subtract_background(frame, average_frame, normalize=True):
    # Subtract background and clip negative values to zero.
    # normalize=True (default): stretch result to full 0-255 range, which
    #   improves contrast and makes Otsu auto-thresholding work correctly
    #   on the background-subtracted image.
    # normalize=False: preserve original intensity scale (useful if you want
    #   to apply a fixed manual threshold to the raw subtracted values).
    backsubbed = frame.astype(np.int16) - average_frame.astype(np.int16)
    backsubbed = np.clip(backsubbed, 0, 255).astype(np.uint8)
    if normalize:
        backsubbed = float_to8bit(backsubbed)
    return backsubbed

def process_video(min_area, max_area, thresh, input_video=None, output_path=None, max_frames=None, fps=None, save_as="mp4", illumination=0, subsample=None, backsub=None, background=None, average_frame=None):
    """
    Pre-Process TIFF stack or video, to preview the annotation of worms, and optimize settings. Save video with objects detected as MP4 or TIFF stack.

    Parameters
    ----------
    input_video : imiter object
        Video read in via imiter(vid) - imageio v3. 
    output_path : str or None
        Path to save annotated output (mp4 or tif). If None, nothing saved.
    max_frames : int or None
        Process only first N frames (for testing).
    save_as : str
        "mp4" or "tif"
    subsample : Frames to subsample. Video length will be n/subsample
    """
    os.makedirs(output_path,exist_ok=True)

    import time

    frames = []
    # ----- convert to grayscale 8-bit ---------
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        start_time = time.time()
        for i, frame in enumerate(input_video):
            if i % subsample == 0: # this is zero only for multiples of subsample
                if i / subsample <= max_frames: # read in only first 50 frames of subsample
                    if frame.ndim == 3 and frame.shape[-1] == 3:
                        if i == 1:
                            print("converting to grayscale images")
                        frame = rgb2gray(frame)
                        frame = (frame * 255).astype(np.uint8)
                    elif frame.ndim == 2:
                        if i == 1: 
                            print("already grayscale, converting to 8-bit")
                        frame = frame.astype(np.uint8)
                    if backsub == True:
                        subtracted = subtract_background(frame = frame, average_frame = average_frame)
                        if i == 0:
                            thresh = filters.threshold_otsu(subtracted)
                        frame = subtracted
                    # Append the processed frame to the list
                    frames.append(frame)
        end_time = time.time()
    print(f"Converting frames using imiter of video took {end_time - start_time} seconds")
    
    first_frame = frames[0]

    # result_path = os.path.join(f"{os.path.splitext(input_path)[0]}_results")
    print(f"results will be saved to {output_path}")
    workingDir = os.getcwd()
    
    if save_as == "mp4" and output_path:
        writer = imageio.get_writer(os.path.join(output_path,"worms_annotated.mp4"), fps=fps)
    else:
        writer = None

    overlays = []  # store frames if saving as TIFF

    # --- Get threshold from first frame ---
    if thresh is None:
        mask1, props1, global_thresh = preprocess_frame(first_frame, 
                                                        min_area, 
                                                        max_area, 
                                                        thresh, 
                                                        illumination=illumination)
        print("Auto calculated threshold value is",global_thresh)
 
    else:
        global_thresh = thresh
        print("Manual threshold value is",global_thresh)
        mask1, props1, _ = preprocess_frame(first_frame,
                                            min_area,
                                            max_area,
                                            thresh=global_thresh,
                                            illumination=illumination)
        
    # ------ plot first frame mask with histogram of object sizes ------
    areas = []
    for prop in props1:
        area = prop.area
        areas.append(area)
    print('Mean area of objects in pixels is', np.mean(areas))
    if (np.mean(areas) < min_area) | (np.mean(areas) > max_area):
        print("which is outside of your min and max area estimate")
    f = plt.figure(figsize=(6,4))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.imshow(mask1, cmap="gray")
    ax.set_title(f"Frame 1, worms: {len(props1)}")
    # ax.axis("off")
    ax2.hist(areas, bins = 10)
    ax2.set_title("Areas of detected objects")
    ax.set_xlabel("Area")
    ax.set_ylabel("Number of objects")
    plt.tight_layout()
    plt.show(block=False)

    all_areas = []
    for i, frame in enumerate(frames):
        if max_frames and i >= max_frames:
            break

        _, props, _ = preprocess_frame(frame,
                                       min_area,
                                       max_area,
                                       thresh=global_thresh,
                                       illumination=illumination)
                                    # Debug: show first frame

        overlay = draw_boxes_labels_gray(frame, props)
        frame_areas = [prop.area for prop in props]
        all_areas.extend(frame_areas)

        if save_as == "mp4" and writer:
            writer.append_data(overlay)
        elif save_as == "tif":
            overlays.append(overlay)
    print(len(all_areas))
    
    f2 = plt.figure(figsize=(6,4))
    plt.hist(all_areas, bins = 20)
    plt.title(f"{len(all_areas)} objects detected (unfiltered, 50 frames)")
    plt.show(block=False)
    
    if save_as == "mp4" and writer:
        writer.close()
    elif save_as == "tif" and output_path:
        tifffile.imwrite(os.path.join(output_path,"worms_annotated.tif"), np.array(overlays), photometric="minisblack")
    return global_thresh

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
            print("error - too many params")
    return overlay

def collect_detections(stack, global_thresh, min_area, max_area, illumination=0):
    # this is just a wrapper for preprocess_frame to track centroids
    records = []
    for frame_no, frame in enumerate(stack):
        print(f"\rThresholding and getting centroids for frame: {frame_no}", end="", flush=True)
        time.sleep(0.0001)
        _, props, _ = preprocess_frame(frame,
                                       min_area=min_area,
                                       max_area=max_area,
                                       thresh=global_thresh,
                                       illumination=illumination)
        for prop in props:
            y, x = prop.centroid  # note (row, col) = (y, x)
            pixel_vals = frame[prop.slice][prop.image]

            records.append({
                "frame": frame_no,
                "x": x,
                "y": y,
                "area": prop.area,
                "major_axis": prop.axis_major_length,
                "minor_axis": prop.axis_minor_length,
                "orientation": prop.orientation,
                "eccentricity": prop.eccentricity,
                "euler_num": prop.euler_number,
                "solidity": prop.solidity,
                "area_convex": prop.area_convex,
                # fluorescence metrics
                "mean_intensity": np.mean(pixel_vals),
                "median_intensity": np.median(pixel_vals),
                "max_intensity": np.max(pixel_vals),
                "integrated_intensity": np.sum(pixel_vals)
            })
    print()  # newline after frame counter
    return pd.DataFrame(records)

def link_tracks(detections, search_range=50, memory=3, quiet=False):
    """
    Link worm detections into tracks.
    search_range: max distance a worm can move between frames
    memory: how many frames a worm can vanish and still be linked
    """
    print("Linking tracks")
    if quiet:
        tp.quiet()
    try:
        linked = tp.link_df(detections, search_range=search_range, memory=memory)
    except tp.linking.utils.SubnetOversizeException as e:
        raise RuntimeError(
            f"\nTrackpy SubnetOversizeException: search_range={search_range} is too large "
            f"for the density of detected objects.\n"
            f"Try reducing 'search_range' in your config file (e.g. to {max(1, search_range // 2)}) "
            f"or increasing 'min_area'/'max_area' to reduce false detections.\n"
            f"Original error: {e}"
        ) from None
    return linked

def draw_tracks_gray(frame, props, tracks, frame_no):
    overlay = np.array(frame).copy() # should inherit frames from above, no need to convert again
    if overlay.shape[-1] == 3:
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

def filter_boundary_particles(tracks, frame_shape, margin_px):
    """Remove particles whose median centroid is within margin_px of any frame edge.

    Censors LED-ring artifacts and plate-edge detections. Use half a worm-length
    (~major_axis median / 2) as margin_px.

    Parameters
    ----------
    tracks : pd.DataFrame  with 'particle', 'x', 'y' columns
    frame_shape : (H, W) tuple — shape of the video frames
    margin_px : float — exclusion zone width in pixels from each frame edge

    Returns
    -------
    filtered : pd.DataFrame
    n_removed : int — number of particles removed
    """
    H, W = frame_shape
    grp = tracks.groupby("particle")
    med_x = grp["x"].median()
    med_y = grp["y"].median()
    dist_to_edge = pd.concat([med_x, W - med_x, med_y, H - med_y], axis=1).min(axis=1)
    keep = dist_to_edge[dist_to_edge >= margin_px].index
    n_removed = len(dist_to_edge) - len(keep)
    if n_removed:
        print(f"  filter_boundary_particles: removed {n_removed} particles "
              f"with median centroid within {margin_px:.0f} px of frame edge")
    return tracks[tracks["particle"].isin(keep)].copy(), n_removed


def stitch_tracks(tracks, max_gap_frames, max_gap_pixels):
    """Post-hoc track stitching: merge track fragments where one ends close in
    space and time to where another begins.

    Intended for pumping mode where worms are relatively stationary and broken
    tracks are caused by temporary detection failure rather than identity confusion.

    Parameters
    ----------
    tracks : pd.DataFrame
        Must have 'particle', 'frame', 'x', 'y'.
    max_gap_frames : int
        Maximum frame gap between the end of one track and the start of another.
    max_gap_pixels : float
        Maximum pixel distance between endpoint and startpoint to be stitched.

    Returns
    -------
    tracks : pd.DataFrame with relabelled particle IDs
    n_stitched : int  number of stitches applied
    """
    # Collect per-particle start/end info
    endpoints = []
    for pid, g in tracks.groupby("particle"):
        g = g.sort_values("frame")
        endpoints.append({
            "particle":    pid,
            "start_frame": int(g["frame"].iloc[0]),
            "start_x":     float(g["x"].iloc[0]),
            "start_y":     float(g["y"].iloc[0]),
            "end_frame":   int(g["frame"].iloc[-1]),
            "end_x":       float(g["x"].iloc[-1]),
            "end_y":       float(g["y"].iloc[-1]),
        })

    # Find candidate (from → to) pairs
    candidates = []
    for a in endpoints:
        for b in endpoints:
            if a["particle"] == b["particle"]:
                continue
            gap = b["start_frame"] - a["end_frame"]
            if not (0 < gap <= max_gap_frames):
                continue
            dist = np.sqrt((b["start_x"] - a["end_x"]) ** 2 +
                           (b["start_y"] - a["end_y"]) ** 2)
            if dist <= max_gap_pixels:
                candidates.append({
                    "from": a["particle"],
                    "to":   b["particle"],
                    "gap":  gap,
                    "dist": dist,
                    "cost": dist + gap,
                })

    # Greedy match: sort by cost, each end/start used at most once
    candidates.sort(key=lambda c: c["cost"])
    used_ends   = set()
    used_starts = set()
    merges = []
    for c in candidates:
        if c["from"] not in used_ends and c["to"] not in used_starts:
            merges.append((c["from"], c["to"]))
            used_ends.add(c["from"])
            used_starts.add(c["to"])

    if not merges:
        return tracks, 0

    # Build canonical ID mapping (resolve chains A→B→C to all→A)
    mapping = {}
    for frm, to in merges:
        canonical = mapping.get(frm, frm)
        mapping[to] = canonical
    # Second pass: ensure chained remaps are fully resolved
    for key in list(mapping):
        root = mapping[key]
        while root in mapping:
            root = mapping[root]
        mapping[key] = root

    tracks = tracks.copy()
    tracks["particle"] = tracks["particle"].map(lambda p: mapping.get(p, p))
    print(f"Stitched {len(merges)} track pairs "
          f"({tracks['particle'].nunique()} particles remaining)")
    return tracks, len(merges)


def filter_area_change(tracks, max_area_cv = 0.2):
    """
    Remove worms (particles) with fewer than `min_length` frames.
    """
    
    grouped_tracks = tracks.groupby("particle")
    # mean_area = grouped_tracks['area'].agg(
    # mean_area = grouped_tracks[)
    area_variation = grouped_tracks['area'].agg(
    cv=lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0)

    # Set a maximum allowed coefficient of variation for the area
    max_area_cv = max_area_cv # Example: allow up to a 10% change relative to the mean

    # Get the IDs of the particles to keep
    particles_to_keep = area_variation[area_variation['cv'] <= max_area_cv].index

    # Filter the original trajectories DataFrame
    stable_trajectories = tracks[tracks['particle'].isin(particles_to_keep)]
    return stable_trajectories

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
    plt.show(block=False)

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

def _count_consecutive_true(series):
    result = np.zeros(len(series), dtype=int)
    count = 0
    for i, val in enumerate(series):
        if val:
            count += 1
            result[i] = count
        else:
            count = 0
    return result


def calculate_speed_parameters(df,
                               pixel_length=60,
                               frame_rate=10,
                               window_size=7,
                               speed_threshold=0.5,
                               smooth_window=5,
                               min_displacement_for_angle=1.5,
                               max_instantaneous_speed=5.0,
                               stability_threshold=0.90):
    """Phase 1 (centroid mode): smooth positions and compute speed/directional-stability.
    Output is suitable for centroid-mode CSV or as input to calculate_postural_states().
    """
    df = df.copy()
    df = df.sort_values(['particle', 'frame'])

    df['x_smooth'] = df['x']
    df['y_smooth'] = df['y']
    for particle in df['particle'].unique():
        mask = df['particle'] == particle
        particle_data = df.loc[mask].copy()
        if len(particle_data) >= smooth_window:
            df.loc[mask, 'x_smooth'] = median_filter(particle_data['x'].values, size=smooth_window, mode='nearest')
            df.loc[mask, 'y_smooth'] = median_filter(particle_data['y'].values, size=smooth_window, mode='nearest')

    df['vx_pixels'] = df.groupby('particle')['x_smooth'].diff()
    df['vy_pixels'] = df.groupby('particle')['y_smooth'].diff()
    # Divide by actual frame gap so speed is correct across trackpy linking gaps
    # (diff spans N frames but default assumes 1 frame, inflating speed N-fold)
    df['_frame_gap'] = df.groupby('particle')['frame'].diff().fillna(1).clip(lower=1)
    # vx/vy are signed velocity components (mm/s); speed is their scalar magnitude
    df['vx'] = (df['vx_pixels'] / pixel_length) * frame_rate / df['_frame_gap']
    df['vy'] = (df['vy_pixels'] / pixel_length) * frame_rate / df['_frame_gap']
    df['speed_instantaneous'] = np.sqrt(df['vx']**2 + df['vy']**2).clip(upper=max_instantaneous_speed)
    df['speed'] = (df.groupby('particle')['speed_instantaneous']
                   .transform(lambda x: x.rolling(window_size, min_periods=1, center=True).median()))
    df['displacement_mm'] = np.sqrt(df['vx_pixels']**2 + df['vy_pixels']**2) / pixel_length / df['_frame_gap']
    df = df.drop(columns=['_frame_gap'])

    df['movement_angle'] = np.nan
    significant_motion = df['displacement_mm'] > min_displacement_for_angle
    df.loc[significant_motion, 'movement_angle'] = np.arctan2(
        df.loc[significant_motion, 'vx_pixels'],
        df.loc[significant_motion, 'vy_pixels']
    )
    df['movement_angle'] = df.groupby('particle')['movement_angle'].ffill()

    df['major_axis_x'] = np.sin(df['orientation'])
    df['major_axis_y'] = np.cos(df['orientation'])
    df['major_axis_component'] = df['vx'] * df['major_axis_x'] + df['vy'] * df['major_axis_y']

    df['d_orientation'] = df.groupby('particle')['orientation'].diff()
    df['d_orientation'] = np.where(
        df['d_orientation'] > np.pi/2, df['d_orientation'] - np.pi,
        np.where(df['d_orientation'] < -np.pi/2, df['d_orientation'] + np.pi, df['d_orientation'])
    )
    df['angular_velocity'] = df['d_orientation'] * frame_rate

    df['sin_angle'] = np.sin(df['movement_angle'])
    df['cos_angle'] = np.cos(df['movement_angle'])
    df['mean_sin'] = (df.groupby('particle')['sin_angle']
                      .transform(lambda x: x.rolling(window_size, min_periods=1, center=False).mean()))
    df['mean_cos'] = (df.groupby('particle')['cos_angle']
                      .transform(lambda x: x.rolling(window_size, min_periods=1, center=False).mean()))
    df['mean_movement_angle'] = np.arctan2(df['mean_sin'], df['mean_cos'])
    df['direction_stability'] = np.sqrt(df['mean_sin']**2 + df['mean_cos']**2)

    df['mean_speed'] = (df.groupby('particle')['speed']
                        .transform(lambda x: x.rolling(window_size, min_periods=1).mean()))
    df['speed_std'] = (df.groupby('particle')['speed']
                       .transform(lambda x: x.rolling(window_size, min_periods=1).std()))
    df['speed_cv'] = df['speed_std'] / (df['mean_speed'] + 1e-6)

    df['angle_from_mean'] = df['movement_angle'] - df['mean_movement_angle']
    df['angle_from_mean'] = np.where(
        df['angle_from_mean'] > np.pi, df['angle_from_mean'] - 2*np.pi,
        np.where(df['angle_from_mean'] < -np.pi, df['angle_from_mean'] + 2*np.pi, df['angle_from_mean'])
    )

    df['is_moving'] = df['speed'] > speed_threshold
    df['is_directionally_stable'] = (
        df['is_moving'] &
        (df['direction_stability'] > stability_threshold) &
        (df['mean_speed'] > speed_threshold) &
        (df['speed_cv'] < 0.6)
    )
    df['stable_run_counter'] = (df.groupby('particle')['is_directionally_stable']
                                 .transform(_count_consecutive_true))

    # Per-particle area stability: CV of area across all frames.
    # After illumination normalisation, real worms should have low CV;
    # fragmented tracks or false detections show high CV.
    if 'area' in df.columns:
        grp_area = df.groupby('particle')['area']
        area_mean = grp_area.transform('mean')
        area_std  = grp_area.transform('std').fillna(0)
        df['area_cv'] = area_std / (area_mean + 1e-6)
    return df


def calculate_postural_states(df,
                               frame_rate=10,
                               direction_threshold=np.pi * 2 / 3,
                               speed_threshold=0.5,
                               min_run_length=10,
                               reversal_persistence=2,
                               pirouette_speed_threshold=0.3,
                               pirouette_eccentricity_threshold=0.8,
                               min_pirouette_duration=2,
                               area_reliability_threshold=0.70,
                               merge_reversal_gap=5):
    """Phase 2 (postural mode): add reversal/pirouette/movement_type columns.
    Requires df to have already been through calculate_speed_parameters().

    Reversal detection uses a stable long-window reference direction (2 * frame_rate frames)
    rather than the short window_size rolling mean. The short rolling mean catches up to the
    reversal direction within window_size frames, making ongoing reversals invisible and
    falsely detecting the return-to-forward as a new large angle event. The long window stays
    pointing forward throughout a typical short reversal, giving correct onset detection and
    avoiding false detection of the post-reversal turn.

    Reversal exit: the backward heading is stored at entry; exit fires when current
    movement_angle differs by > π/2 from that stored heading (worm has turned away from
    its reversal direction), or on speed drop, or after 3-second timeout.

    Area reliability gate (area_reliability_threshold): frames where the segmented area
    drops below this fraction of the particle's median area are flagged unreliable.
    Reversal entry is blocked on unreliable frames (partial body thresholding or worm
    partially out of frame give jittery centroids that look like direction reversals).

    Merge gap (merge_reversal_gap): after detection, adjacent reversal events on the same
    particle separated by ≤ this many frames are merged into one. This repairs splits
    caused by brief unreliable frames mid-reversal, ensuring a single behavioural event
    is not double-counted.
    """
    df = df.copy()
    max_reversal_frames = int(3 * frame_rate)
    stable_window = int(2 * frame_rate)  # long enough to resist shifting during a short reversal

    # Long-window reference direction for reversal onset detection.
    # Uses a backward-looking rolling window so it represents where the worm was heading
    # before any potential reversal started. The short window_size mean catches up within
    # a few frames; this one stays pointing forward for the full duration of a short reversal.
    df['_ma_filled'] = df['movement_angle'].fillna(0)
    df['_ma_sin'] = np.sin(df['_ma_filled'])
    df['_ma_cos'] = np.cos(df['_ma_filled'])
    df['_stable_sin'] = (df.groupby('particle')['_ma_sin']
                         .transform(lambda x: x.rolling(stable_window, min_periods=1).mean()))
    df['_stable_cos'] = (df.groupby('particle')['_ma_cos']
                         .transform(lambda x: x.rolling(stable_window, min_periods=1).mean()))
    stable_angle = np.arctan2(df['_stable_sin'], df['_stable_cos'])
    raw_diff = df['_ma_filled'] - stable_angle
    df['_angle_from_stable'] = ((raw_diff + np.pi) % (2 * np.pi) - np.pi).abs()
    df.drop(columns=['_ma_filled', '_ma_sin', '_ma_cos', '_stable_sin', '_stable_cos'],
            inplace=True)

    # ========== PIROUETTE FRAME PRE-DETECTION ==========
    # Compute pirouette-frame flag before the reversal loop so omega-turn frames can be
    # excluded from reversal entry. An omega turn (low eccentricity + low speed) should
    # never be mislabeled as a reversal even if the CoM moves in an off-axis direction.
    df['is_pirouette_frame'] = (
        (df['eccentricity'] < pirouette_eccentricity_threshold) &
        (df['speed'] < pirouette_speed_threshold)
    )

    # ========== AREA RELIABILITY FLAG ==========
    # Frames where area falls below area_reliability_threshold * per-particle median are
    # unreliable: partial body thresholding (edge illumination) or worm partially out of
    # frame cause centroid jitter that mimics a direction reversal. Block entry on these.
    median_area = df.groupby('particle')['area'].transform('median')
    df['_reliable_area'] = df['area'] >= area_reliability_threshold * median_area

    # ========== REVERSAL DETECTION ==========
    df['is_reversal'] = False
    df['reversal_id'] = 0

    for particle in df['particle'].unique():
        mask = df['particle'] == particle
        particle_df = df.loc[mask]

        angles = particle_df['_angle_from_stable'].values
        speeds = particle_df['speed'].values
        movement_angles = particle_df['movement_angle'].fillna(0).values
        is_pir_frame = particle_df['is_pirouette_frame'].values
        reliable_area = particle_df['_reliable_area'].values
        n = len(particle_df)

        # Rolling max speed over the half-second immediately preceding each frame
        # (not including the current frame). Reversal entry requires that the worm
        # was actively locomoting in this window: worms that have been paused for
        # ≥ 0.5 s cannot initiate a reversal, so this gate blocks angle-noise
        # false positives during genuine pauses.
        entry_lookback = max(int(0.5 * frame_rate), reversal_persistence + 1)
        recent_max_speed = np.array([
            speeds[max(0, i - entry_lookback):i].max() if i > 0 else 0.0
            for i in range(n)
        ])

        is_rev = np.zeros(n, dtype=bool)
        rev_id = np.zeros(n, dtype=int)

        in_reversal = False
        reversal_counter = 0
        reversal_start_i = 0
        reversal_angle = 0.0  # movement direction at reversal entry (backward heading)
        pending = 0            # consecutive qualifying frames

        for i in range(n):
            # Entry: large angle from stable mean + not in an omega turn + reliable area.
            # Speed is NOT required at entry because a reversal always begins with deceleration
            # through near-zero speed; requiring speed > threshold would consistently delay
            # onset detection by 1-2 frames. The angle_from_stable signal plus pirouette
            # exclusion are sufficient to gate entry correctly.
            # Unreliable-area frames are excluded to prevent centroid jitter from partial
            # body thresholding (edge of plate, variable illumination) from triggering entry.
            above = (angles[i] > direction_threshold
                     and not is_pir_frame[i]
                     and reliable_area[i]
                     and recent_max_speed[i] > speed_threshold)

            if not in_reversal:
                if above:
                    pending += 1
                    if pending >= reversal_persistence:
                        in_reversal = True
                        reversal_counter += 1
                        reversal_start_i = i - reversal_persistence + 1
                        reversal_angle = movement_angles[i]
                        is_rev[reversal_start_i:i + 1] = True
                        rev_id[reversal_start_i:i + 1] = reversal_counter
                else:
                    pending = 0
            else:
                # Exit when worm is no longer heading in the reversal direction.
                # Compare current movement angle to the stored backward heading at entry;
                # diff > π/2 means the worm has turned more than 90° away from backward.
                diff = movement_angles[i] - reversal_angle
                diff = (diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
                no_longer_backward = abs(diff) > np.pi / 2

                if (no_longer_backward
                        or speeds[i] < speed_threshold * 0.3
                        or (i - reversal_start_i) >= max_reversal_frames):
                    in_reversal = False
                    pending = 0
                else:
                    is_rev[i] = True
                    rev_id[i] = reversal_counter

        df.loc[mask, 'is_reversal'] = is_rev
        df.loc[mask, 'reversal_id'] = rev_id

    # ========== MERGE ADJACENT REVERSALS ==========
    # Brief unreliable frames mid-reversal (area drop, out-of-frame) can cause the exit
    # condition to fire and re-entry to follow, splitting one behavioural reversal into two
    # events. Merge pairs of reversal events on the same particle that are separated by
    # ≤ merge_reversal_gap frames so they are counted as a single reversal.
    if merge_reversal_gap > 0:
        for particle in df['particle'].unique():
            mask = df['particle'] == particle
            particle_df = df.loc[mask]
            is_rev = particle_df['is_reversal'].values.copy()
            rev_id = particle_df['reversal_id'].values.copy()
            n = len(is_rev)

            # Locate starts and ends of reversal runs (local indices within particle)
            padded = np.concatenate([[False], is_rev, [False]])
            starts = np.where(np.diff(padded.astype(int)) == 1)[0]
            ends   = np.where(np.diff(padded.astype(int)) == -1)[0]  # first non-reversal index

            for k in range(len(starts) - 1):
                gap = starts[k + 1] - ends[k]  # frames between end of event k and start of k+1
                if gap <= merge_reversal_gap:
                    # Fill gap frames with reversal; keep the earlier reversal_id
                    is_rev[ends[k]:starts[k + 1]] = True
                    rev_id[ends[k]:starts[k + 1]] = rev_id[ends[k] - 1]

            df.loc[mask, 'is_reversal'] = is_rev
            df.loc[mask, 'reversal_id'] = rev_id

    df.drop(columns=['_reliable_area'], inplace=True)

    prev_rev = df.groupby('particle')['is_reversal'].shift(1).fillna(False).infer_objects(copy=False)
    df['reversal_start'] = df['is_reversal'] & ~prev_rev
    df['reversal_end'] = ~df['is_reversal'] & prev_rev

    # ========== PIROUETTE DETECTION (full labeling) ==========
    # is_pirouette_frame was already computed above; now add duration and sustained flag.
    df['pirouette_duration'] = (df.groupby('particle')['is_pirouette_frame']
                                 .transform(_count_consecutive_true))
    df['is_pirouette'] = df['pirouette_duration'] >= min_pirouette_duration

    prev_pirouette = df.groupby('particle')['is_pirouette'].shift(1)
    df['pirouette_start'] = df['is_pirouette'] & (prev_pirouette != True)
    df['pirouette_type'] = None

    # Classify pirouette type: preceded by a reversal within 5 frames = post_reversal
    frames_after_reversal_threshold = 5
    df['recent_reversal'] = (
        df.groupby('particle')['reversal_start']
        .transform(lambda x: x.rolling(frames_after_reversal_threshold, min_periods=1).sum() > 0)
    )
    df.loc[df['pirouette_start'] & df['recent_reversal'], 'pirouette_type'] = 'post_reversal'
    df.loc[df['pirouette_start'] & ~df['recent_reversal'], 'pirouette_type'] = 'spontaneous'

    for particle in df['particle'].unique():
        mask = df['particle'] == particle
        particle_df = df.loc[mask].copy()
        current_type = None
        pirouette_types = []
        for is_pir, pir_type in zip(particle_df['is_pirouette'], particle_df['pirouette_type']):
            if pir_type is not None:
                current_type = pir_type
            pirouette_types.append(current_type if is_pir else None)
        df.loc[mask, 'pirouette_type'] = pirouette_types

    # ========== CLASSIFY MOVEMENT TYPE ==========
    df['movement_type'] = 'stationary'
    df.loc[df['is_moving'] & ~df['is_directionally_stable'], 'movement_type'] = 'meandering'
    df.loc[df['is_directionally_stable'] & (df['stable_run_counter'] < min_run_length), 'movement_type'] = 'short_run'
    df.loc[df['is_directionally_stable'] & (df['stable_run_counter'] >= min_run_length), 'movement_type'] = 'forward_run'
    df.loc[df['is_reversal'], 'movement_type'] = 'reversal'
    df.loc[df['is_pirouette'], 'movement_type'] = 'pirouette'

    df['frames_since_reversal'] = 0
    for particle in df['particle'].unique():
        mask = df['particle'] == particle
        particle_df = df.loc[mask].copy()
        frames_since = 0
        frames_list = []
        for is_rev_start in particle_df['reversal_start']:
            frames_since = 0 if is_rev_start else frames_since + 1
            frames_list.append(frames_since)
        df.loc[mask, 'frames_since_reversal'] = frames_list

    df['frames_since_pirouette'] = 0
    for particle in df['particle'].unique():
        mask = df['particle'] == particle
        particle_df = df.loc[mask].copy()
        frames_since = 0
        frames_list = []
        for is_pir_start in particle_df['pirouette_start']:
            frames_since = 0 if is_pir_start else frames_since + 1
            frames_list.append(frames_since)
        df.loc[mask, 'frames_since_pirouette'] = frames_list

    df.drop(columns=['_angle_from_stable'], inplace=True)
    return df


def calculate_motion_parameters(df,
                                pixel_length=60,           # pixels/mm
                                frame_rate=10,             # frames/sec
                                window_size=7,
                                direction_threshold=np.pi * 2 / 3,
                                speed_threshold=0.5,       # mm/s
                                min_run_length=10,
                                reversal_persistence=2,
                                smooth_window=5,
                                min_displacement_for_angle=1.5,  # mm
                                pirouette_speed_threshold=0.3,   # mm/s
                                pirouette_eccentricity_threshold=0.8,
                                min_pirouette_duration=2,
                                max_instantaneous_speed=5.0,     # mm/s
                                stability_threshold=0.90):
    """Convenience wrapper: calculate_speed_parameters() then calculate_postural_states()."""
    df = calculate_speed_parameters(
        df,
        pixel_length=pixel_length,
        frame_rate=frame_rate,
        window_size=window_size,
        speed_threshold=speed_threshold,
        smooth_window=smooth_window,
        min_displacement_for_angle=min_displacement_for_angle,
        max_instantaneous_speed=max_instantaneous_speed,
        stability_threshold=stability_threshold,
    )
    return calculate_postural_states(
        df,
        frame_rate=frame_rate,
        direction_threshold=direction_threshold,
        speed_threshold=speed_threshold,
        min_run_length=min_run_length,
        reversal_persistence=reversal_persistence,
        pirouette_speed_threshold=pirouette_speed_threshold,
        pirouette_eccentricity_threshold=pirouette_eccentricity_threshold,
        min_pirouette_duration=min_pirouette_duration,
    )

# Example usage and interpretation
def classify_movement(row):
    """
    Classify whether the object is moving forward or backward along its major axis.
    """
    if pd.isna(row['major_axis_component']):
        return 'no_data'
    elif abs(row['major_axis_component']) < 0.1:  # adjust threshold as needed
        return 'stationary'
    elif row['major_axis_component'] > 0:
        return 'forward'
    else:
        return 'backward'

def create_annotated_video(video_path, df, particle_id, output_folder,
                           pixel_length=60, frame_rate=10,
                           global_thresh=None, min_area=100, max_area=5000,
                           illumination=0, crop_size=300, source_crop_px=100,
                           show_mask=True):
    """
    Create a video of a single particle with movement type annotations.
    
    Parameters:
    -----------
    video_path : str
        Path to the original video file
    df : pandas DataFrame
        DataFrame with tracking data and movement classifications
    particle_id : int
        ID of the particle to visualize
    output_folder : str
        Folder to save the output video
    pixel_length : float
        Pixels per mm (for scale bar)
    frame_rate : int
        Frames per second
    global_thresh : int
        Threshold value for segmentation (if None, will estimate)
    min_area : int
        Minimum area for objects
    max_area : int
        Maximum area for objects
    illumination : int
        0 for dark objects on bright background, 1 for bright on dark
    crop_size : int
        Size of the cropped region around the particle
    show_mask : bool
        If True, overlay colored mask on worm. If False, just show bounding box
    """
    
    # Get data for this particle
    particle_df = df[df['particle'] == particle_id].copy()
    particle_df = particle_df.sort_values('frame')
    particle_median_area = particle_df['area'].median() if 'area' in particle_df.columns else None
    
    if len(particle_df) == 0:
        print(f"No data found for particle {particle_id}")
        return
    
    # Check if particle has all three movement types
    movement_types = particle_df['movement_type'].unique()
    has_forward = 'forward_run' in movement_types
    has_reversal = 'is_reversal' in particle_df.columns and particle_df['is_reversal'].any()
    has_pirouette = 'is_pirouette' in particle_df.columns and particle_df['is_pirouette'].any()
    
    print(f"Particle {particle_id} has:")
    print(f"  Forward runs: {has_forward}")
    print(f"  Reversals: {has_reversal}")
    print(f"  Pirouettes: {has_pirouette}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Auto-threshold if not provided
    if global_thresh is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, sample_frame = cap.read()
        if ret:
            if len(sample_frame.shape) == 3:
                sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
            from skimage import filters
            global_thresh = int(filters.threshold_otsu(sample_frame))
            print(f"Auto-detected threshold: {global_thresh}")
    
    # Create output video - with cropped view on the side
    output_width = width + crop_size + 20  # Original + crop + margin
    output_height = max(height, crop_size + 250)  # Accommodate info panel
    
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_folder, f'particle_{particle_id}_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # Colors for different movement types (BGR format)
    colors = {
        'stationary': (128, 128, 128),      # Gray
        'meandering': (0, 255, 255),        # Yellow
        'short_run': (255, 200, 100),       # Light blue
        'forward_run': (0, 255, 0),         # Green
        'reversal': (0, 0, 255),            # Red
        'pirouette': (255, 0, 255)          # Magenta
    }
    
    # Get frame range for this particle
    start_frame = int(particle_df['frame'].min())
    end_frame = int(particle_df['frame'].max())
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    prev_movement_type = None
    # Persistent state — hold last known values so gap frames don't go black
    x, y = width // 2, height // 2
    color = (128, 128, 128)
    movement_type = 'stationary'
    row = None

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        canvas[:height, :width] = frame

        frame_data = particle_df[particle_df['frame'] == frame_idx]

        if len(frame_data) > 0:
            row = frame_data.iloc[0]
            x = int(row['x'])
            y = int(row['y'])
            movement_type = row['movement_type']
            color = colors.get(movement_type, (255, 255, 255))

            # Segment and apply mask overlay when object is detected near centroid
            _, props, _ = preprocess_frame(gray_frame,
                                           min_area=min_area,
                                           max_area=max_area,
                                           thresh=global_thresh,
                                           illumination=illumination)
            best_prop = None
            min_dist = float('inf')
            for prop in props:
                prop_y, prop_x = prop.centroid
                dist = np.sqrt((prop_x - x)**2 + (prop_y - y)**2)
                # Reject objects whose area is outside 50–175 % of this particle's
                # median area so a nearby worm cannot steal the highlight.
                if particle_median_area is not None:
                    if not (0.50 * particle_median_area <= prop.area <= 1.75 * particle_median_area):
                        continue
                if dist < min_dist:
                    min_dist = dist
                    best_prop = prop

            if best_prop is not None and min_dist < 25:
                if show_mask:
                    colored_mask = np.zeros_like(frame)
                    object_mask = np.zeros((height, width), dtype=bool)
                    object_mask[best_prop.slice][best_prop.image] = True
                    colored_mask[object_mask] = color
                    canvas[:height, :width] = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

            # Transition label
            if prev_movement_type is not None and movement_type != prev_movement_type:
                transition_text = f"{prev_movement_type} -> {movement_type}"
                cv2.putText(canvas, "TRANSITION!", (width + 10, crop_size + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(canvas, transition_text, (width + 10, crop_size + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            prev_movement_type = movement_type

        # ── Crop inset from mask-only canvas (before trajectory is drawn) ────
        x_start = max(0, x - source_crop_px)
        x_end   = min(width,  x + source_crop_px)
        y_start = max(0, y - source_crop_px)
        y_end   = min(height, y + source_crop_px)
        crop = canvas[y_start:y_end, x_start:x_end].copy()

        # ── Trajectory on main view only (past 60 frames) ────────────────────
        if row is not None:
            past_frames = particle_df[
                (particle_df['frame'] <= frame_idx) &
                (particle_df['frame'] > frame_idx - 60)
            ]
            if len(past_frames) > 1:
                points = np.array([[int(r['x']), int(r['y'])] for _, r in past_frames.iterrows()])
                cv2.polylines(canvas[:height, :width], [points], False, color, 2)
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
            crop_bordered = cv2.copyMakeBorder(crop_resized, 3, 3, 3, 3,
                                               cv2.BORDER_CONSTANT, value=color)
            canvas[10:10+crop_size+6, width+10:width+10+crop_size+6] = crop_bordered

        info_x = width + 10
        info_y_start = crop_size + 100
        cv2.putText(canvas, f"Particle: {particle_id}", (info_x, info_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"Frame: {frame_idx}", (info_x, info_y_start + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"Type:", (info_x, info_y_start + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"{movement_type}", (info_x, info_y_start + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if row is not None:
            if 'speed' in row:
                cv2.putText(canvas, f"Speed: {row['speed']:.2f} mm/s",
                            (info_x, info_y_start + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            if 'eccentricity' in row:
                cv2.putText(canvas, f"Ecc: {row['eccentricity']:.2f}",
                            (info_x, info_y_start + 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        legend_y = output_height - 160
        cv2.putText(canvas, "Legend:", (10, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, (mt, col) in enumerate(colors.items()):
            y_pos = legend_y + 20 + i * 20
            cv2.rectangle(canvas, (10, y_pos - 12), (30, y_pos), col, -1)
            cv2.putText(canvas, mt, (35, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(canvas)
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"\rProcessing frame {frame_idx}/{end_frame}", end="", flush=True)
    
    cap.release()
    out.release()
    
    print(f"\nVideo saved to: {output_path}")
    return output_path


# The find_particle_with_all_behaviors function remains the same
def find_particle_with_all_behaviors(df):
    """
    Find a particle that exhibits forward run, reversal, and pirouette.
    """
    candidates = []
    
    for particle_id in df['particle'].unique():
        particle_df = df[df['particle'] == particle_id]
        
        # Check for each behavior
        has_forward = (particle_df['movement_type'] == 'forward_run').any()
        has_reversal = particle_df['is_reversal'].any() if 'is_reversal' in particle_df.columns else False
        has_pirouette = particle_df['is_pirouette'].any() if 'is_pirouette' in particle_df.columns else False
        
        behavior_count = sum([has_forward, has_reversal, has_pirouette])
        
        if behavior_count == 3:  # Has all three
            # Count how many of each
            forward_frames = (particle_df['movement_type'] == 'forward_run').sum()
            reversal_frames = particle_df['is_reversal'].sum()
            pirouette_frames = particle_df['is_pirouette'].sum()
            distance_traveled = np.sqrt( (particle_df['x'].max() - particle_df['x'].min())**2 + (particle_df['y'].max() - particle_df['y'].min())**2 ) / 60
            
            candidates.append({
                'particle': particle_id,
                'track_length': len(particle_df),
                'forward_frames': forward_frames,
                'reversal_frames': reversal_frames,
                'pirouette_frames': pirouette_frames,
                'distance_traveled': distance_traveled,
                'score': min(len(particle_df), distance_traveled)
            })
    
    if not candidates:
        print("No particle found with all three behaviors (forward, reversal, pirouette)")
        return None
    
    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    best = candidates[0]
    print(f"\nBest particle for demonstration: {best['particle']}")
    print(f"  Track length: {best['track_length']} frames")
    print(f"  Distance traveled: {best['distance_traveled']} mm")
    print(f"  Forward run frames: {best['forward_frames']}")
    print(f"  Reversal frames: {best['reversal_frames']}")
    print(f"  Pirouette frames: {best['pirouette_frames']}")
    
    return best['particle']


# ── Pumping analysis functions ────────────────────────────────────────────────

def collect_detections_pumping(stack, global_thresh, min_area, max_area, illumination=0,
                               raw_stack=None):
    """Like collect_detections but also stores per-ROI pixel arrays for pharynx reconstruction.

    Parameters
    ----------
    stack : list of np.ndarray
        Background-subtracted frames used for thresholding and detection.
    global_thresh : float
        Fixed threshold applied to every frame.
    min_area, max_area : int
        ROI area filter bounds (pixels).
    illumination : int
        0 = light objects on dark background, 1 = dark objects on light background.
    raw_stack : list of np.ndarray or None
        Optional raw (pre-background-subtraction) frames.  When provided,
        intensity metrics (mean_intensity, max_intensity, etc.) are measured
        from these frames using the mask derived from ``stack``.  This avoids
        background-subtraction normalization artifacts in the pumping signal.
        When None, intensities are measured from ``stack`` (previous behaviour).

    Returns
    -------
    detections : pd.DataFrame
        One row per detection; includes _det_id column for later pixel lookup.
    pixel_store : dict
        {det_id: {'intensities': np.ndarray, 'mask': np.ndarray, 'bbox': tuple}}
    """
    records = []
    pixel_store = {}
    det_id = 0
    for frame_no, frame in enumerate(stack):
        print(f"\rThresholding and getting centroids for frame: {frame_no}", end="", flush=True)
        time.sleep(0.0001)
        _, props, _ = preprocess_frame(frame,
                                       min_area=min_area,
                                       max_area=max_area,
                                       thresh=global_thresh,
                                       illumination=illumination)
        raw_frame = raw_stack[frame_no] if raw_stack is not None else frame
        for prop in props:
            y, x = prop.centroid
            pixel_vals = raw_frame[prop.slice][prop.image]
            q75 = np.percentile(pixel_vals, 75)
            records.append({
                "frame":                frame_no,
                "x":                    x,
                "y":                    y,
                "area":                 prop.area,
                "area_convex":          prop.area_convex,
                "mean_intensity":       float(np.mean(pixel_vals)),
                "max_intensity":        float(np.max(pixel_vals)),
                "mean_top_quartile":    float(np.mean(pixel_vals[pixel_vals >= q75])),
                "integrated_intensity": float(np.sum(pixel_vals)),
                "_det_id":              det_id,
            })
            pixel_store[det_id] = {
                "intensities": pixel_vals.copy(),
                "mask":        prop.image.copy(),  # bool array within bbox
                "bbox":        prop.bbox,           # (min_row, min_col, max_row, max_col)
            }
            det_id += 1
    print()
    return pd.DataFrame(records), pixel_store


def build_pixel_store_for_tracks(tracks, raw_pixel_store):
    """Remap pixel_store keys from det_id to (particle, frame) after track linking.

    Drops _det_id from tracks in-place and returns the remapped dict.
    """
    final_store = {}
    for _, row in tracks.iterrows():
        did = int(row["_det_id"])
        if did in raw_pixel_store:
            final_store[(int(row["particle"]), int(row["frame"]))] = raw_pixel_store[did]
    tracks.drop(columns=["_det_id"], inplace=True, errors="ignore")
    return final_store


def analyze_pumping(tracks, frame_rate, min_track_frames=None, peak_prominence=10):
    """Detect pumping peaks per particle using scipy and pyampd (if installed).

    Parameters
    ----------
    tracks : pd.DataFrame
        Must contain 'particle', 'frame', 'mean_intensity'.
    frame_rate : int
        Frames per second.
    min_track_frames : int or None
        Minimum track length to analyse. Default: 1 second (= frame_rate frames).
    peak_prominence : float
        Prominence threshold for scipy.signal.find_peaks.

    Returns
    -------
    pump_events : pd.DataFrame
        Columns: particle, frame, time_s, method ('scipy' | 'ampd')
    pump_summary : pd.DataFrame
        Columns: particle, track_duration_s, n_pumps_scipy, mean_rate_hz_scipy,
                 n_pumps_ampd, mean_rate_hz_ampd  (ampd cols None if unavailable)
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d

    try:
        from pyampd.ampd import find_peaks as ampd_find_peaks
        has_ampd = True
    except ImportError:
        has_ampd = False
        print("pyampd not installed — using scipy peak detection only")

    if min_track_frames is None:
        min_track_frames = frame_rate  # 1 second default

    pump_events  = []
    pump_summary = []

    for particle, group in tracks.groupby("particle"):
        group = group.sort_values("frame")
        if len(group) < min_track_frames:
            continue

        frames_arr = group["frame"].values
        intensity  = group["mean_intensity"].values
        duration_s = len(frames_arr) / frame_rate

        # Light smoothing for AMPD only: suppresses single-frame mask-shape
        # noise that can create spurious local maxima when pyampd detrends.
        # scipy uses the raw signal — prominence already rejects noise spikes
        # and smoothing at fast pump rates (>3 Hz at 20fps) merges real peaks.
        # Window=2 is the minimum that kills single-frame spikes; window=3
        # collapses peaks at ~5 Hz (every 4 frames at 20fps) for weak signals.
        smoothed = uniform_filter1d(intensity.astype(float), size=2)

        # ── scipy ──────────────────────────────────────────────────────────────
        scipy_idx, _ = find_peaks(intensity.astype(float), prominence=peak_prominence)
        for idx in scipy_idx:
            pump_events.append({
                "particle": particle,
                "frame":    int(frames_arr[idx]),
                "time_s":   round(float(frames_arr[idx]) / frame_rate, 4),
                "method":   "scipy",
            })

        # ── pyampd ─────────────────────────────────────────────────────────────
        ampd_idx = []
        if has_ampd and np.ptp(smoothed) > 0:
            try:
                raw_idx = ampd_find_peaks(smoothed)
                # pyampd detrends internally; after detrending, original minima
                # can appear as local maxima.  Filter to indices that are true
                # local maxima in the smoothed signal.
                n = len(smoothed)
                ampd_idx = [
                    idx for idx in raw_idx
                    if (idx == 0 or smoothed[idx] > smoothed[idx - 1])
                    and (idx == n - 1 or smoothed[idx] > smoothed[idx + 1])
                ]
                for idx in ampd_idx:
                    pump_events.append({
                        "particle": particle,
                        "frame":    int(frames_arr[idx]),
                        "time_s":   round(float(frames_arr[idx]) / frame_rate, 4),
                        "method":   "ampd",
                    })
            except Exception as e:
                print(f"  pyampd failed for particle {particle}: {e}")

        n_scipy = len(scipy_idx)
        n_ampd  = len(ampd_idx) if has_ampd else None

        # Quality flags for censoring
        few_pumps = n_scipy < 5 or (n_ampd is not None and n_ampd < 5)
        if n_ampd is not None and n_ampd > 0 and n_scipy > 0:
            method_disagree = abs(n_scipy - n_ampd) / max(n_scipy, n_ampd) > 0.20
        else:
            method_disagree = n_ampd is not None  # one is zero, other isn't
        flag_censor = few_pumps or method_disagree

        pump_summary.append({
            "particle":           particle,
            "track_duration_s":   round(duration_s, 2),
            "n_pumps_scipy":      n_scipy,
            "mean_rate_hz_scipy": round(n_scipy / duration_s, 3) if duration_s > 0 else None,
            "n_pumps_ampd":       n_ampd,
            "mean_rate_hz_ampd":  round(n_ampd / duration_s, 3) if n_ampd is not None and duration_s > 0 else None,
            "flag_censor":        flag_censor,
            "flag_few_pumps":     few_pumps,
            "flag_method_disagree": method_disagree,
        })

    return pd.DataFrame(pump_events), pd.DataFrame(pump_summary)


# Hard-coded quiescent IPI threshold used when no HMM model is available.
# 1.0 s = 20 frames at 20 fps — an unambiguous missed-pump gap.
DEFAULT_QUIESCENT_IPI_THRESH = 1.0   # seconds


def classify_pumping_states(pump_events, pump_summary, model_path=None):
    """Classify per-particle pumping states.

    Primary path: hard threshold at DEFAULT_QUIESCENT_IPI_THRESH (1.0 s).
    Enhancement:  if a valid pumping_hmm.pkl bundle is found, the HMM's
    Viterbi decoder is used for smoothed state assignments and the bundle's
    quiescent threshold overrides the default.  HMM-specific columns
    (frac_slow, frac_fast, rate_slow_hz, rate_fast_hz) are populated only
    when a model is loaded; they are NaN otherwise.

    Parameters
    ----------
    pump_events : pd.DataFrame
        Output of analyze_pumping — must contain 'particle', 'time_s', 'method'.
    pump_summary : pd.DataFrame
        Output of analyze_pumping — results are appended as new columns.
    model_path : str or None
        Path to pumping_hmm.pkl.  If None, looks in models/pumping_hmm.pkl
        relative to the package root.  Pass an empty string '' to force
        threshold-only mode even when a model file exists.

    Returns
    -------
    pump_summary : pd.DataFrame
        Original rows with the following columns added:
          frac_quiescent      — fraction of track time in scipy gaps > threshold
          frac_slow           — HMM slow-state fraction (NaN without model)
          frac_fast           — HMM fast-state fraction (NaN without model)
          rate_slow_hz        — mean rate in slow-state intervals (NaN without model)
          rate_fast_hz        — mean rate in fast-state intervals (NaN without model)
          rate_active_ampd_hz — AMPD rate during non-quiescent windows
          flag_hmm_censored   — True if particle had too few scipy events
    ipi_states_df : pd.DataFrame
        Per-IPI state assignments (all particles with >= min_events):
          particle, t_start_s, t_end_s, ipi_s, state (quiescent / active or slow / fast)
    """
    import pickle
    import os

    # ── Attempt to load HMM bundle ────────────────────────────────────────────
    model      = None
    state_names = None

    if model_path != "":          # empty string = force threshold mode
        if model_path is None:
            pkg_root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.normpath(
                os.path.join(pkg_root, "..", "models", "pumping_hmm.pkl"))

        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    bundle = pickle.load(f)
                model       = bundle["model"]
                state_names = bundle["state_names"]
                q_thresh    = bundle["quiescent_ipi_thresh"]
                min_events  = bundle["min_scipy_events"]
                train_p10   = bundle["training_ipi_p10"]
                train_p90   = bundle["training_ipi_p90"]

                # Validate: check batch IPI distribution is within training range
                all_sc_ipis = []
                for p in pump_events["particle"].unique():
                    sc = pump_events[(pump_events["particle"] == p) &
                                     (pump_events["method"] == "scipy")
                                     ].sort_values("time_s")
                    if len(sc) >= 2:
                        all_sc_ipis.extend(np.diff(sc["time_s"].values).tolist())

                if all_sc_ipis:
                    batch_ipi         = np.array(all_sc_ipis)
                    batch_median      = float(np.median(batch_ipi))
                    frac_above_thresh = float((batch_ipi > q_thresh).mean())

                    out_of_range = (batch_median < train_p10 or
                                    batch_median > train_p90)
                    bad_frac     = (frac_above_thresh < 0.01 or
                                    frac_above_thresh > 0.80)

                    if out_of_range or bad_frac:
                        print(f"  [HMM] WARNING: batch median IPI ({batch_median:.3f}s) "
                              f"or quiescent fraction ({frac_above_thresh*100:.1f}%) "
                              f"outside training range — using HMM states but "
                              f"interpret frac_quiescent with caution.")
                    else:
                        print(f"  [HMM] Boundary check passed  "
                              f"(median IPI={batch_median:.3f}s, "
                              f"{frac_above_thresh*100:.1f}% above threshold)")

            except Exception as exc:
                print(f"  [HMM] Could not load model ({exc}) — "
                      f"falling back to {DEFAULT_QUIESCENT_IPI_THRESH} s threshold")
                model = None
        else:
            print(f"  [HMM] No model found at {model_path} — "
                  f"using hard threshold ({DEFAULT_QUIESCENT_IPI_THRESH} s)")

    if model is None:
        q_thresh   = DEFAULT_QUIESCENT_IPI_THRESH
        min_events = 4          # need at least 3 IPIs for meaningful stats
        print(f"  [threshold] Classifying quiescence with IPI > {q_thresh} s")

    # ── Per-particle classification ───────────────────────────────────────────
    hmm_rows       = []
    ipi_state_rows = []
    for _, summ_row in pump_summary.iterrows():
        p = summ_row["particle"]

        sc = pump_events[(pump_events["particle"]==p) &
                         (pump_events["method"]=="scipy")].sort_values("time_s")
        am = pump_events[(pump_events["particle"]==p) &
                         (pump_events["method"]=="ampd")].sort_values("time_s")

        base_row = {"particle": p, "flag_hmm_censored": False}

        if len(sc) < min_events:
            base_row["flag_hmm_censored"] = True
            for col in ("frac_quiescent","frac_slow","frac_fast",
                        "rate_slow_hz","rate_fast_hz","rate_active_ampd_hz"):
                base_row[col] = np.nan
            hmm_rows.append(base_row)
            continue

        t_sc   = sc["time_s"].values
        ipi_sc = np.diff(t_sc)
        t_am   = am["time_s"].values

        total_t = ipi_sc.sum()

        # ── quiescent fraction (scipy gaps) ───────────────────────────────────
        q_mask        = ipi_sc > q_thresh
        quiescent_t   = ipi_sc[q_mask].sum()
        frac_quiescent= quiescent_t / total_t if total_t > 0 else np.nan

        # Build quiescent interval list for AMPD filtering
        q_intervals = [(t_sc[i], t_sc[i+1])
                       for i, is_q in enumerate(q_mask) if is_q]

        def _in_quiescent(t):
            return any(s <= t <= e for s, e in q_intervals)

        # ── State assignment: HMM decode or hard threshold ────────────────────
        def _r(v):
            return round(float(v), 4) if v is not None and not np.isnan(v) else np.nan

        if model is not None:
            # HMM Viterbi decode on log-IPI sequence
            log_ipi = np.log(ipi_sc).reshape(-1, 1)
            states  = model.predict(log_ipi)
            state_labels = [state_names[s] for s in states]

            frac = {}
            rates = {}
            for s_idx, s_name in enumerate(state_names):
                mask = states == s_idx
                frac[s_name]  = ipi_sc[mask].sum() / total_t if total_t > 0 else 0.0
                rates[s_name] = (1.0 / ipi_sc[mask].mean()) if mask.sum() > 0 else np.nan
        else:
            # Hard threshold: quiescent if IPI > q_thresh, else active
            state_labels = ["quiescent" if ipi > q_thresh else "active"
                            for ipi in ipi_sc]
            frac  = {}
            rates = {}

        # ── AMPD active rate (exclude quiescent windows) ──────────────────────
        am_active = np.array([t for t in t_am if not _in_quiescent(t)])
        if len(am_active) > 1:
            ipi_am_active = np.diff(am_active)
            ipi_am_active = ipi_am_active[ipi_am_active < q_thresh]
            rate_active_ampd = (1.0 / ipi_am_active.mean()
                                if len(ipi_am_active) > 0 else np.nan)
        else:
            rate_active_ampd = np.nan

        base_row.update({
            "frac_quiescent":      round(frac_quiescent, 4),
            "frac_slow":           round(frac["slow"], 4) if "slow" in frac else np.nan,
            "frac_fast":           round(frac["fast"], 4) if "fast" in frac else np.nan,
            "rate_slow_hz":        _r(rates.get("slow")),
            "rate_fast_hz":        _r(rates.get("fast")),
            "rate_active_ampd_hz": _r(rate_active_ampd),
        })
        hmm_rows.append(base_row)

        # Per-IPI state records
        for label, t0, t1, dur in zip(state_labels, t_sc[:-1], t_sc[1:], ipi_sc):
            ipi_state_rows.append({
                "particle":  int(p),
                "t_start_s": round(float(t0),  6),
                "t_end_s":   round(float(t1),  6),
                "ipi_s":     round(float(dur), 6),
                "state":     label,
            })

    hmm_df = pd.DataFrame(hmm_rows)
    pump_summary = pump_summary.merge(hmm_df, on="particle", how="left")

    mode_label   = "HMM" if model is not None else "threshold"
    n_classified = (~hmm_df["flag_hmm_censored"]).sum()
    n_censored   = hmm_df["flag_hmm_censored"].sum()
    print(f"  [{mode_label}] Classified {n_classified} particles  "
          f"({n_censored} censored — fewer than {min_events} scipy events)")

    ipi_states_df = pd.DataFrame(ipi_state_rows)
    return pump_summary, ipi_states_df


def find_best_pumping_particle(tracks, pump_summary):
    """Pick the longest track with the most detected pumps (scipy)."""
    if pump_summary.empty:
        print("No particles met the minimum track length for pumping analysis.")
        return None
    scored = pump_summary.copy()
    scored["score"] = scored["track_duration_s"] * scored["n_pumps_scipy"].fillna(0)
    best_row = scored.loc[scored["score"].idxmax()]
    best = int(best_row["particle"])
    print(f"\nBest pumping particle: {best}")
    print(f"  Track duration:  {best_row['track_duration_s']:.1f} s")
    print(f"  Pumps (scipy):   {best_row['n_pumps_scipy']}")
    if best_row["n_pumps_ampd"] is not None:
        print(f"  Pumps (ampd):    {best_row['n_pumps_ampd']}")
    return best


def _pump_rate_color(rate_hz):
    """Trail colour keyed to rolling pump rate (BGR)."""
    if rate_hz > 4:
        return (255, 0, 255)    # magenta
    elif rate_hz >= 2:
        return (0, 220, 255)    # yellow
    else:
        return (150, 150, 150)  # grey


def _draw_intensity_panel(canvas, x_off, y_off, panel_w, panel_h,
                          intensity_vals, is_pump_vals):
    """Render a rolling intensity trace with pump-event markers into canvas."""
    # Dark background
    cv2.rectangle(canvas, (x_off, y_off),
                  (x_off + panel_w - 1, y_off + panel_h - 1), (35, 35, 35), -1)

    n = len(intensity_vals)
    if n < 2:
        return

    mg = 12  # margin px
    pw, ph = panel_w - 2 * mg, panel_h - 2 * mg
    y_min = min(intensity_vals) * 0.92
    y_max = max(intensity_vals) * 1.08
    if y_max <= y_min:
        y_max = y_min + 1

    def to_px(i, val):
        cx = x_off + mg + int(i * pw / max(n - 1, 1))
        cy = y_off + mg + int(ph - (val - y_min) / (y_max - y_min) * ph)
        cy = max(y_off + mg, min(y_off + mg + ph, cy))
        return cx, cy

    # Intensity trace
    for i in range(1, n):
        cv2.line(canvas, to_px(i - 1, intensity_vals[i - 1]),
                 to_px(i,     intensity_vals[i]),
                 (210, 210, 210), 1)

    # Pump event markers: red dot on the peak
    for i, is_pump in enumerate(is_pump_vals):
        if is_pump:
            cx, cy = to_px(i, intensity_vals[i])
            cv2.circle(canvas, (cx, cy), 3, (0, 0, 220), -1)

    # Current-position cursor (white vertical)
    cx, _ = to_px(n - 1, intensity_vals[-1])
    cv2.line(canvas, (cx, y_off + mg), (cx, y_off + mg + ph), (255, 255, 255), 1)

    # Axis labels
    cv2.putText(canvas, f"{y_max:.0f}",
                (x_off + 2, y_off + mg + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 160), 1)
    cv2.putText(canvas, f"{y_min:.0f}",
                (x_off + 2, y_off + mg + ph),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 160), 1)
    cv2.putText(canvas, "Mean intensity",
                (x_off + mg, y_off + panel_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 160), 1)


def create_pumping_video(video_path, tracks, pump_events, particle_id,
                         output_folder, frame_rate=20,
                         global_thresh=None, min_area=50, max_area=300,
                         illumination=0, crop_size=None, pump_method="scipy"):
    """Create annotated pumping video for a single particle.

    Right panel layout (top → bottom):
      - Contrast-normalised crop of the worm (global normalization, stable across frames)
      - Particle info text (ID, time, rate, count)
      - Rolling intensity plot with pump-event markers

    The right panel is half the video width. The crop is extracted at crop_size pixels
    but displayed scaled up to fill the panel width.

    Trail on the main frame is coloured by rolling pump rate:
      > 4 Hz  → magenta  |  2–4 Hz → yellow  |  < 2 Hz → grey
    No ROI boxes, masks, or crosshairs are drawn on the worm.
    Crop and trail persist on dropped frames using last known position.
    """
    particle_df    = tracks[tracks["particle"] == particle_id].copy().sort_values("frame")
    particle_pumps = pump_events[
        (pump_events["particle"] == particle_id) &
        (pump_events["method"]   == pump_method)
    ]
    pump_frames = set(particle_pumps["frame"].tolist())

    if len(particle_df) == 0:
        print(f"No data for particle {particle_id}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or frame_rate
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crop extraction size: slightly wider than the median object diameter
    if crop_size is None:
        median_area = float(particle_df["area"].median())
        obj_radius  = np.sqrt(median_area / np.pi)  # equivalent-circle radius
        crop_size   = int(obj_radius * 2.5)          # ~25% padding each side
        crop_size   = max(crop_size, 20)
    print(f"  Crop extraction size: {crop_size}px  (median area={particle_df['area'].median():.0f})")

    # Right panel is half the video width; crop display fills it (minus border)
    panel_w           = width // 2
    border            = 3
    display_crop_size = panel_w - 2 * border   # px used for the scaled crop display
    info_h    = 110                             # info text area height
    plot_h    = height - (display_crop_size + 2 * border) - info_h
    plot_h    = max(plot_h, 80)

    output_width  = width + panel_w
    output_height = height

    # ── Pre-pass: compute stable global crop normalization bounds ─────────────
    # Sample up to 30 evenly-spaced particle frames to estimate the pixel range
    # in the crop region. This avoids per-frame auto-contrast flicker.
    sample_idx   = particle_df.index[
        np.linspace(0, len(particle_df) - 1, min(30, len(particle_df)), dtype=int)
    ]
    sample_df    = particle_df.loc[sample_idx]
    crop_pixels  = []
    half         = crop_size // 2
    cap2         = cv2.VideoCapture(video_path)
    for _, srow in sample_df.iterrows():
        sf = int(srow["frame"])
        cap2.set(cv2.CAP_PROP_POS_FRAMES, sf)
        ret2, frm2 = cap2.read()
        if not ret2:
            continue
        gf2 = cv2.cvtColor(frm2, cv2.COLOR_BGR2GRAY) if frm2.ndim == 3 else frm2
        cx, cy = int(srow["x"]), int(srow["y"])
        xs2, xe2 = max(0, cx - half), min(width,  cx + half)
        ys2, ye2 = max(0, cy - half), min(height, cy + half)
        crop2 = gf2[ys2:ye2, xs2:xe2]
        if crop2.size > 0:
            crop_pixels.append(crop2.flatten())
    cap2.release()

    if crop_pixels:
        all_pix = np.concatenate(crop_pixels)
        # Use global_thresh as lower bound to clip background pixels out entirely.
        # Fall back to 30th percentile if thresh is unavailable.
        if global_thresh is not None:
            crop_lo = float(global_thresh)
        else:
            crop_lo = float(np.percentile(all_pix, 30))
        crop_hi = float(np.percentile(all_pix, 99))
    else:
        crop_lo = float(global_thresh) if global_thresh is not None else 30.0
        crop_hi = 255.0
    crop_range = max(crop_hi - crop_lo, 1.0)
    print(f"  Crop normalization: lo={crop_lo:.1f} (thresh)  hi={crop_hi:.1f}")

    output_path = os.path.join(output_folder, f"particle_{particle_id}_pumping.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    start_frame      = int(particle_df["frame"].min())
    end_frame        = int(particle_df["frame"].max())
    rolling_window_s = 5
    trail_frames     = 60
    plot_window_s    = 8   # seconds of history shown in intensity plot

    # Smooth x/y positions for stable crop centering (rolling mean, ~0.5 s window)
    smooth_win = max(3, int(frame_rate * 0.5))
    particle_df = particle_df.copy()
    particle_df["x_smooth"] = (particle_df["x"]
                                .rolling(smooth_win, center=True, min_periods=1)
                                .mean())
    particle_df["y_smooth"] = (particle_df["y"]
                                .rolling(smooth_win, center=True, min_periods=1)
                                .mean())
    # Fast lookup: frame → smoothed position
    smooth_pos = {int(r["frame"]): (int(round(r["x_smooth"])), int(round(r["y_smooth"])))
                  for _, r in particle_df.iterrows()}

    # State preserved across dropped frames
    last_x, last_y       = None, None
    last_crop_bordered   = None   # pre-rendered crop+border image

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        canvas[:height, :width] = frame

        # ── Rolling rate ──────────────────────────────────────────────────────
        window_frames = int(rolling_window_s * frame_rate)
        recent_pumps  = particle_pumps[
            (particle_pumps["frame"] >= frame_idx - window_frames) &
            (particle_pumps["frame"] <= frame_idx)
        ]
        rolling_rate = len(recent_pumps) / rolling_window_s
        cum_pumps    = len(particle_pumps[particle_pumps["frame"] <= frame_idx])
        trail_color  = _pump_rate_color(rolling_rate)

        # ── Update position and crop when data exists this frame ──────────────
        frame_data = particle_df[particle_df["frame"] == frame_idx]
        if len(frame_data) > 0:
            row = frame_data.iloc[0]
            last_x, last_y = smooth_pos.get(frame_idx, (int(row["x"]), int(row["y"])))

            xs = max(0, last_x - half);  xe = min(width,  last_x + half)
            ys = max(0, last_y - half);  ye = min(height, last_y + half)
            crop_gray = gray_frame[ys:ye, xs:xe]
            if crop_gray.size > 0:
                crop_clipped = np.clip(crop_gray.astype(np.float32), crop_lo, crop_hi)
                crop_norm    = ((crop_clipped - crop_lo) / crop_range * 255).astype(np.uint8)
                crop_bgr     = cv2.cvtColor(crop_norm, cv2.COLOR_GRAY2BGR)
                crop_r       = cv2.resize(crop_bgr, (display_crop_size, display_crop_size))
                last_crop_bordered = cv2.copyMakeBorder(
                    crop_r, border, border, border, border,
                    cv2.BORDER_CONSTANT, value=trail_color)

        # ── Draw trail + crop using last known position (persists on drops) ───
        if last_x is not None:
            # Trail
            past = particle_df[
                (particle_df["frame"] <= frame_idx) &
                (particle_df["frame"] >  frame_idx - trail_frames)
            ]
            if len(past) > 1:
                pts = np.array([[int(r["x"]), int(r["y"])]
                                for _, r in past.iterrows()], dtype=np.int32)
                overlay = canvas[:height, :width].copy()
                cv2.polylines(overlay, [pts], False, trail_color, 2)
                canvas[:height, :width] = cv2.addWeighted(
                    overlay, 0.6, canvas[:height, :width], 0.4, 0)

            # Crop inset
            if last_crop_bordered is not None:
                ch, cw = last_crop_bordered.shape[:2]
                canvas[0:ch, width:width + cw] = last_crop_bordered

            # ── Info text ─────────────────────────────────────────────────────
            ix  = width + 5
            iy0 = display_crop_size + 2 * border + 18
            cv2.putText(canvas, f"Particle {particle_id}",
                        (ix, iy0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            cv2.putText(canvas, f"t = {frame_idx / frame_rate:.1f} s",
                        (ix, iy0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            cv2.putText(canvas, f"Rate: {rolling_rate:.2f} Hz",
                        (ix, iy0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, trail_color, 1)
            cv2.putText(canvas, f"Pumps: {cum_pumps}",
                        (ix, iy0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            cv2.putText(canvas, f"({pump_method})",
                        (ix, iy0 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1)

            # ── Intensity plot ────────────────────────────────────────────────
            plot_win_frames = int(plot_window_s * frame_rate)
            plot_start = frame_idx - plot_win_frames
            plot_data  = particle_df[
                (particle_df["frame"] >= plot_start) &
                (particle_df["frame"] <= frame_idx)
            ].sort_values("frame")

            if len(plot_data) >= 2:
                i_vals  = plot_data["mean_intensity"].tolist()
                p_vals  = [f in pump_frames for f in plot_data["frame"]]
                plot_y0 = display_crop_size + 2 * border + info_h
                _draw_intensity_panel(
                    canvas, width, plot_y0, panel_w, plot_h,
                    i_vals, p_vals
                )

        out.write(canvas)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"\rProcessing frame {frame_idx}/{end_frame}", end="", flush=True)

    cap.release()
    out.release()
    print(f"\nPumping video saved to: {output_path}")
    return output_path


# # Usage:
# best_particle = find_particle_with_all_behaviors(df_motion)

# if best_particle is not None:
#     video_folder = os.path.dirname(video_path)
    
#     output_video = create_annotated_video(
#         video_path=video_path,
#         df=df_motion,
#         particle_id=best_particle,
#         output_folder=video_folder,
#         pixel_length=60,
#         frame_rate=10,
#         global_thresh=None,  # Will auto-detect using Otsu
#         min_area=100,
#         max_area=5000,
#         illumination=0,      # 0 for worms lighter than background
#         crop_size=150,
#         show_mask=True
#     )
