"""
Compare global-Otsu vs per-frame Otsu thresholding on a preprocessed video.

Applies the same normalization + backsub pipeline that FastTrack uses, then
runs the detection pass two ways and plots:
  1. Per-frame Otsu threshold over time (how stable is it after preprocessing?)
  2. Per-frame detection count: global thresh vs per-frame thresh
  3. Distribution of per-frame thresholds

Usage:
    python dev/compare_thresh_strategies.py [path/to/video.avi]
"""

import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import uniform_filter1d
from skimage import filters, measure, morphology

VIDEO = (
    "/Volumes/User Homes/ODlab-user/UserFolders/Nawaphat"
    "/6 CEST-2.1/Locomotion_Off food/0_COMPLETE!!/N2"
    "/20260121 N2 off food-01212026142725-0000.avi"
)
CONFIG = "configs/IR_medium.yaml"
OUT    = "dev/compare_thresh_strategies.png"


# ── Preprocessing (mirrors FastTrack.py steps 3-4) ────────────────────────

def load_and_preprocess(video_path, config):
    min_area   = config["min_area"]
    max_area   = config["max_area"]
    frame_rate = config["frame_rate"]
    backsub_n  = config["backsub_frames"]

    cap = cv2.VideoCapture(video_path)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    cap.release()
    print(f"Loaded {len(frames)} frames")

    # ── Illumination normalization (streaming, uint8) ──
    medians  = np.array([float(np.median(f)) for f in frames])
    raw_p99s = np.array([float(np.percentile(f, 99)) for f in frames])

    ref_fast   = float(np.median(medians))
    scale_fast = np.ones(len(frames))
    if ref_fast > 1:
        for i in range(len(frames)):
            if medians[i] > 1:
                s = ref_fast / medians[i]
                if abs(s - 1.0) > 0.01:
                    scale_fast[i] = s

    p99s = raw_p99s * scale_fast
    if np.std(p99s) < 2.0:
        p99s = np.array([float(np.percentile(f, 90)) for f in frames]) * scale_fast
    win  = max(3, int(frame_rate * 10))
    p_slow   = uniform_filter1d(p99s, size=win, mode="reflect")
    scale_slow = float(np.mean(p_slow)) / p_slow
    combined = scale_fast * scale_slow

    for i in range(len(frames)):
        s = float(combined[i])
        if abs(s - 1.0) > 0.005:
            f32 = frames[i].astype(np.float32)
            f32 *= s
            np.clip(f32, 0, 255, out=f32)
            frames[i] = f32.astype(np.uint8)

    # ── Background subtraction ──
    frame_list = np.linspace(1, n - 1, backsub_n, dtype=int)
    bg = np.zeros(frames[0].shape, dtype=np.float32)
    for i in frame_list:
        bg += frames[i].astype(np.float32)
    bg /= backsub_n

    for i in range(len(frames)):
        f32 = frames[i].astype(np.float32)
        f32 -= bg
        np.clip(f32, 0, 255, out=f32)
        frames[i] = f32.astype(np.uint8)

    print(f"Preprocessing done. frame[0]: min={frames[0].min()}  max={frames[0].max()}  mean={frames[0].mean():.1f}")
    return frames, min_area, max_area


def otsu_thresh(frame):
    return float(filters.threshold_otsu(frame))


def detect_count(frame, thresh, min_area, max_area, illumination=0):
    bw = frame > thresh if illumination == 0 else frame < thresh
    bw = morphology.remove_small_objects(bw, min_size=min_area)
    labeled = measure.label(bw)
    props = measure.regionprops(labeled)
    return sum(1 for p in props if min_area <= p.area_convex <= max_area)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    with open(CONFIG) as f:
        config = yaml.safe_load(f)

    illumination = config["illumination"]
    min_area     = config["min_area"]
    max_area     = config["max_area"]

    frames, min_area, max_area = load_and_preprocess(video_path, config)

    # Global threshold from first frame
    global_thresh = otsu_thresh(frames[0])
    print(f"Global Otsu threshold (from frame 0): {global_thresh:.1f}")

    # Per-frame thresholds and detection counts
    per_thresh   = []
    counts_global = []
    counts_perframe = []

    print("Running detection pass...")
    for i, frame in enumerate(frames):
        print(f"\r  frame {i}/{len(frames)}", end="", flush=True)
        t = otsu_thresh(frame)
        per_thresh.append(t)
        counts_global.append(detect_count(frame, global_thresh, min_area, max_area, illumination))
        counts_perframe.append(detect_count(frame, t, min_area, max_area, illumination))
    print()

    per_thresh    = np.array(per_thresh)
    counts_global = np.array(counts_global)
    counts_perframe = np.array(counts_perframe)

    print(f"\nPer-frame Otsu:  mean={per_thresh.mean():.1f}  std={per_thresh.std():.2f}  "
          f"min={per_thresh.min():.1f}  max={per_thresh.max():.1f}")
    print(f"Detection count (global):    mean={counts_global.mean():.1f}  std={counts_global.std():.2f}")
    print(f"Detection count (per-frame): mean={counts_perframe.mean():.1f}  std={counts_perframe.std():.2f}")
    print(f"Frames where counts differ:  {np.sum(counts_global != counts_perframe)}/{len(frames)}")

    # ── Plot ──
    frames_x = np.arange(len(frames))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle(
        f"Global vs per-frame Otsu — global thresh={global_thresh:.0f}",
        fontsize=12
    )

    # Panel 1: per-frame threshold over time
    ax = axes[0]
    ax.plot(frames_x, per_thresh, linewidth=0.7, color="steelblue", label="per-frame Otsu")
    ax.axhline(global_thresh, color="red", linewidth=1.2, linestyle="--", label=f"global={global_thresh:.0f}")
    ax.set_ylabel("Threshold value")
    ax.set_xlabel("Frame")
    ax.legend(fontsize=9)
    ax.set_title("Per-frame Otsu threshold (stability after normalization + backsub)")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Panel 2: detection counts over time
    ax = axes[1]
    ax.plot(frames_x, counts_global,   linewidth=0.6, color="red",       alpha=0.8, label=f"global (std={counts_global.std():.2f})")
    ax.plot(frames_x, counts_perframe, linewidth=0.6, color="steelblue", alpha=0.8, label=f"per-frame (std={counts_perframe.std():.2f})")
    ax.set_ylabel("Detections / frame")
    ax.set_xlabel("Frame")
    ax.legend(fontsize=9)
    ax.set_title("Detection count per frame")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Panel 3: distribution of per-frame thresholds
    ax = axes[2]
    ax.hist(per_thresh, bins=40, color="steelblue", alpha=0.7, density=True)
    ax.axvline(global_thresh, color="red", linewidth=1.5, linestyle="--", label=f"global={global_thresh:.0f}")
    ax.axvline(per_thresh.mean(), color="black", linewidth=1.0, linestyle=":", label=f"per-frame mean={per_thresh.mean():.0f}")
    ax.set_xlabel("Otsu threshold value")
    ax.set_ylabel("Density")
    ax.set_title(f"Per-frame threshold distribution  (std={per_thresh.std():.2f})")
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
