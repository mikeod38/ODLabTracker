# ODLabTracker

A Python tool for tracking *C. elegans* behavior from video. Supports two analysis modes:

- **Postural** — tracks body posture and classifies movement states (forward runs, reversals, pirouettes)
- **Pumping** — tracks pharyngeal pumping rate from GCaMP or brightfield video using intensity peak detection

---

## Installation

### macOS / Linux

1. Clone the repository:

   ```bash
   git clone https://github.com/ODonnellLab/ODLabTracker.git
   cd ODLabTracker
   ```

2. Install in editable mode:

   ```bash
   pip install -e .
   ```

Requires Python 3.9+.

---

### Windows

The `av` (PyAV) package requires FFmpeg binaries that are not bundled in the standard pip wheel on Windows. The most reliable approach is to use [Miniconda](https://docs.anaconda.com/miniconda/) and install `av` via conda-forge before running pip.

**Step 1 — Install Miniconda**

Download and run the Miniconda installer for Windows from:
https://docs.anaconda.com/miniconda/

During installation, check "Add Miniconda3 to my PATH" (or use the **Anaconda Prompt** that is added to the Start menu).

**Step 2 — Clone the repository**

Install [Git for Windows](https://git-scm.com/download/win) if you don't have it, then in an Anaconda Prompt or PowerShell:

```
git clone https://github.com/ODonnellLab/ODLabTracker.git
cd ODLabTracker
```

**Step 3 — Create a conda environment**

```
conda create -n odlabtracker python=3.11
conda activate odlabtracker
```

**Step 4 — Install `av` via conda-forge**

```
conda install av -c conda-forge
```

**Step 5 — Install remaining dependencies**

```
pip install -e .
```

**Step 6 — Run the tracker**

Always activate the environment first in any new terminal session:

```
conda activate odlabtracker
python track.py -c configs\your_config.yaml -f path\to\video.avi
```

> **Note:** Use backslashes (`\`) for file paths on Windows, or wrap paths in quotes if they contain spaces.

---

## Documentation

- **[Pumping analysis tutorial](docs/pumping_tutorial.md)** — full walkthrough of pumping mode: config setup, running a batch, output files, and interpreting HMM state classification results
- **[Postural comparison across genotypes](docs/postural_comparison.md)** — batch speed and locomotion analysis: running `batch_postural_comparison.py`, output files, and statistical model

---

## Usage

All analysis runs through the `track.py` entry point. The analysis mode is set by the `mode:` key in the config file.

```bash
python track.py -c configs/your_config.yaml -f path/to/video.avi
```

Omit `-f` to use an interactive file picker.

### Options

| Flag | Description |
|------|-------------|
| `-f` | Path to input video (`.avi`, `.mp4`, `.tif`/`.tiff`) |
| `-c` | Path to YAML config file (required) |
| `-v` | Verbose output |

---

## Config Files

Example configs are in `configs/`. Copy and edit the one closest to your setup.

| Config | Use case |
|--------|----------|
| `IR_medium.yaml` | IR brightfield, medium magnification |
| `PGlow_GCaMP_2.5x_8bin.yaml` | Fluorescence imaging, 2.5× objective |
| `Stereo0.5X_small.yaml` | Stereo scope, 0.5× objective |
| `Stereo1X_small.yaml` | Stereo scope, 1× objective |

### Common parameters (both modes)

```yaml
mode: postural        # postural or pumping
min_area: 100         # minimum object area in pixels
max_area: 5000        # maximum object area in pixels
gap_range: 5          # max frames a worm can disappear and still be re-linked
search_range: 30      # max pixel distance to link detections across frames
min_length: 10        # minimum track length to keep (frames)
frame_rate: 10        # frames per second
illumination: 1       # 0 = bright objects on dark bg (IR), 1 = dark on light
subsample: 1          # keep 1 in every N frames (use >1 for high-FPS video)
backsub: true         # subtract background before thresholding
backsub_frames: 10    # number of frames averaged for background model
pixel_length: 60      # pixels per mm (for speed calculations)
thresh: None          # manual threshold; None = auto (Otsu)
thresh_method: otsu   # auto-threshold method: otsu, triangle, yen, li
min_thresh: null      # floor for auto-threshold; null = auto-detect from background noise (backsub only); set explicitly to override
max_objects: null     # max detections per frame; threshold raised if exceeded
normalize_illumination: false  # two-pass illumination normalization: per-frame median correction + smoothed slow-drift correction; recommended for IR setups with LED flicker
```

### Postural-specific parameters

```yaml
mode: postural
speed_threshold: 0.03        # minimum speed to classify as moving (mm/s)
boundary_margin: null        # censor particles whose median centroid is within this many px of any frame edge;
                             # null = auto (half the median major axis, ~half a worm length)
max_area_cv: null            # exclude tracks with within-track area CV above this value; null = disabled
                             # (try 0.4 to filter fragmented or merging tracks)
save_annotated_video: false  # write a per-worm annotated video alongside tracks.csv; slow (~60 s/video),
                             # disable for batch runs
reversal_persistence: 2      # consecutive qualifying frames required to confirm reversal onset;
                             # filters single-frame direction spikes during pauses
merge_reversal_gap: 5        # merge adjacent reversal events separated by ≤ this many frames;
                             # repairs splits caused by brief unreliable frames mid-reversal
area_reliability_threshold: 0.70  # suppress reversal entry when particle area drops below this fraction
                                  # of its per-particle median (guards against partial thresholding artifacts)
```

### Pumping-specific parameters

```yaml
mode: pumping
min_pump_track_frames: 20   # minimum track length for pumping analysis
peak_prominence: 10         # scipy find_peaks prominence threshold
```

---

## Outputs

Results are saved to a folder named `<video_name>_results/` next to the input video.

### Postural mode

| File | Description |
|------|-------------|
| `tracks.csv` | Per-frame centroid positions and motion statistics |
| `trajectories.png` | Overlay of all tracks on the first frame |

### Pumping mode

| File | Description |
|------|-------------|
| `tracks.csv` | Per-frame centroid positions and intensity measurements |
| `pumping_events.csv` | Frame index and method for each detected pump |
| `pumping_summary.csv` | Per-particle: track duration, pump count, mean rate (Hz) |
| `pumping_ipi_states.csv` | Per-IPI state assignments (quiescent/active) for non-censored particles |
| `pixel_data.pkl` | Per-ROI pixel arrays keyed by `(particle, frame)` |
| `particle_<N>_pumping.mp4` | Annotated video for the best-tracked particle |

The pumping video shows:
- **Main view** — full video with a rate-coloured position trail (magenta >4 Hz, yellow 2–4 Hz, grey <2 Hz)
- **Right panel** — zoomed crop of the pharynx (globally normalised contrast) and a rolling intensity trace with red dots marking detected pumps

---

## Diagnostics

### Inspect first frame detection

```bash
python dev/inspect_first_frame.py -f path/to/video.avi -c configs/your_config.yaml
```

Shows the thresholded first frame with detected objects overlaid. Useful for tuning `min_area`, `max_area`, and `thresh`.

### Inspect pumping signal

```bash
python dev/inspect_pumping_signal.py -f path/to/video.avi -c configs/your_config.yaml --max-frames 200
```

Runs detection and linking on a short clip and plots the per-particle intensity trace with detected peaks. Useful for tuning `peak_prominence` before running the full analysis.

---

## Batch and parallel processing

Two scripts are provided for running `track.py` on a folder of videos. Both search the target directory recursively for `.avi`, `.mp4`, `.tif`, and `.tiff` files.

### Sequential — `run_fasttrack_batch.py`

Processes one file at a time and streams output live to the terminal.

```bash
python run_fasttrack_batch.py <directory> -c configs/your_config.yaml
```

Use this when:
- You want to watch progress in real time
- RAM is limited
- You need to Ctrl-C and resume without losing completed results

### Parallel — `run_fasttrack_parallel.py`

Processes N files simultaneously, each in its own subprocess.

```bash
python run_fasttrack_parallel.py <directory> -c configs/your_config.yaml -j 4
```

Output for each file is written to a log file in `logs/` so jobs don't interleave in the terminal. Monitor any individual job with:

```bash
tail -f logs/video_name.log
```

Completion status prints to the terminal as each job finishes.

After all jobs finish, pumping results are automatically collected into:

| File | Description |
|------|-------------|
| `batch_pumping_summary.csv` | One row per particle; includes bout statistics and derived rates |
| `batch_pumping_plots.png` | Strip plots of active pump rate, quiescence entry rate, and mean bout duration per condition |
| `batch_pumping_report.md` | Markdown summary with plots, per-condition stats, and mixed model results |

Per-particle bout statistics computed from `pumping_events.csv`:

| Column | Description |
|--------|-------------|
| `n_pumps_ampd_active` | AMPD pump count outside quiescent windows |
| `active_duration_s` | Track duration minus total quiescent time |
| `n_quiescent_bouts` | Number of quiescent bouts (maximal runs of IPIs > 0.417 s) |
| `total_quiescent_time_s` | Total seconds spent in quiescent bouts |

Six Poisson GLMMs (via R/lme4) are fit across two balanced comparisons:
- **Food type**: fed animals, OP50 vs JUb39 — active rate, entry rate, exit rate
- **Feeding state**: OP50 animals, fed vs off-food — active rate, entry rate, exit rate

To supply genotype/food/condition labels, provide a metadata CSV:

```bash
python run_fasttrack_parallel.py <directory> -c configs/... -j 4 --metadata metadata.csv
```

The metadata CSV must have columns: `folder_name, genotype, food, condition`.

Use `--no-summary` to skip the collection step.

Use this when:
- You have many files and want to use idle CPU cores
- You are comfortable monitoring log files

**Choosing `-j` (number of workers):** each tracking job loads the full video into RAM during processing. A rough guide:

| RAM | Recommended `-j` |
|-----|-----------------|
| 16 GB | 2–4 |
| 32 GB | 4–8 |
| 64 GB+ | 8+ |

Setting `-j` too high will cause the system to swap memory to disk, which is slower than running sequentially. When in doubt, start with `-j 2` and increase if memory stays comfortable.

---

## Project structure

```
ODLabTracker/
├── track.py                         # Main entry point (dispatches by mode)
├── FastTrack.py                     # Postural tracking and movement classification
├── FastTrackPumping.py              # Pumping analysis
├── configs/                         # Example YAML config files
├── dev/                             # Diagnostic and analysis scripts
│   ├── inspect_first_frame.py       # Visualize threshold and detections on frame 1
│   ├── inspect_pumping_signal.py    # Plot pumping intensity trace with detected peaks
│   ├── batch_postural_comparison.py # Multi-genotype locomotion speed comparison
│   └── LOCOMOTION_ANALYSIS_METHODS.md  # Statistical methods for locomotion analysis
├── docs/
│   ├── pumping_tutorial.md          # Pumping mode walkthrough
│   └── postural_comparison.md       # Postural comparison pipeline guide
├── src/ODLabTracker/
│   ├── tracking.py                  # Core tracking and analysis library
│   └── locomotion.py                # Batch locomotion metric extraction and LME fitting
└── pyproject.toml
```
