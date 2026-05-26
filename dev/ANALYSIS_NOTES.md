# ODLabTracker — Analysis Session Notes

Running log of conclusions, decisions, and open questions from each session.
Most recent session first.

---

## Session: 2026-05-26 — LME fixed-date refactor, package extraction, methods write-up

### LME shrinkage bias fix
- **Problem**: random date effects (`(1|date)`) undergo REML shrinkage that fails to absorb
  day-to-day N2 speed variability with ~3 recordings per genotype. This biased fold-change
  estimates upward for strains recorded on atypically fast N2 days.
- **Fix**: switched to fixed date effects (`date` as a covariate). Fixed effects do not
  shrink and are equivalent to within-date normalization.
- **Consequence**: only recordings from dates with a same-date N2 are included in the LME
  (`keep_n2_dates()` R helper). Genotypes with no same-date N2 get `NaN` stats.
- **Validation**: cat-1, cest-2.1, and cest2.1+tbh-1 all moved below 1.0 fold-change
  after the fix, consistent with their raw normalized values.

### Sort order: LME fold-change (not raw speed)
- Genotypes in the strip plot are now ordered by ascending LME fold-change estimate.
- Genotypes with `NaN` stats (no same-date N2) are sorted by raw speed and placed at bottom.

### Reversal/pirouette rate panels removed from plot
- Rate panels hidden because ~3 min recordings yield ~4 reversals/worm — too few for
  reliable LME estimates. LME fits remain cached in `postural_comparison_stats.csv`.
- See "Open questions / TODO" for when to revisit.

### Figure changes
- N2 recordings added back as grey dots on the N2 row.
- N2 grand mean labeled in µm/s (converted from mm/s) to distinguish from fold-change axis.
- Speed distribution panel added as right panel (shared y-axis with strip plot).
  - Gaussian KDE, bandwidth = 0.30; densities clipped below 3% of peak.
  - Sub-recording violins within each genotype; N2 recordings pooled.
- Legend anchored dynamically: left edge at max observed dot + 1% of axis width.

### Generalization and package extraction
- Removed all hardcoded NAS/dataset-specific paths from `batch_postural_comparison.py`.
- Added `--data-dir`, `--out-dir`, `--title`, `--refit`, `--refresh` CLI arguments.
- Core analysis functions moved to `src/ODLabTracker/locomotion.py` so they are importable
  after `pip install -e .`. The script is now a thin CLI wrapper around these functions.
- Supplemental CSV exported: `supplemental_particle_data.csv` (per-particle metrics for
  independent reanalysis).
- Methods written to `dev/LOCOMOTION_ANALYSIS_METHODS.md` (statistical rationale,
  R reproduction code from supplemental CSV).

### Key functions in `src/ODLabTracker/locomotion.py`
- `load_recording(results_dir, frame_rate, min_speed)` — load single recording
- `scan_dataset_particles(data_dir, frame_rate, exclusions, min_speed)` — batch load
- `scan_dataset_fwd_frames(...)` — load per-forward-run-frame data for LME
- `add_normalization(df, n2_ref, metrics)` — fold-change vs N2 by date
- `fit_lme_stats(frame_df, particle_df, order, metrics, out_dir, n2_ref)` — R LME via subprocess
- `genotype_order(df, stat_df, n2_ref)` — sort by LME fold-change

---

## Session: 2026-05-20 (continued) — Frame-gap bug, speed QC, genotype comparison

### Frame-gap speed inflation bug (tracking.py)
- **Bug**: trackpy linking introduces gaps where `frame.diff()` returns N (not 1).
  Dividing vx/vy by 1 frame time inflated speed N-fold at every gap.
  These inflated values clipped to `max_instantaneous_speed = 0.6`, causing a spike
  at 0.6 mm/s and corrupting the rolling median and all downstream postural states.
- **Fix** (`calculate_speed_parameters()`): compute `_frame_gap = frame.diff().fillna(1).clip(lower=1)`,
  divide vx/vy by `_frame_gap` before clipping. Dropped from dict before returning df.
- Confirmed: with the fix the 0.6 spike disappears; per-particle median speeds are ~0.13–0.26 mm/s.

### Forward-run speed: mean → median
- `fwd_speed` per particle changed from `.mean()` to `.median()` for robustness.
- Speed reported in comparison plot = median across particles (per-recording).

### N2 QC: censored Mar 11 and Mar 12
- **Mar 12 / 20260312**: p99 CV unchanged (0.0435 → 0.0435) — normalization completely
  failed. Speed anomalously low (67 µm/s). Excluded as clear artifact.
- **Mar 11 / 20260311**: normalization succeeded but speed anomalously low (84 µm/s).
  No obvious artifact; conservative exclusion to avoid pulling down grand-mean N2.
  No other genotypes on either date, so cross-genotype impact is zero.
- Both added to `data/nawaphat_postural_results/exclude.csv`.

### Speed weighting QC (dev/qc_speed_weighting.py → plots 35, 36)
- Compared per-particle median vs frame-pooled median across all N2 dates.
- Frame-pooled median is duration-weighted (longer tracks dominate more).
- Difference is small (<10% on most dates) and no systematic direction — weighting
  choice does not explain the Jan vs March speed difference.
- Track length vs speed scatter (plot 36): no consistent length-speed bias across videos.
  A few videos show mild positive correlation but not enough to explain outlier dates.
- **Conclusion**: cannot find an artifactual explanation for the Jan vs March speed difference.
  Likely reflects genuine biological or batch differences (worm age, prep).

### Batch retrack: all 32 genotypes
- 131 videos retracked with updated tracking.py (frame-gap fix + boundary margin).
- 12 NAS workers caused I/O saturation; for future retracks use 4–6 workers.
- Two failures: bas-1/20260307 and cat-1/20260120 — both already in exclude.csv.
- Final dataset: 117 recordings, 32 genotypes, 13709 total particles (after exclusions).

### Genotype comparison changes (batch_postural_comparison.py)
- **Sort order**: mutants now sorted by mean normalized speed (slowest first) instead of reversal rate.
  Unmatched genotypes (no same-date N2) sorted by raw speed and placed at bottom.
- **Event annotation**: reversal and pirouette panels now annotate each genotype row with
  `n(events)/n(worms)` (sum across all recordings for that genotype) in small gray text.
- **Supplementary table**: `postural_comparison.csv` now includes `n_reversals`, `n_pirouettes`,
  and `n_excluded` columns for upload as supplementary data.

---

## Session: 2026-05-20 — Memory fix, profiling, N2 batch retrack

### Memory crash & fix
- Root cause: illumination normalization converted entire frame buffer to float32
  (`frames = [f.astype(np.float32) for f in frames]`), quadrupling RAM per worker.
  With 12 workers this caused an OOM crash.
- Fix: compute normalization scale factors analytically from uint8 frame statistics
  (percentile scales linearly with a scalar: `p99(f × s) = p99(f) × s`), then apply
  the combined fast+slow scale one frame at a time, immediately converting back to uint8.
  Background subtraction likewise processes one frame at a time.
- Result: frame buffer stays at uint8 throughout (~2.5 GB for 1800-frame N2 video);
  peak float32 usage is one frame. 12-worker batch run now succeeds.

### Two-pass normalization — confirmed working
- **Pass 1 (fast)**: per-frame median correction. Scale factor = ref_median / frame_median.
  Removes high-frequency LED flicker (irregular, frame-to-frame).
- **Pass 2 (slow)**: p99 (or p90 fallback) smoothed over 10 s window captures slow drift
  in worm-pixel brightness. Scale factor = ref_slow / p_slow.
- Both scale factors computed from raw uint8 statistics; combined and applied in one pass.
- Verified against normalization_comparison.png reference (Mar 6: p99 CV 0.0465→0.0048,
  Mar 7: 0.0400→0.0070). Streaming refactor produces identical results.
- Most Jan–Feb recordings: fast correction alone is sufficient (no slow drift detected).
  Mar 6 and Mar 7 are the primary videos where the slow correction matters.

### Per-video profiling (Jan 21 N2, 1800 frames)
| Step                        | Time   | RSS    |
|-----------------------------|--------|--------|
| Load frames (uint8)         | 5.4 s  | 2.49 GB |
| Illumination normalization  | 9.6 s  | 2.95 GB |
| Background subtraction      | 2.2 s  | 3.34 GB |
| Detection (regionprops)     | 28.5 s | 2.99 GB |
| Link tracks (trackpy)       | 0.1 s  | 3.00 GB |
| Speed parameters            | 0.1 s  | — |
| Trajectory plot + CSV       | 1.9 s  | — |
| Postural states + CSV       | 1.9 s  | — |
| **Annotated video**         | **62.1 s** | 3.19 GB |
- Detection is the dominant compute step; annotated video re-reads the entire video and
  dominates wall time. Disabled in batch runs via `save_annotated_video: false`.

### New config options (IR_medium.yaml)
- `normalize_illumination: true` — two-pass normalization (was already set)
- `save_annotated_video: false` — skip 60 s annotated video step in batch
- `boundary_margin: null` — auto = half median major_axis (~half worm body length)
- `max_area_cv: null` — disabled by default; ~0.4 would filter fragmented tracks

### Boundary margin filter
- Removes particles whose median centroid is within `boundary_margin` px of any frame edge.
- Auto-margin uses **half** the median `major_axis` (~half worm body length).
  One full body length was too conservative; half lets worms approach the edge without
  their centroid being censored.
- Implemented in `tracking.filter_boundary_particles()`.

### Per-frame Otsu vs global threshold
- Tested on Jan 21 (1800 frames, after normalization + backsub).
- Global Otsu from frame 0 = 41; per-frame mean = 39.1 ± 1.29 (range 36–43).
- Detection count std: 1.98 (global) vs 1.85 (per-frame) — negligible difference.
- **Conclusion**: normalization + backsub stabilises the histogram sufficiently.
  Per-frame Otsu not worth adding. Global threshold is appropriate.
- Note: using a temporal median frame for Otsu would NOT work — worm pixels regress
  to background in the temporal median, giving a unimodal (background-only) histogram.

### N2 batch QC trends (27 videos, Jan 20 – Mar 12 2026)
- Speed: ~0.15–0.26 mm/s in Jan–Feb; drops to ~0.07–0.12 mm/s in March recordings.
- Area: ~200–280 px² in most recordings; elevated in Jan 22–24 and Mar 6–7 (~370–430 px²).
  March recordings appear to be a distinct batch — different zoom or worm age.
- Track length: erratic (46–456 frames median). Short-track days (Mar 6, Mar 12, Jan 20)
  correlate with high particle counts → many fragmented detections.
- N worms: Feb 6 (333) and Feb 18 (319) are outliers (~3× typical). Likely debris/false
  detections that the boundary filter or area_cv filter may help remove.
- Jan 20 excluded from most analyses (bright outer ring artifact from 20260120 recording;
  confirmed as censored in earlier session).

---

## Session: 2026-05 (earlier) — Reversal detection, annotated video, batch reanalysis

### Reversal detection improvements
- Added area gate and speed gate to reduce false reversal calls.
- Merge adjacent reversals within `merge_reversal_gap` frames to repair splits
  from brief unreliable frames mid-reversal.
- `reversal_persistence`: consecutive qualifying frames required to confirm onset;
  filters single-frame spikes from forward-filled movement_angle during pauses.

### Annotated video mask snapping
- Tightened mask search radius from 50 px to 25 px to avoid snapping to nearby worm.

### Illumination normalization origin
- Problem: irregular LED flicker + slow oscillation in worm-pixel brightness
  (p99 std ~85–90% larger than post-correction).
- First identified on Mar 6 and Mar 7 recordings.
- normalization_comparison.png in dev/ shows the before/after for those two videos.

---

## Open questions / TODO

- [ ] Why does Mar 12 show almost no improvement from slow correction
      (CV 0.0435→0.0435)? Possibly a different instability type not captured by 10 s window.
- [ ] Zoom calibration for March recordings — confirm whether pixel_length differs.
      Speed and area both elevated in Mar 6/7, suggesting different magnification.
- [ ] Feb 6 and Feb 18 outlier particle counts — check whether boundary/area_cv filters
      bring them in line with other recordings.
- [ ] 2-state HMM needs retraining with quiescent-enriched data (~20–30% IPIs > 1 s).
      AMPD fires false peaks during quiescence; use scipy IPIs for HMM state classification.
