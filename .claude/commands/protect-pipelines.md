Before finishing any code change in this project, verify that the other core analysis pipelines are not broken.

The project has three independent analysis modes, each with its own entry point and shared library code in `src/ODLabTracker/tracking.py`:

| Pipeline | Entry point | Key functions |
|---|---|---|
| Centroid / Postural | `FastTrack.py` | `collect_detections`, `link_tracks`, `calculate_speed_parameters`, `calculate_postural_states`, `create_annotated_video` |
| Pumping | `FastTrackPumping.py` | `process_video` (pumping-specific), `collect_detections`, `subtract_background` |
| Batch parallel | `run_fasttrack_parallel.py` | calls FastTrack.py per file |

**Rules:**
1. Any change to `tracking.py` must not alter function signatures in a backwards-incompatible way without updating all callers (`FastTrack.py`, `FastTrackPumping.py`, `run_fasttrack_parallel.py`).
2. Pumping-specific code (`process_video`, HMM peak detection, pumping summary) must not be coupled to postural state variables (`is_reversal`, `is_pirouette`, `movement_type`), and vice versa.
3. Shared utility functions (`preprocess_frame`, `collect_detections`, `link_tracks`, `filter_short_tracks`, `calculate_speed_parameters`, `subtract_background`, `plot_trajectories`) must remain compatible with both pipelines.
4. Config YAML keys used by one pipeline must not be silently required by another — use `.get()` with a safe default for any key not present in all configs.

**Check to run after changes:**

```bash
cd /Users/mikeodonnell/git/ODLabTracker && ~/.pyenv/shims/python dev/check_deps.py
```

Then inspect `dependency_report.md` for any newly broken imports or missing declarations.

Also grep for any function you renamed or removed to confirm no other pipeline calls it:
```bash
grep -r "FUNCTION_NAME" /Users/mikeodonnell/git/ODLabTracker --include="*.py"
```
