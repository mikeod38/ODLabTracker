# Postural comparison across genotypes

`dev/batch_postural_comparison.py` aggregates ODLabTracker tracking output across a multi-genotype dataset, computes per-particle locomotion metrics, normalizes to same-date N2 controls, fits linear mixed-effects models (LME) via R/lme4, and produces a publication-ready figure.

## Dependencies

- Python: ODLabTracker package, numpy, pandas, matplotlib, scipy, pyarrow
- R: lme4 (≥ 1.1), lmerTest (≥ 3.1)

Install R packages with:

```r
install.packages(c("lme4", "lmerTest"))
```

## Running the analysis

```bash
# First run — scan raw tracking output from a dataset directory:
python dev/batch_postural_comparison.py \
    --data-dir /path/to/dataset \
    --out-dir  results/ \
    --exclude  exclude.csv \
    --title    "My experiment — forward-run speed"

# Replot from cached data (no NAS required):
python dev/batch_postural_comparison.py --out-dir results/

# Re-run LME models only (data already cached):
python dev/batch_postural_comparison.py --out-dir results/ --refit

# Full rescan + refit:
python dev/batch_postural_comparison.py --out-dir results/ --data-dir /path/to/dataset --refresh
```

The dataset directory should contain one subdirectory per genotype, each containing dated recording folders with `tracks.csv` files produced by ODLabTracker postural mode.

The exclusion file is a CSV with columns `genotype,date` (date as YYYYMMDD). Rows matching those genotype–date pairs are dropped before normalization. Lines starting with `#` are ignored.

## Outputs

| File | Description |
|------|-------------|
| `postural_comparison.csv` | Per-recording summary: speed, reversal rate, pirouette rate, normalized values |
| `postural_comparison_stats.csv` | LME coefficients, fold-changes, and FDR-corrected q-values per genotype |
| `postural_comparison.png` | Strip plot + per-recording speed distributions |
| `supplemental_particle_data.csv` | Per-particle raw metrics for independent reanalysis |
| `particle_data.parquet` | Cached per-particle data (re-used on subsequent runs) |
| `fwd_frame_data.parquet` | Cached per-forward-run-frame data used by the LME |

## Statistical model and methods

See [dev/LOCOMOTION_ANALYSIS_METHODS.md](../dev/LOCOMOTION_ANALYSIS_METHODS.md) for full details on the normalization strategy, LME model structure, and R code to reproduce the analysis from the supplemental table.

## Using the core functions independently

The analysis functions are available as a Python package after `pip install -e .`:

```python
from ODLabTracker.locomotion import (
    scan_dataset_particles,   # load per-particle metrics from a dataset
    add_normalization,        # fold-change vs same-date N2
    fit_lme_stats,            # R/lme4 LME via subprocess
    genotype_order,           # sort genotypes by LME fold-change
)
```
