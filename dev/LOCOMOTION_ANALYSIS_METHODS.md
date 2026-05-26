# Methods: C. elegans locomotion analysis

Script: `dev/batch_postural_comparison.py`

## Dependencies

- Python: ODLabTracker package, numpy, pandas, matplotlib, scipy, pyarrow
- R: lme4 (≥1.1), lmerTest (≥3.1)

Install Python dependencies via the ODLabTracker package. Install R packages with:

```r
install.packages(c("lme4", "lmerTest"))
```

## Running the analysis

```bash
# First run — scan raw data from dataset directory:
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

The exclusion file is a CSV with columns `genotype,date` (date as YYYYMMDD). Rows matching
those genotype–date pairs are dropped before normalization. Lines starting with `#` are ignored.

## Data extraction (`_parse_tracks`, `load_recording_particles`)

Per-particle locomotion metrics were extracted from `tracks.csv` files produced by ODLabTracker.
Particles with a mean all-frame speed < 0.03 mm/s were excluded to remove quiescent animals.
Forward-run speed was computed as the per-particle median speed across frames classified as
`forward_run` by the ODLabTracker movement classifier. Reversal and pirouette rates were
computed as events per minute of total recording duration.

## Normalization (`add_normalization`)

Recording-level speed values were expressed as fold-change relative to the mean N2 speed
recorded on the same date. Recordings without a same-date N2 were normalized to the grand N2
mean across all dates and shown as open circles in figures.

## Statistical model (`fit_lme_stats`)

Genotype effects on forward-run speed were estimated using a linear mixed-effects model fit
via lme4 and lmerTest in R (called via subprocess). Speed was modeled at the per-frame level
within forward-run epochs:

```
speed ~ genotype + date + (1|recording_id) + (1|particle_uid)
```

**Why fixed date effects?** Date was treated as a fixed effect rather than a random effect.
With ~3 recordings per genotype, the between-date variance is poorly estimated and random date
effects undergo REML shrinkage that fails to fully absorb day-to-day variability in N2
locomotion. This biases genotype fold-change estimates for strains measured on atypically fast
or slow N2 days. Fixed date effects are equivalent to within-date normalization and do not
shrink (`keep_n2_dates()` in `fit_lme_stats()`).

Only recordings from dates on which N2 controls were also recorded were included in the LME.
Genotypes without any same-date N2 are excluded from the model (fold-change reported as NaN).

Genotype coefficients were expressed as fold-changes relative to the N2 LME intercept.
P-values used Satterthwaite denominator degrees-of-freedom approximation (lmerTest), reflecting
the number of independent recordings rather than the number of particles. Multiple comparisons
were controlled by Benjamini-Hochberg FDR correction (`_bh_correct()`) across all testable
genotypes per metric.

## Visualization (`make_plot`)

The strip plot (left panel) shows per-recording normalized speed values. Filled circles
indicate recordings with a same-date N2 reference; open circles indicate recordings normalized
to the grand N2 mean. LME point estimates and 95% confidence intervals (±1.96 SE) are shown
as diamonds with horizontal bars. Significance thresholds: * q < 0.05, ** q < 0.01,
*** q < 0.001. Genotypes are ordered by ascending LME speed fold-change (`genotype_order()`).

The distribution panel (right panel) shows kernel density estimates (Gaussian kernel,
bandwidth = 0.30) of per-particle forward-run speeds for each recording, normalized to the
same-date N2 mean. Each recording is a separate violin; N2 recordings are pooled.

## Reproducing from the supplemental table

The file `supplemental_particle_data.csv` contains per-particle raw data with columns:

| Column | Description |
|--------|-------------|
| `genotype` | Strain name |
| `date` | Recording date (YYYYMMDD) |
| `recording_id` | Unique recording identifier (`genotype##date`) |
| `forward_run_speed_mm_s` | Per-particle median forward-run speed (mm/s) |
| `reversal_rate_per_min` | Reversals per minute |
| `pirouette_rate_per_min` | Pirouettes per minute |

To reproduce the LME speed model in R directly from this table:

```r
library(lme4)
library(lmerTest)

d <- read.csv("supplemental_particle_data.csv")
d$genotype <- relevel(as.factor(d$genotype), ref = "N2")
d$date     <- factor(d$date)

# Per-particle forward-run speed model
# (equivalent to the per-frame model in the script; particle medians used here)
fit <- lmer(forward_run_speed_mm_s ~ genotype + date + (1|recording_id),
            data = d, REML = TRUE,
            control = lmerControl(optimizer = "bobyqa"))
summary(fit)
```

Note: the published analysis used per-forward-run-frame data (not per-particle medians) with
an additional `(1|particle_uid)` random effect. The per-particle model above is a close
approximation and does not require the full ~3 M-row frame-level dataset.
