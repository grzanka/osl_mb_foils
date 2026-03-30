# Cross-Modality Comparisons

Combines MC simulation, EBT3 film, and MBO foil detector measurements for cross-modality and cross-facility comparison.

## Pipelines

### facility.py — `comparison_facility`

Loads processed data from all three modalities for a single facility, scales MC and MBO profiles to match EBT3 dose at a reference position, and generates overlay plots with FWHM and range annotations.

**Input:** MC CSV, EBT CSV, MBO NPZ
**Output:** PDF with scaled overlays, metrics boxes, summary table

Key parameters:
- `normalize_at_x`: X position (mm) where MC and MBO are scaled to match EBT dose
- `mbo_gap_exclude_min_mm` / `mbo_gap_exclude_max_mm`: MBO gap region to exclude (CCB only, 9–16 mm)

### summary.py — `comparison_summary`

Loads all six data sources (MC/EBT/MBO × CCB/AIC-144), creates per-modality cross-facility plots, per-facility paired comparisons, a combined normalized plot, and a full metrics difference table.

**Input:** All processed CSV and NPZ files from upstream pipelines
**Output:** PDF with 7+ figures and full metrics table; CSV with summary metrics

Key parameters:
- `reference_dose_gy`: Reference dose for normalization

## Config Files

| File | Description |
|------|-------------|
| facility_aic144.yaml | AIC-144: MC vs EBT3 vs MBO (includes single-foil variant) |
| facility_ccb.yaml | CCB: MC vs EBT3 vs MBO (with gap exclusion 9–16 mm) |
| summary.yaml | Full cross-facility, cross-modality summary |
