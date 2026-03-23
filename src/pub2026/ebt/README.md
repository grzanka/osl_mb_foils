# EBT3 Film Dosimetry

Processes EBT3 radiochromic film scans for dose profile extraction and cross-facility comparison.

## Pipelines

### analysis.py — `ebt_analysis`

Processes pairs of EBT3 film scans (signal + background): loads TIFF pairs, calibrates to dose, detects and crops to the irradiated region, identifies pencil alignment marks, and extracts horizontal dose profiles.

**Input:** TIFF file pairs (signal and background scans)
**Output:** PDF with dose maps, pencil mark detection, profile plots; NPZ with extracted profiles

Key parameters:
- `file_pairs`: List of signal/background TIFF path pairs
- `crop_dose_threshold_gy`: Threshold for detecting irradiated region
- `pencil_threshold_gy`: Threshold for pencil mark detection
- `pencil_mark1/2_y_range_mm`, `pencil_mark1/2_x_range_mm`: Search regions for pencil marks
- `circle_radius_mm`: Expected irradiated circle radius
- `rotation_k`: Number of 90° rotations to apply

### comparison.py — `ebt_comparison`

Loads processed EBT profiles from both facilities, shifts them to align pencil mark positions, and compares dose profiles with FWHM and distal falloff metrics.

**Input:** NPZ files from the analysis pipeline
**Output:** PDF with dose maps, profile overlays, metrics; CSV with profiles

Key parameters:
- `pencil_mark_x_aic144_mm` / `pencil_mark_x_ccb_mm`: Pencil mark positions for alignment
- `strip_width_mm`: Width of horizontal strip for profile averaging

## Config Files

| File | Description |
|------|-------------|
| analysis_aic144.yaml | AIC-144 film processing (2 pairs) |
| analysis_ccb.yaml | CCB film processing (3 pairs) |
| comparison.yaml | Cross-facility comparison |
