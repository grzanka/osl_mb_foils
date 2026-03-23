# MBO Foil Detector Analysis

Processes multi-blade oscillating (MBO) foil detector data: background subtraction, two-foil edge matching, profile extraction, and cross-facility comparison.

## Pipelines

### explore.py — `mbo_explore`

Loads detector data collection, computes noise statistics, subtracts background using sigma-clipping, and generates overview visualizations of each detector.

**Input:** Directory of detector data files
**Output:** PDF with detector images, noise statistics; PKL with processed data

Key parameters:
- `cut_px`: Number of pixels to crop from edges
- `median_filter_size`: Size of median filter for denoising
- `n_sigma`: Sigma threshold for background subtraction
- `probe_radius`: Radius for noise probe region

### match.py — `mbo_match`

Performs two-foil matching: loads the processed detector data, detects edges, rotates and crops individual foils, refines circle positions via gradient-based edge detection, applies vertical shifts, merges two foil images with optimal margin, and extracts horizontal profiles using spline fitting.

**Input:** PKL file from explore pipeline
**Output:** PDF with matching visualizations; NPZ with merged image and extracted profile

Key parameters:
- `left_foil_id` / `right_foil_id`: Which detectors to merge
- `edge_threshold`: Threshold for initial edge detection
- `circle_gradient_high/low`: Thresholds for gradient-based circle refinement
- `circle_edge_distance_mm`: Maximum distance from initial circle for edge point filtering
- `profile_y_min_mm` / `profile_y_max_mm`: Vertical range for profile extraction
- `section1/2_x_min/max`: Horizontal sections for spline fitting

### comparison.py — `mbo_comparison`

Loads profile NPZ files from both facilities, normalizes to a reference dose at a reference position, and compares profiles with FWHM, range, and distal falloff metrics.

**Input:** NPZ files from match pipeline
**Output:** PDF with profile overlay, normalized falloff, metrics table; CSV with profiles

Key parameters:
- `reference_dose_gy`: Dose value for normalization
- `reference_x_mm`: Position for normalization

## Config Files

| File | Description |
|------|-------------|
| explore_aic144.yaml | AIC-144 detector exploration |
| explore_ccb.yaml | CCB detector exploration |
| match_aic144.yaml | AIC-144 two-foil matching (foils 1 & 2) |
| match_ccb.yaml | CCB two-foil matching (foils 2 & 3) |
| comparison.yaml | Cross-facility MBO comparison |
