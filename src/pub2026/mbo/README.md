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

### background_subtraction_merge.py — `mbo_background_subtraction_merge`

Merges two adjacent foils after background subtraction. Takes per-foil images from the background-subtraction step (Scenario A difference and Scenario B ratio), detects edges on original aligned images, and merges the foil pairs into panoramic images. Produces two merges per scenario: a simple merge and a shift-optimised merge where the inter-foil distance is adjusted to best match a smooth depth-dose profile shape.

**Input:** Background subtraction NPZ + aligned NPZ files (reference and target)
**Output:** PDF with merged 2D images, horizontal profiles at multiple Y offsets, RSS optimisation plots; NPZ with merged images

Key parameters:
- `bg_subtraction_npz`: NPZ from the background subtraction step
- `reference_npz` / `target_npz`: Aligned NPZs (for edge detection)
- `foil_ids`: Which foils to merge (left, right)
- `edge_threshold`: Threshold for edge detection on aligned images
- `profile_y_offsets_mm`: Y offsets from circle centre for profile extraction
- `section1/2_x_min/max`: Horizontal sections for spline shift optimisation
- `vmax_scenario_a` / `vmax_scenario_b`: Intensity limits for visualisation

## Config Files

| File | Description |
|------|-------------|
| explore_aic144.yaml | AIC-144 detector exploration |
| explore_ccb.yaml | CCB detector exploration |
| match_aic144.yaml | AIC-144 two-foil matching (foils 1 & 2) |
| match_ccb.yaml | CCB two-foil matching (foils 2 & 3) |
| comparison.yaml | Cross-facility MBO comparison |
| post_align_bg_sub_merge_ccb.yaml | CCB foil merge after background subtraction |
