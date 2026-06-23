# Monte Carlo Simulation Analysis

Processes MC simulation data from SHIELD-HIT12A for depth-dose validation and wedge profile analysis.

## Pipelines

### depth_validation.py — `mc_depth_validation`

Compares MC simulated Bragg peak with experimental ion chamber measurements at CCB.

**Input:** MC depth-dose CSV (e.g., `graph_z_profile_DOSE.csv`)
**Output:** PDF with overlay plot, experimental/MC ratio, and statistical summary

Key parameters:
- `mc_csv_path`: Path to MC simulation CSV
- `threshold`: Minimum dose threshold for ratio calculation (default: 0.1)

### wedge_profile.py — `mc_wedge`

Processes MC wedge simulation profiles for dose, fluence, and energy. Supports both CSV (small geometry) and DAT (big geometry) file formats.

**Input:** Three MC files (dose, fluence, energy)
**Output:** PDF with triple-axis overlay, individual quantity plots, dose markers (FWHM, R90, R80, R20), and processed CSV

Key parameters:
- `file_format`: `"csv"` for small geometry, `"dat"` for big geometry
- `depth_filter_min_mm` / `depth_filter_max_mm`: Depth filtering range
- `depth_shift_mm`: Shift to align depth origin

### comparison.py — `mc_comparison`

Compares MC profiles between AIC-144 and CCB facilities.

**Input:** Two processed CSV files (from wedge pipeline)
**Output:** PDF with normalized overlay, difference plots, and metrics table

## Config Files

| File | Description |
|------|-------------|
| depth_validation_ccb.yaml | CCB Bragg peak validation |
| wedge_ccb_small.yaml | CCB small geometry wedge |
| wedge_ccb_big.yaml | CCB big geometry wedge |
| wedge_aic144_big.yaml | AIC-144 big geometry wedge |
| comparison.yaml | AIC-144 vs CCB comparison |
