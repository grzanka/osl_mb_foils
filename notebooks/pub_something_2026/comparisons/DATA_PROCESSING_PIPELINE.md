# Data Processing Pipeline for Depth-Dose Profile Comparisons

This document describes the complete data analysis pipeline for comparing depth-dose profiles obtained from:
- **MC**: Monte Carlo simulations
- **EBT3**: Radiochromic film dosimetry
- **MBO**: Metal-bonded oxide foil detectors

Final results are presented in `2.0-comparisons.ipynb`.

---

## 1. Monte Carlo (MC) Data Processing

### 1.1 Raw Data Format

MC data originates from simulation output files (`.dat` format) with three columns:
- X coordinate (cm)
- Quantity value
- Uncertainty

Three separate files per facility:
| File | Contents |
|------|----------|
| `entryx_p1.dat` | Proton fluence |
| `entryx_p2.dat` | Kinetic energy |
| `entryx_p3.dat` | Dose |

### 1.2 Processing Pipeline

**Step 1: Data Loading**
```python
df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['x_cm', 'quantity', 'uncertainty'])
```

**Step 2: Unit Conversion and Normalization**
- Depth: cm → mm (×10)
- Dose: normalized to maximum value

```python
df_merged['depth'] = df_dose['x_cm'] * 10  # mm
df_merged['dose'] = df_dose['dose'] / df_dose['dose'].max()  # normalized
df_merged['dose_err'] = df_dose['dose_err'] / df_dose['dose'].max()
```

**Step 3: Depth Range Filtering**
- Filter to range: **-15 mm ≤ depth ≤ +17 mm**
- Total range after transformation: **0 to 32 mm**

**Step 4: Value Reversal and Depth Shift**
To align MC coordinate system with experimental data:
1. Reverse all quantity values (dose, fluence, kinetic energy)
2. Add 15 mm offset to depth values

```python
df_mc = df_merged[(df_merged['depth'] >= -15) & (df_merged['depth'] <= 17)].copy()
for col in ['dose', 'dose_err', 'fluence', 'fluence_err', 'kinetic_energy', 'kinetic_energy_err']:
    df_mc[col] = df_mc[col].iloc[::-1].values
df_mc['depth'] = df_mc['depth'] + 15
```

### 1.3 Output Files

| Facility | Output File | Shape |
|----------|-------------|-------|
| CCB | `mc_ccb_big.csv` | 320 × 7 |
| AIC144 | `mc_aic144_big.csv` | 320 × 7 |

Columns: `depth`, `dose`, `dose_err`, `fluence`, `fluence_err`, `kinetic_energy`, `kinetic_energy_err`

---

## 2. EBT3 Film Processing

### 2.1 Raw Data Format

- TIFF images (16-bit RGB) from flatbed scanner
- Resolution: 150 DPI or 72 DPI (pixel size: 0.169 mm or 0.353 mm)

Each measurement requires two scans:
- **Signal scan**: irradiated film
- **Background scan**: unexposed film from same lot

### 2.2 Dose Calibration

EBT3 proton calibration curve (20 Gy range):
```
D(Gy) = 7.94·netOD + 21.39·netOD² + 70.33·netOD³
```

Net optical density calculation:
```
netOD = log₁₀(I₀/I)
```

Default background intensities (16-bit):
| Channel | Value |
|---------|-------|
| Red | 42804.451 |
| Green | 44273.485 |
| Blue | 27929.477 |

### 2.3 Processing Pipeline

**Step 1: TIFF Loading**
```python
signal_raw = tifffile.imread(signal_path)  # shape: (H, W, 3)
signal_dpi = get_tiff_dpi(signal_path)
px_to_mm = 25.4 / signal_dpi
```

**Step 2: Dose Conversion**
```python
dose_signal = ebt3_dose_Gy(signal_raw)  # Uses red channel by default
```

**Step 3: Region-of-Interest Detection**

Auto-crop parameters:
| Parameter | CCB | AIC144 |
|-----------|-----|--------|
| `dose_threshold_gy` | 2.0 Gy | 0.5 Gy |
| `smooth_sigma_mm` | 0.2 mm | 0.1 mm |
| `min_diameter_mm` | 10.0 mm | 5.0 mm |
| `margin_mm` | 4.0 mm | 9.0 mm |

Algorithm:
1. Apply Gaussian smoothing (σ = `smooth_sigma_mm`)
2. Threshold at `dose_threshold_gy`
3. Connected component analysis
4. Select largest component with area ≥ π·(min_diameter/2)²
5. Expand bounding box by `margin_mm`

**Step 4: Background Recalibration**
1. Extract circular region (r = 10 mm) from background scan center
2. Calculate mean background intensity within circle
3. Recalculate dose using pair-specific background

```python
bg_mean_from_circle = np.mean(bg_raw_cropped[circular_mask])
new_bg_values = (bg_mean_from_circle, DEFAULT_BG_G, DEFAULT_BG_B)
dose_recalibrated = ebt3_dose_Gy(pair_data.signal_raw, bg_values=new_bg_values)
```

**Step 5: Rotation**
- Rotate 90° counter-clockwise (k=1) to align beam axis with X-axis

**Step 6: Pencil Mark Detection (CCB only)**

For beam incidence angle determination:
1. Binary threshold at dose > 1-3 Gy (CCB: 3 Gy, AIC144: 13 Gy)
2. Connected component labeling
3. Filter regions within expected Y ranges:
   - Mark 1: Y ∈ [40, 55] mm
   - Mark 2: Y ∈ [0, 10] mm
4. Fit line X = a·Y + b through both marks
5. Incidence angle: θ = arctan(a)

**Step 7: Profile Extraction**

Central profile extraction:
- Average over ±10 pixels (half-width) perpendicular to beam axis
- Profile position: image center

### 2.4 Output Files

| Facility | 2D Array | Profile |
|----------|----------|---------|
| CCB | `ebt_2D_ccb.npz` | `ebt_profile_ccb.csv` |
| AIC144 | `ebt_2D_aic144.npz` | `ebt_profile_aic144.csv` |

Profile columns: `x_mm`, `dose_Gy`

---

## 3. MBO Foil Processing

### 3.1 Raw Data Format

- TIFF images from optical scanner
- Pixel size: **74 µm** (0.074 mm/px)
- Two image types per foil:
  - `raw`: signal after irradiation
  - `lv`: live view reference

### 3.2 Preprocessing Pipeline

**Step 1: Image Loading and Edge Trimming**
```python
CUT_PX = 300  # pixels removed from all edges
image = read_tiff_img(path, cut_px=CUT_PX)
```

**Step 2: Noise Characterization**
Statistics from border strips (20 px width):
```python
strips = [image[:, :20], image[:, -20:], image[:20, :], image[-20:, :]]
noise_mean, noise_std = np.mean(strips), np.std(strips)
```

**Step 3: Detector Region Masking**

Parameters:
| Parameter | Value |
|-----------|-------|
| `GAUSSIAN_KERNEL` | 3 px |
| `MEDIAN_FILTER_SIZE` | 5 px |
| `N_SIGMA` | 7 |
| `PROBE_RADIUS` | 30 px |

```python
lv_smooth = gaussian_filter(lv_image, sigma=GAUSSIAN_KERNEL)
detector_mask = lv_smooth > noise_mean + N_SIGMA * noise_std
```

**Step 4: Background Subtraction**
```python
result = image - background_image + offset  # Safe subtraction for uint16
```

**Step 5: Circle Detection**

Hough transform for detector boundary:
```python
circle = find_circle_hough_method(img_for_detection)
```

### 3.3 Foil Matching and Alignment

For CCB data: merge adjacent foils to extend depth range.

**Step 1: Edge Detection**
- Sample positions: X = [140, 160, 180, 200, 220, 240, 260] px
- Threshold: 400 intensity units
- Stripe width: 10 px for averaging

**Step 2: Edge Line Fitting**
- Robust linear fit to edge crossing points
- Calculate edge slope in degrees

**Step 3: Image Rotation**
```python
rotation_angle = edge_slope_deg + 90  # Make edge vertical
rotated_image = rotate_image(image, rotation_angle)
```

**Step 4: Centering and Cropping**
- Crop size: 300 × 300 px (22.2 × 22.2 mm)
- Center on detector circle

**Step 5: Circle Refinement (Gradient-Based)**

Parameters:
| Parameter | Value |
|-----------|-------|
| `CIRCLE_GRADIENT_HIGH` | 300 |
| `CIRCLE_GRADIENT_LOW` | 200 |
| `CIRCLE_EDGE_DISTANCE_MM` | 0.5 mm |
| `CIRCLE_RADIUS_TOLERANCE` | 5% |

Algorithm:
1. Radial scanning from center (360 angles)
2. Detect intensity drop from HIGH → LOW threshold
3. Filter edge points within 0.5 mm of initial circle
4. Least-squares circle fit constrained to ±5% initial radius

**Step 6: Image Merging**

Optimization for merge margin:
1. Extract sections for fitting:
   - Section 1: X ∈ [10, 16] mm
   - Section 2: X ∈ [21, 27] mm
   - Y averaging: [9, 13] mm
2. Apply Gaussian smoothing (σ = 4 px)
3. Test shift range: [-5, +5] mm (500 steps)
4. Fit function: parabola (2nd degree polynomial)
5. Minimize residual sum of squares (RSS)
6. Apply optimal shift to merge margin

**Step 7: Profile Extraction**
- Y range for averaging: 9-13 mm
- Gaussian smoothing: σ = 4 px (0.296 mm)

### 3.4 Output Files

| Facility | Merged Image | Profile |
|----------|--------------|---------|
| CCB | `merged_foils_3_2.npz` | `profile_foils_3_2.npz` |
| AIC144 | `mbo_aic144_single_foil_rotated.npz` | - |

Profile arrays: `x_mm`, `profile_opt_smooth`

---

## 4. Final Comparison Processing

### 4.1 Data Alignment

**MBO-EBT3 Alignment:**
1. Find maximum position in each profile
2. Shift MBO X-coordinates to match EBT3 maximum position:
   ```python
   mbo_x_shift = ebt_max_x - mbo_max_x
   mbo_x_shifted = mbo_x + mbo_x_shift
   ```
3. Scale MBO values to match EBT3 at X = 0:
   ```python
   scale_mbo = ebt_dose_at_0 / mbo_value_at_0
   mbo_values_scaled = mbo_values * scale_mbo
   ```

**MC-EBT3 Scaling:**
```python
scale_mc = ebt_dose_at_0 / mc_dose_at_0
mc_dose_scaled = mc_dose * scale_mc
```

### 4.2 Metrics Calculation

Metrics computed using linear interpolation on dense grid (n=1000 points):

**Full Width at Half Maximum (FWHM):**
1. Find maximum value and half-maximum threshold
2. Locate left edge: first crossing from below
3. Locate right edge: last crossing from above
4. FWHM = right_edge - left_edge

**Range at 90% (X₉₀):**
- Position where dose drops to 90% on distal side (after maximum)

**Distal Falloff (80-20%):**
- Distance from 80% to 20% dose level on distal side

```python
x_dense, y_dense = get_dense_data(x, y, n_points=1000)
fwhm, _, _, _ = calculate_fwhm(x_dense, y_dense)
x90 = find_distal_percent(x_dense, y_dense, 0.9)
falloff_80_20 = x20 - x80
```

### 4.3 Typical Metrics Results

| Metric | CCB EBT3 | CCB MBO | AIC144 EBT3 | AIC144 MBO |
|--------|----------|---------|-------------|------------|
| FWHM | ~23 mm | ~23 mm | ~22 mm | ~22 mm |
| Range 90% | ~20 mm | ~20 mm | ~19 mm | ~19 mm |
| Falloff 80-20% | ~2 mm | ~2 mm | ~2 mm | ~2 mm |

---

## 5. Data File Summary

### 5.1 Input Data Locations

```
raw_data/
├── mc_ccb/big/           # MC simulation outputs (CCB)
│   ├── entryx_p1.dat     # Fluence
│   ├── entryx_p2.dat     # Kinetic energy
│   └── entryx_p3.dat     # Dose
├── mc_aic144/big/        # MC simulation outputs (AIC144)
├── ebt_ccb/              # EBT3 film scans (CCB)
│   ├── szyba1_150dpi/    # With glass, 150 DPI
│   ├── szyba1_72dpi/     # With glass, 72 DPI
│   └── bez_szyby_150dpi/ # Without glass, 150 DPI
├── ebt_aic144/           # EBT3 film scans (AIC144)
├── foils_mbo_ccb/2025-12-19/  # MBO foils (CCB)
│   └── tlo/              # Background scans
└── foils_mbo_aic144/2025-11-25/  # MBO foils (AIC144)
```

### 5.2 Intermediate Data Files

```
pub_something_2026/
├── mc/
│   ├── mc_ccb_big.csv
│   └── mc_aic144_big.csv
├── ebt_foils/
│   ├── ebt_2D_ccb.npz
│   ├── ebt_2D_aic144.npz
│   ├── ebt_profile_ccb.csv
│   └── ebt_profile_aic144.csv
├── mbo_foils/
│   ├── mbo_profile_ccb.csv
│   └── mbo_profile_aic144.csv
└── comparisons/
    └── 2.0-comparisons.ipynb  # Final comparison notebook
```

---

## 6. Processing Notebooks Reference

| Stage | Notebook | Description |
|-------|----------|-------------|
| MC CCB | `mc/0.3_ccb_mc_wedge_big.ipynb` | Load, filter, transform MC data |
| MC AIC144 | `mc/0.3_aic144_mc_wedge_big.ipynb` | Load, filter, transform MC data |
| EBT CCB | `ebt_foils/1.0_ebt_ccb_analysis.ipynb` | TIFF processing, calibration, cropping |
| EBT AIC144 | `ebt_foils/1.0_ebt_aic144_analysis.ipynb` | TIFF processing, calibration, cropping |
| MBO CCB Explore | `mbo_foils/0.1-explore-ccb.ipynb` | Raw data exploration, background subtraction |
| MBO CCB Match | `mbo_foils/0.2-match-ccb.ipynb` | Edge detection, rotation, merging |
| MBO AIC144 | `mbo_foils/0.2-match-aic144_single.ipynb` | Single foil processing |
| **Comparison** | `comparisons/2.0-comparisons.ipynb` | Profile alignment, metrics, visualization |

---

## 7. Key Processing Parameters Summary

| Domain | Parameter | Value |
|--------|-----------|-------|
| **MC** | Depth filter range | [-15, +17] mm |
| **MC** | Depth offset | +15 mm |
| **EBT3** | Calibration coefficients | a₁=7.94, a₂=21.39, a₃=70.33 |
| **EBT3** | Background (Red channel) | 42804.451 |
| **EBT3** | ROI detection threshold | 0.5-2.0 Gy |
| **EBT3** | Rotation | 90° CCW |
| **MBO** | Pixel size | 74 µm |
| **MBO** | Edge cut | 300 px |
| **MBO** | Noise threshold | 7σ |
| **MBO** | Circle radius tolerance | ±5% |
| **MBO** | Profile smoothing | σ = 4 px |
| **Comparison** | Interpolation points | 1000 |
| **Comparison** | Plot X range | [-10, 50] mm |

---

*Document generated: February 2026*
*Pipeline version: pub_something_2026*
