"""Depth-dose metrics calculations for beam characterization."""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def calculate_depth_dose_metrics(
    df_data: pd.DataFrame,
    data_name: str = "Data",
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    ref_x_coord: Optional[float] = None,
    interp_func: Optional[Callable] = None,
    verbose: bool = True
) -> Optional[Dict]:
    """Calculate depth-dose metrics for a dataset.
    
    Calculates standard dosimetric parameters from a depth-dose curve:
    - Range (R90): depth where dose equals 90% of maximum
    - Distal falloff: distance from 90% to 10% dose
    - FWHM: full width at half maximum
    - Max-to-plateau ratio
    
    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame with depth and dose columns
    data_name : str
        Name of the dataset for display. Default "Data".
    x_col : str, optional
        Name of x-axis (depth) column. If None, auto-detects 'x' or 'depth'.
    y_col : str, optional
        Name of y-axis (dose) column. If None, auto-detects 'y' or 'dose'.
    ref_x_coord : float, optional
        Reference x-coordinate for interpolating the first point value
        (used for MC data comparison with experimental data).
    interp_func : callable, optional
        Interpolation function for first point value calculation.
    verbose : bool
        Whether to print results. Default True.
    
    Returns
    -------
    dict or None
        Dictionary with calculated metrics, or None if data is empty.
        Keys: 'dose_09', 'dose_01', 'dose_drop_09_to_01', 'fwhm_left',
              'fwhm_right', 'fwhm', 'first_point_inverse'
              
    Notes
    -----
    Dose values are assumed to be normalized (max = 1.0).
    Depth values can be in any unit (typically mm or cm).
    """
    if df_data is None or df_data.empty:
        print(f"{data_name} not loaded")
        return None
    
    # Auto-detect column names
    if x_col is None:
        x_col = 'x' if 'x' in df_data.columns else 'depth'
    if y_col is None:
        y_col = 'y' if 'y' in df_data.columns else 'dose'
    
    metrics = {}
    
    # 1. Find distance where dose is 0.9 (Range - scanning right to left)
    dose_09 = None
    for i in range(len(df_data) - 1, 0, -1):
        y_curr = df_data[y_col].iloc[i]
        y_prev = df_data[y_col].iloc[i-1]
        if y_curr <= 0.9 and y_prev > 0.9:
            x_curr, x_prev = df_data[x_col].iloc[i], df_data[x_col].iloc[i-1]
            dose_09 = x_prev + (0.9 - y_prev) * (x_curr - x_prev) / (y_curr - y_prev)
            break
    
    # 1b. Find distance where dose drops to 0.1 (scanning right to left)
    dose_01 = None
    for i in range(len(df_data) - 1, 0, -1):
        y_curr = df_data[y_col].iloc[i]
        y_prev = df_data[y_col].iloc[i-1]
        if y_curr <= 0.1 and y_prev > 0.1:
            x_curr, x_prev = df_data[x_col].iloc[i], df_data[x_col].iloc[i-1]
            dose_01 = x_prev + (0.1 - y_prev) * (x_curr - x_prev) / (y_curr - y_prev)
            break
    
    # Calculate drop distance from 0.9 to 0.1
    dose_drop_09_to_01 = None
    if dose_09 is not None and dose_01 is not None:
        dose_drop_09_to_01 = dose_01 - dose_09
    
    # 2. Calculate FWHM (Full Width at Half Maximum)
    # Scanning left to right (first crossing at 0.5)
    fwhm_left = None
    for i in range(len(df_data) - 1):
        y1, y2 = df_data[y_col].iloc[i], df_data[y_col].iloc[i+1]
        if y1 < 0.5 and y2 >= 0.5:
            x1, x2 = df_data[x_col].iloc[i], df_data[x_col].iloc[i+1]
            fwhm_left = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
            break
    
    # Scanning right to left (second crossing at 0.5)
    fwhm_right = None
    for i in range(len(df_data) - 1, 0, -1):
        y_curr = df_data[y_col].iloc[i]
        y_prev = df_data[y_col].iloc[i-1]
        if y_curr < 0.5 and y_prev >= 0.5:
            x_curr, x_prev = df_data[x_col].iloc[i], df_data[x_col].iloc[i-1]
            fwhm_right = x_prev + (0.5 - y_prev) * (x_curr - x_prev) / (y_curr - y_prev)
            break
    
    fwhm = None
    if fwhm_left is not None and fwhm_right is not None:
        fwhm = fwhm_right - fwhm_left
    
    # 3. Calculate inverse of the first point value (Max to Plateau Ratio)
    if ref_x_coord is not None and interp_func is not None:
        try:
            first_point_value = float(interp_func(ref_x_coord))
            first_point_inverse = 1.0 / first_point_value
        except:
            first_point_inverse = 1.0 / df_data[y_col].iloc[0]
    else:
        first_point_inverse = 1.0 / df_data[y_col].iloc[0]
    
    metrics['dose_09'] = dose_09
    metrics['dose_01'] = dose_01
    metrics['dose_drop_09_to_01'] = dose_drop_09_to_01
    metrics['fwhm_left'] = fwhm_left
    metrics['fwhm_right'] = fwhm_right
    metrics['fwhm'] = fwhm
    metrics['first_point_inverse'] = first_point_inverse
    
    if verbose:
        print(f"=== {data_name} Metrics ===")
        print(f"Range (distance at dose 0.9): {dose_09 / 10:.4f} cm" if dose_09 is not None else "Range (distance at dose 0.9): Not found")
        print(f"Distance at dose 0.1: {dose_01 / 10:.4f} cm" if dose_01 is not None else "Distance at dose 0.1: Not found")
        print(f"Drop from 0.9 to 0.1: {dose_drop_09_to_01 / 10:.4f} cm" if dose_drop_09_to_01 is not None else "Drop from 0.9 to 0.1: Not calculated")
        print(f"FWHM: {fwhm / 10:.4f} cm" if fwhm is not None else "FWHM: Not calculated")
        print(f"Left edge of FWHM (0.5): {fwhm_left / 10:.4f} cm" if fwhm_left is not None else "Left edge: Not found")
        print(f"Right edge of FWHM (0.5): {fwhm_right / 10:.4f} cm" if fwhm_right is not None else "Right edge: Not found")
        print(f"Inverse of first point value: {first_point_inverse:.4f}")
        print()
    
    return metrics


def calculate_ratio_statistics(
    df_exp: pd.DataFrame,
    df_mc: pd.DataFrame,
    threshold: float = 0.1,
    exp_x_col: str = 'x',
    exp_y_col: str = 'y',
    mc_x_col: str = 'depth',
    mc_y_col: str = 'dose'
) -> Tuple[pd.Series, Dict]:
    """Calculate ratio between experimental and MC data.
    
    Parameters
    ----------
    df_exp : pd.DataFrame
        Experimental data with depth and dose columns
    df_mc : pd.DataFrame
        MC data with depth and dose columns
    threshold : float
        Minimum dose value for ratio calculation. Default 0.1.
    exp_x_col, exp_y_col : str
        Column names for experimental data
    mc_x_col, mc_y_col : str
        Column names for MC data
        
    Returns
    -------
    ratio : pd.Series
        Ratio (exp/MC) at each experimental point where dose > threshold
    stats : dict
        Statistics: 'mean', 'min', 'max', 'std', 'n_points'
    """
    # Create interpolation function for MC data
    mc_interp = interp1d(df_mc[mc_x_col], df_mc[mc_y_col], kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    
    # Interpolate MC values at experimental data points
    mc_interp_values = mc_interp(df_exp[exp_x_col])
    
    # Calculate ratio only where experimental dose > threshold
    ratio = pd.Series(index=df_exp.index, dtype=float)
    valid_mask = df_exp[exp_y_col] > threshold
    ratio[valid_mask] = df_exp.loc[valid_mask, exp_y_col] / mc_interp_values[valid_mask]
    
    stats = {
        'n_points': valid_mask.sum(),
        'mean': ratio[valid_mask].mean(),
        'min': ratio[valid_mask].min(),
        'max': ratio[valid_mask].max(),
        'std': ratio[valid_mask].std()
    }
    
    return ratio, stats


def print_metrics_comparison(
    metrics_exp: dict,
    metrics_mc: dict,
    title: str = "EXPERIMENTAL DATA vs MC DATA"
) -> None:
    """Print a comparison table of metrics between experimental and MC data.
    
    Parameters
    ----------
    metrics_exp : dict
        Metrics dictionary for experimental data
    metrics_mc : dict
        Metrics dictionary for MC data
    title : str
        Title for the comparison table
    """
    print("="*90)
    print(f"FEATURE COMPARISON: {title}")
    print("="*90)
    print()
    
    features = [
        ("Range (dose 90%)", "mm", lambda m: m['dose_09']),
        ("Falloff 90% to 10%", "mm", lambda m: m['dose_drop_09_to_01']),
        ("FWHM", "mm", lambda m: m['fwhm']),
        ("Max to Plateau Ratio", "-", lambda m: m['first_point_inverse']),
    ]
    
    print(f"{'Feature':<30} {'Unit':<6} {'Experimental':<16} {'MC':<16} {'Absolute Diff':<16} {'Relative Diff (%)':<16}")
    print("-"*100)
    
    for feature_name, unit, getter in features:
        exp_val = getter(metrics_exp)
        mc_val = getter(metrics_mc)
        
        if exp_val is not None and mc_val is not None:
            abs_diff = mc_val - exp_val
            rel_diff = (abs_diff / exp_val) * 100
            print(f"{feature_name:<30} {unit:<6} {exp_val:>15.4f} {mc_val:>15.4f} {abs_diff:>15.4f} {rel_diff:>15.2f}%")
        else:
            print(f"{feature_name:<30} {unit:<6} {'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>15}")
    
    print("-"*100)
    print()
    print("Note: Relative difference is calculated as (MC - Exp) / Exp * 100")
    print("      Negative values indicate MC is smaller/lower than Experimental data")
    print()
