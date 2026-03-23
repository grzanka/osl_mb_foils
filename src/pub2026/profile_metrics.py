"""Shared profile analysis utilities for pub2026 pipelines.

Consolidated from duplicated implementations across mc, ebt, mbo, and comparisons modules.
Provides FWHM, distal percentage, falloff, dense interpolation, and MBO alignment.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d


def get_dense(x: np.ndarray,
              y: np.ndarray,
              n: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Densely interpolate a profile for smoother metric calculations.

    Parameters
    ----------
    x, y : array-like
        Original profile coordinates.
    n : int
        Number of output points.

    Returns
    -------
    xd, yd : dense coordinate arrays.
    """
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
    xd = np.linspace(float(np.min(x)), float(np.max(x)), n)
    return xd, f(xd)


def find_fwhm(
        profile: np.ndarray,
        x_mm: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
    """Find FWHM boundaries by linear interpolation.

    Parameters
    ----------
    profile : 1-D array of values.
    x_mm : 1-D array of positions (same length as *profile*).

    Returns
    -------
    (left, right, half_max) — positions of left/right crossings and the
    half-maximum value.  *left* or *right* may be ``None`` if no crossing
    is found.
    """
    profile = np.asarray(profile, dtype=float)
    x_mm = np.asarray(x_mm, dtype=float)

    max_val = float(np.max(profile))
    max_idx = int(np.argmax(profile))
    hm = max_val / 2.0

    left: Optional[float] = None
    for i in range(max_idx - 1, -1, -1):
        if profile[i] < hm <= profile[i + 1]:
            frac = (hm - profile[i]) / (profile[i + 1] - profile[i])
            left = float(x_mm[i] + frac * (x_mm[i + 1] - x_mm[i]))
            break

    right: Optional[float] = None
    for i in range(max_idx, len(profile) - 1):
        if profile[i] >= hm > profile[i + 1]:
            frac = (hm - profile[i]) / (profile[i + 1] - profile[i])
            right = float(x_mm[i] + frac * (x_mm[i + 1] - x_mm[i]))
            break

    return left, right, hm


def find_fwhm_dense(
    xd: np.ndarray, yd: np.ndarray
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Find FWHM from a dense profile using threshold crossings.

    Returns
    -------
    (fwhm, x_left, x_right, half_max) — all ``None`` when no crossing found.
    """
    hm = float(np.max(yd)) / 2.0
    above = yd >= hm
    tr = np.diff(above.astype(int))
    rise = np.where(tr == 1)[0]
    fall = np.where(tr == -1)[0]
    if len(rise) > 0 and len(fall) > 0:
        xl = float(xd[rise[0]])
        xr = float(xd[fall[-1]])
        return xr - xl, xl, xr, hm
    return None, None, None, None


def find_distal_pct(profile: np.ndarray, x_mm: np.ndarray,
                    pct: float) -> Tuple[Optional[float], Optional[float]]:
    """Find the distal position where the profile drops to *pct* % of max.

    Parameters
    ----------
    profile, x_mm : 1-D arrays.
    pct : target percentage (0-100 scale, e.g. 90 for 90 %).

    Returns
    -------
    (position, value) or ``(None, None)`` when not found.
    """
    profile = np.asarray(profile, dtype=float)
    x_mm = np.asarray(x_mm, dtype=float)

    max_val = float(np.max(profile))
    max_idx = int(np.argmax(profile))
    target = max_val * pct / 100.0

    for i in range(max_idx, len(profile) - 1):
        if profile[i] >= target > profile[i + 1]:
            frac = (target - profile[i]) / (profile[i + 1] - profile[i])
            pos = float(x_mm[i] + frac * (x_mm[i + 1] - x_mm[i]))
            return pos, float(target)
    return None, None


def find_distal_pct_dense(xd: np.ndarray, yd: np.ndarray,
                          pct_fraction: float) -> Optional[float]:
    """Find distal position from a dense profile (*pct_fraction* in 0–1 scale).

    Returns
    -------
    Position (float) or ``None``.
    """
    thr = pct_fraction * float(np.max(yd))
    mx = int(np.argmax(yd))
    distal_mask = np.arange(len(xd)) > mx
    below = yd[distal_mask] < thr
    if np.any(below):
        return float(xd[distal_mask][np.where(below)[0][0]])
    return None


def falloff_80_20(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Compute the 80 %–20 % distal falloff width (using dense interpolation)."""
    xd, yd = get_dense(x, y)
    x80 = find_distal_pct_dense(xd, yd, 0.8)
    x20 = find_distal_pct_dense(xd, yd, 0.2)
    if x80 is not None and x20 is not None:
        return x20 - x80
    return None


def calculate_profile_metrics(profile: np.ndarray, x_mm: np.ndarray) -> Dict:
    """Compute a standard set of profile metrics.

    Returns
    -------
    dict with keys: fwhm, pos_90, pos_80, pos_20, pos_10, dist_90_10, dist_80_20.
    """
    left, right, _ = find_fwhm(profile, x_mm)
    fwhm = (right -
            left) if (left is not None and right is not None) else float('nan')

    pos_90, _ = find_distal_pct(profile, x_mm, 90.0)
    pos_80, _ = find_distal_pct(profile, x_mm, 80.0)
    pos_20, _ = find_distal_pct(profile, x_mm, 20.0)
    pos_10, _ = find_distal_pct(profile, x_mm, 10.0)

    def _safe_diff(a, b):
        if a is not None and b is not None:
            return b - a
        return float('nan')

    return {
        'fwhm': fwhm,
        'pos_90': pos_90,
        'pos_80': pos_80,
        'pos_20': pos_20,
        'pos_10': pos_10,
        'dist_90_10': _safe_diff(pos_90, pos_10),
        'dist_80_20': _safe_diff(pos_80, pos_20),
    }


def all_metrics_dense(x: np.ndarray, y: np.ndarray) -> Dict:
    """Calculate metrics using dense interpolation (for comparison modules).

    Returns
    -------
    dict with keys: fwhm, x90, penumbra_80_20 (alias falloff).
    """
    xd, yd = get_dense(x, y)
    fw, *_ = find_fwhm_dense(xd, yd)
    x90 = find_distal_pct_dense(xd, yd, 0.9)
    p8020 = falloff_80_20(x, y)
    return {'fwhm': fw, 'x90': x90, 'penumbra_80_20': p8020, 'falloff': p8020}


def align_mbo_to_reference(
        mbo_x: np.ndarray,
        mbo_y: np.ndarray,
        ref_max_x: float,
        ref_interp: interp1d,
        match_x: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Shift MBO profile to align its max with *ref_max_x* and scale to match
    the reference at *match_x*.

    Returns
    -------
    (x_shifted, y_scaled)
    """
    mx_idx = int(np.argmax(mbo_y))
    x_shifted = mbo_x - mbo_x[mx_idx] + ref_max_x
    f = interp1d(x_shifted, mbo_y, kind='linear', fill_value='extrapolate')
    scale = float(ref_interp(match_x)) / float(f(match_x))
    return x_shifted, mbo_y * scale


def get_value_at_x(x_arr: np.ndarray, profile: np.ndarray,
                   target_x: float) -> float:
    """Linearly interpolate a profile value at *target_x*."""
    idx = int(np.abs(x_arr - target_x).argmin())
    if 0 < idx < len(x_arr) - 1:
        if x_arr[idx] > target_x:
            x0, x1 = x_arr[idx - 1], x_arr[idx]
            y0, y1 = profile[idx - 1], profile[idx]
        else:
            x0, x1 = x_arr[idx], x_arr[idx + 1]
            y0, y1 = profile[idx], profile[idx + 1]
        frac = (target_x - x0) / (x1 - x0) if x1 != x0 else 0.0
        return float(y0 + frac * (y1 - y0))
    return float(profile[idx])
