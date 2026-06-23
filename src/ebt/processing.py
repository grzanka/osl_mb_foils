"""Image processing utilities for EBT film analysis."""

from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def auto_crop_dose(
    dose_array: npt.NDArray,
    px_to_mm: float,
    dose_threshold_gy: float = 0.5,
    smooth_sigma_mm: float = 1.0,
    min_diameter_mm: float = 10.0,
    margin_mm: float = 5.0
) -> Tuple[npt.NDArray, Tuple[int, int, int, int]]:
    """Crop dose array to the dominant irradiated region.

    The algorithm:
    1. Smooth dose to suppress noise.
    2. Threshold at dose_threshold_gy to find irradiated pixels.
    3. Keep the largest connected component that is big enough to hold
       a circle with min_diameter_mm diameter.
    4. Expand bounding box by margin_mm on each side.

    Parameters
    ----------
    dose_array : np.ndarray
        2D dose array
    px_to_mm : float
        Pixel to mm conversion factor
    dose_threshold_gy : float
        Dose threshold for masking (in Gy). Default 0.5 Gy.
    smooth_sigma_mm : float
        Smoothing sigma in mm for Gaussian filter. Default 1.0 mm.
    min_diameter_mm : float
        Minimum diameter of the irradiated region in mm. Components smaller
        than a circle with this diameter are ignored. Default 10.0 mm.
    margin_mm : float
        Margin to add around the detected region in mm. Default 5.0 mm.

    Returns
    -------
    cropped_array : np.ndarray
        Cropped dose array
    bbox : tuple
        Bounding box coordinates (y_min, y_max, x_min, x_max)
        
    Examples
    --------
    >>> dose_cropped, bbox = auto_crop_dose(dose_full, px_to_mm=0.17)
    >>> print(f"Cropped to Y[{bbox[0]}:{bbox[1]}], X[{bbox[2]}:{bbox[3]}]")
    """
    # Ensure non-negative dose for masking
    dose_clipped = np.clip(dose_array, 0, None)

    # Convert mm-based parameters to pixels
    sigma_px = max(0.5, smooth_sigma_mm / px_to_mm)
    margin_px = int(np.ceil(margin_mm / px_to_mm))
    min_radius_px = (min_diameter_mm / px_to_mm) / 2.0
    min_area_px = np.pi * (min_radius_px ** 2)

    # Smooth then threshold
    smoothed = gaussian_filter(dose_clipped, sigma=sigma_px)
    mask = smoothed > dose_threshold_gy

    # Handle case where no pixels exceed threshold
    if not mask.any():
        return dose_array, (0, dose_array.shape[0], 0, dose_array.shape[1])

    # Connected component analysis
    labels, num = ndimage.label(mask)
    if num == 0:
        return dose_array, (0, dose_array.shape[0], 0, dose_array.shape[1])

    # Find components large enough to contain the minimum diameter circle
    sizes = ndimage.sum(mask, labels, index=range(1, num + 1))
    valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_area_px]

    # Pick largest valid component; if none valid, pick absolute largest
    if valid_labels:
        target_label = max(valid_labels, key=lambda lbl: sizes[lbl - 1])
    else:
        target_label = int(np.argmax(sizes)) + 1

    # Get bounding box of target component
    y_idx, x_idx = np.where(labels == target_label)
    if len(y_idx) == 0:
        return dose_array, (0, dose_array.shape[0], 0, dose_array.shape[1])

    # Add margin and clip to array bounds
    y_min = max(0, y_idx.min() - margin_px)
    y_max = min(dose_array.shape[0], y_idx.max() + margin_px)
    x_min = max(0, x_idx.min() - margin_px)
    x_max = min(dose_array.shape[1], x_idx.max() + margin_px)

    cropped = dose_array[y_min:y_max, x_min:x_max]
    return cropped, (y_min, y_max, x_min, x_max)


def rotate_dose(dose_array: npt.NDArray, k: int = 1) -> npt.NDArray:
    """Rotate dose array by 90 degrees counter-clockwise.
    
    Parameters
    ----------
    dose_array : np.ndarray
        2D dose array
    k : int
        Number of 90-degree rotations. Default 1 (90 degrees CCW).
        
    Returns
    -------
    np.ndarray
        Rotated dose array
    """
    return np.rot90(dose_array, k=k)
