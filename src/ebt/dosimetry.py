"""Dosimetry calculations for EBT films."""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

from .calibration import Calibration, ebt3_proton_calib_20Gy


# Default background values for unexposed EBT3 film (16-bit scanner values)
DEFAULT_BG_R = 42804.451
DEFAULT_BG_G = 44273.485
DEFAULT_BG_B = 27929.477


def net_optical_density(
    image: npt.NDArray,
    channel_no: int = 0,
    bg_values: Optional[Tuple[float, float, float]] = None
) -> npt.NDArray:
    """Calculate net optical density from raw RGB image.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array with shape (height, width, 3), typically uint16
    channel_no : int
        Color channel to use (0=R, 1=G, 2=B). Default is 0 (red channel).
    bg_values : tuple, optional
        Background values (R, G, B) for unexposed film. If None, uses defaults.
    
    Returns
    -------
    np.ndarray
        Net optical density array with shape (height, width)
        
    Notes
    -----
    Net optical density is calculated as: netOD = log10(I0 / I)
    where I0 is the background intensity (unexposed film) and I is the
    measured intensity.
    """
    if bg_values is None:
        bg_values = (DEFAULT_BG_R, DEFAULT_BG_G, DEFAULT_BG_B)
    
    bg = bg_values[channel_no]
    
    # Avoid division by zero and log of non-positive numbers
    pixel_values = np.clip(image[:, :, channel_no], 1, None)
    
    return np.log10(bg / pixel_values)


def ebt3_dose_Gy(
    image: npt.NDArray,
    calib: Optional[Calibration] = None,
    channel_no: int = 0,
    bg_values: Optional[Tuple[float, float, float]] = None
) -> npt.NDArray:
    """Convert raw RGB image to dose in Gy using EBT3 calibration.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array with shape (height, width, 3)
    calib : Calibration, optional
        Calibration object. If None, uses default proton calibration (20 Gy range).
    channel_no : int
        Color channel to use (0=R, 1=G, 2=B). Default is 0 (red channel).
    bg_values : tuple, optional
        Background values (R, G, B) for unexposed film. If None, uses defaults.
    
    Returns
    -------
    np.ndarray
        Dose array in Gy with shape (height, width)
        
    Examples
    --------
    >>> import tifffile
    >>> image = tifffile.imread('ebt_scan.tif')
    >>> dose = ebt3_dose_Gy(image)
    """
    if calib is None:
        calib = ebt3_proton_calib_20Gy
        
    netOD = net_optical_density(image, channel_no, bg_values)
    return calib(netOD)
