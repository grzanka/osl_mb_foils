"""File I/O utilities for EBT film analysis."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union, List, Tuple

import numpy as np
import numpy.typing as npt
import tifffile


DEFAULT_DPI = 150.0
"""Default DPI fallback value if not found in TIFF metadata."""


def get_tiff_dpi(filepath: Union[Path, str]) -> float:
    """Extract DPI from TIFF file metadata.
    
    Parameters
    ----------
    filepath : Path or str
        Path to TIFF file
        
    Returns
    -------
    float
        DPI value from file metadata. Returns DEFAULT_DPI (150) if not found.
        
    Notes
    -----
    Reads the XResolution tag from TIFF metadata. Assumes square pixels
    (XResolution equals YResolution).
    """
    filepath = Path(filepath)
    try:
        with tifffile.TiffFile(filepath) as tif:
            page = tif.pages[0]
            x_res = page.tags.get('XResolution')
            if x_res:
                dpi = x_res.value[0] / x_res.value[1]
                return dpi
    except Exception as e:
        print(f"Warning: Could not read DPI from {filepath}: {e}")
    return DEFAULT_DPI


def get_px_to_mm(dpi: float) -> float:
    """Convert DPI to pixel-to-mm conversion factor.
    
    Parameters
    ----------
    dpi : float
        Dots per inch
        
    Returns
    -------
    float
        Conversion factor (mm per pixel)
        
    Notes
    -----
    Uses the conversion: 1 inch = 25.4 mm
    """
    return 25.4 / dpi


@dataclass
class FileData:
    """Container for file-specific data including metadata and dose information.
    
    Parameters
    ----------
    stem : str
        File stem (name without extension)
    path : Path
        Full path to the file
    dpi : float
        DPI resolution of the scan
    px_to_mm : float
        Pixel to mm conversion factor
    raw_image : np.ndarray
        Original RGB image array
    dose_full : np.ndarray
        Full dose distribution array
    shape : tuple
        Shape of the raw image
    dose_cropped : np.ndarray, optional
        Cropped and processed dose array
    crop_bbox : tuple, optional
        Bounding box of the cropped region (y_min, y_max, x_min, x_max)
    """
    stem: str
    path: Path
    dpi: float
    px_to_mm: float
    raw_image: npt.NDArray
    dose_full: npt.NDArray
    shape: tuple
    dose_cropped: Optional[npt.NDArray] = field(default=None)
    crop_bbox: Optional[tuple] = field(default=None)
    
    def __repr__(self) -> str:
        return f"FileData(stem='{self.stem}', dpi={self.dpi:.1f}, px_to_mm={self.px_to_mm:.6f})"
    
    @property
    def width_mm(self) -> float:
        """Width of the image in mm."""
        return self.shape[1] * self.px_to_mm
    
    @property
    def height_mm(self) -> float:
        """Height of the image in mm."""
        return self.shape[0] * self.px_to_mm


def load_tiff_to_filedata(
    filepath: Union[Path, str],
    dose_converter: Optional[Callable] = None
) -> FileData:
    """Load a TIFF file and create a FileData object.
    
    Parameters
    ----------
    filepath : Path or str
        Path to the TIFF file
    dose_converter : callable, optional
        Function to convert raw image to dose. If None, dose_full will be None.
        
    Returns
    -------
    FileData
        FileData object with loaded image and metadata
        
    Examples
    --------
    >>> from src.ebt import ebt3_dose_Gy, load_tiff_to_filedata
    >>> file_data = load_tiff_to_filedata('scan.tif', dose_converter=ebt3_dose_Gy)
    """
    filepath = Path(filepath)
    
    dpi = get_tiff_dpi(filepath)
    px_to_mm = get_px_to_mm(dpi)
    
    raw_image = tifffile.imread(filepath)
    dose_full = dose_converter(raw_image) if dose_converter else None
    
    return FileData(
        stem=filepath.stem,
        path=filepath,
        dpi=dpi,
        px_to_mm=px_to_mm,
        raw_image=raw_image,
        dose_full=dose_full,
        shape=raw_image.shape
    )


@dataclass
class FilePairData:
    """Container for signal-background file pair data.
    
    Parameters
    ----------
    signal_path : Path
        Path to the signal (dose) TIFF file
    background_path : Path
        Path to the background TIFF file
    dpi : float
        DPI resolution of the scan
    px_to_mm : float
        Pixel to mm conversion factor
    signal_raw : np.ndarray
        Raw RGB image array from signal file
    background_raw : np.ndarray
        Raw RGB image array from background file
    dose_full : np.ndarray
        Full dose distribution array (computed from signal)
    background_dose_full : np.ndarray
        Dose computed from background (should be ~0)
    shape : tuple
        Shape of the raw images
    signal_cropped : np.ndarray, optional
        Cropped signal raw image (red channel)
    background_cropped : np.ndarray, optional
        Cropped background raw image (red channel)
    dose_cropped : np.ndarray, optional
        Cropped dose array
    background_dose_cropped : np.ndarray, optional
        Cropped background dose array
    crop_bbox : tuple, optional
        Bounding box (y_min, y_max, x_min, x_max)
    """
    signal_path: Path
    background_path: Path
    dpi: float
    px_to_mm: float
    signal_raw: npt.NDArray
    background_raw: npt.NDArray
    dose_full: npt.NDArray
    background_dose_full: npt.NDArray
    shape: tuple
    signal_cropped: Optional[npt.NDArray] = field(default=None)
    background_cropped: Optional[npt.NDArray] = field(default=None)
    dose_cropped: Optional[npt.NDArray] = field(default=None)
    background_dose_cropped: Optional[npt.NDArray] = field(default=None)
    crop_bbox: Optional[tuple] = field(default=None)
    
    def __repr__(self) -> str:
        return f"FilePairData(signal='{self.signal_path.name}', bg='{self.background_path.name}', dpi={self.dpi:.1f})"
    
    @property
    def stem(self) -> str:
        """Signal file stem."""
        return self.signal_path.stem
    
    @property
    def width_mm(self) -> float:
        """Width of the image in mm."""
        return self.shape[1] * self.px_to_mm
    
    @property
    def height_mm(self) -> float:
        """Height of the image in mm."""
        return self.shape[0] * self.px_to_mm


def get_red_channel(image: npt.NDArray) -> npt.NDArray:
    """Extract red channel from RGB image.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image with shape (height, width, 3)
        
    Returns
    -------
    np.ndarray
        Red channel with shape (height, width)
    """
    return image[:, :, 0].astype(np.float64)
