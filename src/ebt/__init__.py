"""EBT film analysis package for dose measurements."""

from .calibration import Calibration, ebt3_proton_calib_20Gy
from .dosimetry import net_optical_density, ebt3_dose_Gy
from .io import get_tiff_dpi, get_px_to_mm, FileData, load_tiff_to_filedata, FilePairData, get_red_channel
from .processing import auto_crop_dose, rotate_dose

__all__ = [
    'Calibration',
    'ebt3_proton_calib_20Gy',
    'net_optical_density',
    'ebt3_dose_Gy',
    'get_tiff_dpi',
    'get_px_to_mm',
    'FileData',
    'FilePairData',
    'load_tiff_to_filedata',
    'get_red_channel',
    'auto_crop_dose',
    'rotate_dose',
]
