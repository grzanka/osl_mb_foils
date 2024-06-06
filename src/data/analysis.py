import dataclasses
import datetime
import json
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import ndimage
import pandas as pd


@dataclass(frozen=True)
class Circle:
    '''Read-only data class for storing center position and radius of a circle'''
    x: float = float('nan')
    y: float = float('nan')
    r: float = float('nan')

    @property
    def proper(self):
        return self.x > -np.inf and self.y > -np.inf and self.r >= 0.

    def save_json(self, path):
        with open(path, 'w') as json_file:
            json.dump(dataclasses.asdict(self), json_file)

    @classmethod
    def from_json(cls, path: str):
        circle = cls()
        with open(path, 'r') as openfile:
            json_data = json.load(openfile)
            circle = cls(**json_data)
        return circle

    def section_x(self, x: float) -> Tuple[float, float]:
        under_sqrt = self.r**2 - (x - self.x)**2
        result = (float('nan'), float('nan'))
        if under_sqrt >= 0:
            result = (self.y - math.sqrt(under_sqrt),
                      self.y + math.sqrt(under_sqrt))
        return result

    def section_y(self, y: float) -> Tuple[float, float]:
        under_sqrt = self.r**2 - (y - self.y)**2
        result = (float('nan'), float('nan'))
        if under_sqrt >= 0:
            result = (self.x - math.sqrt(under_sqrt),
                      self.x + math.sqrt(under_sqrt))
        return result


def read_tiff_img(file_path: Path, border_px: int = 0) -> npt.NDArray:
    '''
    read tif image and add a border filled with NaN
    new image will have bigger size (by 2 x border_px in each direction) than original tiff
    Note: input file contains 16-bit integers, but we intentionally cast the output to the 32-bit float array
          the reason is that we want to pad the array with NaNs which are transpartent to methods like min/max/mean
          there is no NaN for integers, therefore we are forced to use floats despite int-like values
    '''
    logging.info(f'Reading file {file_path}')
    raw_img = plt.imread(file_path)
    logging.info(
        f'Original image shape: {raw_img.shape}, min value {raw_img.min()}, max value {raw_img.max()}'
    )
    if border_px > 0:
        raw_img = raw_img.astype('float32')
        raw_img = np.pad(raw_img, pad_width=border_px, constant_values=np.nan)
        logging.info(f'Padded image shape: {raw_img.shape}')
    return raw_img


def create_circular_mask(img: npt.NDArray, circle_px: Circle) -> npt.NDArray:
    '''create a circular mask of the same resolution as the image.'''

    y_grid, x_grid = np.ogrid[:img.shape[0], :img.shape[1]]
    # x_grid has shape (img.shape[0], 1)
    # y_grid has shape (1, img.shape[1])

    dist_from_center_squared = (x_grid - circle_px.x)**2 + (y_grid -
                                                            circle_px.y)**2
    # broadcasting will guarantee that the formula above will give us shape (img.shape[0], img.shape[1])

    circ_mask = dist_from_center_squared <= circle_px.r**2
    return circ_mask


def default_circular_mask(img: npt.NDArray) -> npt.NDArray:
    '''
    create a circular mask with circle in the middle of the image
    and maximum possible radius to be fully enclosed in the image
    '''
    center_x = img.shape[0] / 2
    center_y = img.shape[1] / 2
    radius = min(center_x, center_y)

    return create_circular_mask(img, Circle(x=center_x, y=center_y, r=radius))


def median_filter(input: npt.NDArray,
                  size: int = 10,
                  gpu: bool = False) -> npt.NDArray:
    '''apply median filter'''
    logging.info(
        'Before median filter ' +
        f'min {np.nanmin(input)}, mean max {np.nanmean(input):3.3f}, max {np.nanmax(input)}'
    )
    output = np.empty(shape=1)
    if gpu:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import median_filter as median_filter_gpu
            output = median_filter_gpu(cp.asarray(input), size=size).get()
        except (ModuleNotFoundError, ImportError):
            logging.warning(
                'GPU mode selected and no `cupy` library installed')
            return np.full_like(input, np.nan, dtype=np.double)
    else:
        output = ndimage.median_filter(input, size=size)
    logging.info(
        'After median filter ' +
        f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}'
    )
    return output


def subtract_background(input: npt.NDArray,
                        img_bg: Optional[npt.NDArray] = None,
                        const_bg: float = 0,
                        gpu: bool = False) -> npt.NDArray:
    '''Background remove (constant BG (CBG) and imgBG)
    # assume zero background if no `img_bg` option provided'''

    output = input.copy()

    if img_bg is not None:
        output -= img_bg
    logging.info(
        'After background image subtraction ' +
        f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}'
    )

    # subtract constant background factor
    output -= const_bg
    logging.info(
        'After constant background factor subtraction ' +
        f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}'
    )

    # set all pixels with negative values to zero
    # optionally use: `np.clip(a=img_det, a_min=0., out=img_det)`
    output[output < 0] = 0
    logging.info(
        'After removing pixels with negative value ' +
        f'min {np.nanmin(output)}, mean max {np.nanmean(output):3.3f}, max {np.nanmax(output)}'
    )

    return output


def get_mean_std(data: npt.NDArray, circle: Circle) -> Tuple[float, float]:
    mask = create_circular_mask(img=data, circle_px=circle)
    mean = np.nanmean(data[mask]).astype(float)
    std = np.nanstd(data[mask]).astype(float)
    return mean, std


def get_timestamp(filepath: str) -> datetime.datetime:
    metada_contents = ''
    with open(filepath, 'r', encoding="ISO-8859-1") as metadata_file:
        metada_contents = metadata_file.read()

    parsed_json = json.loads(metada_contents)
    time_str = parsed_json['Summary']['Time']  # or 'StartTime'
    result = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S %z')
    return result


def perc_error_to_ref_df(df: pd.DataFrame,
                         xcolname: str,
                         ycolname: str,
                         ref_df: pd.DataFrame,
                         ref_xcolname: str,
                         ref_ycolname: str,
                         threshold: float = 0.01) -> pd.DataFrame:
    filter = df[ycolname] > threshold
    rel_error = df[ycolname][filter] / np.interp(x=df[xcolname][filter],
                                                 xp=ref_df[ref_xcolname],
                                                 fp=ref_df[ref_ycolname])
    df = pd.DataFrame({
        'z': df[xcolname][filter],
        'perc_err': 100 * (rel_error - 1.0)
    })
    return df
