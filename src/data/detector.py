from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

import cv2
import numpy as np
import numpy.typing as npt

from src.data.analysis import Circle, read_tiff_img, create_circular_mask


def img_for_circle_detection(data: npt.NDArray,
                             r: float = 100,
                             nsigma: float = 2.,
                             old_method: bool = False) -> npt.NDArray:
    logging.info(
        'Before setting threshold ' +
        f'min {np.nanmin(data)}, mean {np.nanmean(data):3.3f}, max {np.nanmax(data)}'
    )
    output = np.zeros(shape=data.shape, dtype='uint8')

    circle_inside_det = Circle(x=data.shape[0] / 2, y=data.shape[0] / 2, r=r)
    mask_inside_det = create_circular_mask(data, circle_inside_det)
    upper_threshold = np.mean(
        data,
        where=mask_inside_det) + nsigma * np.std(data, where=mask_inside_det)
    lower_threshold = np.mean(
        data,
        where=mask_inside_det) - nsigma * np.std(data, where=mask_inside_det)

    if old_method:
        lower_threshold = np.percentile(data, 95)
        upper_threshold = np.max(data)

    output[(data > lower_threshold) & (data < upper_threshold)] = 255
    output[np.isnan(data)] = 255
    logging.info(
        'After setting threshold ' +
        f'min {np.min(output)}, mean {np.mean(output):3.3f}, max {np.max(output)}'
    )
    return output


def find_circle_hough_method(data: npt.NDArray,
                             min_radius: float = 100,
                             max_radius: float = 200) -> Circle:
    logging.info(
        f'Detector circle not provided, calculating with Hough method')
    hough_results = cv2.HoughCircles(data,
                                     cv2.HOUGH_GRADIENT,
                                     dp=1,
                                     minDist=10000,
                                     param1=10,
                                     param2=0.9,
                                     minRadius=min_radius,
                                     maxRadius=max_radius)
    logging.info(f'Hough method results {hough_results}')
    result_circle = Circle()
    if hough_results is None:
        print("No detector found by Hough method")
    elif hough_results.shape[0] > 1:
        print("More than one shape found by Hough method")
    elif hough_results.shape[0] == 1 and hough_results.shape[1] == 1:
        # explicit conversion to float is needed to ensure proper JSON serialisation
        # hough_results is a numpy float32 array and float32 is not JSON serialisable
        result_circle = Circle(
            x=float(hough_results[0, 0, 0]),
            y=float(hough_results[0, 0, 1]),
            r=float(hough_results[0, 0, 2]),
        )
        logging.info(f'Detected circle {result_circle}')
    return result_circle


@dataclass(frozen=True)
class DetectorImage:
    image: np.ndarray
    path: Path

    @property
    def init_circle(self) -> Circle:
        return Circle(x=self.image.shape[0] // 2,
                      y=self.image.shape[1] // 2,
                      r=40)


@dataclass(frozen=True)
class DetectorData:
    raw: DetectorImage
    lv: DetectorImage
    det_no: int
    circle: Circle = field(default=Circle())


@dataclass(frozen=True)
class DetectorDataCollection:
    path: Path
    data: dict[int, DetectorData] = field(default_factory=dict)

    def __post_init__(self):
        if not self.data:
            self._load_data()

    def _load_data(self):
        for file_path in sorted(self.path.iterdir()):
            if file_path.name.endswith('lv'):
                # get detector data
                try:
                    det_id = re.findall(r'\d+', file_path.name)[0]
                except IndexError:
                    continue
                det_no = int(det_id)
                # live view images
                lv_path = next(file_path.glob('**/*tif'))
                lv_data = read_tiff_img(lv_path, border_px=0)
                lv_image = DetectorImage(image=lv_data, path=lv_path)
                # raw data images
                try:
                    raw_path = next((self.path / det_id).glob('**/*tif'))
                    raw_data = read_tiff_img(raw_path, border_px=0)
                    raw_image = DetectorImage(image=raw_data, path=raw_path)
                    det_data = DetectorData(raw=raw_image,
                                            lv=lv_image,
                                            det_no=det_no)
                    self.data[det_no] = det_data
                    print(f"{det_no} ", end='')
                except StopIteration:
                    print(f"missing_{det_no} ", end='')
