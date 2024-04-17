import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from src.data.analysis import Circle, median_filter, subtract_background, create_circular_mask


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
    print(lower_threshold, upper_threshold)

    output[(data > lower_threshold) & (data < upper_threshold)] = 255
    output[np.isnan(data)] = 255
    logging.info(
        'After setting threshold ' +
        f'min {np.min(output)}, mean {np.mean(output):3.3f}, max {np.max(output)}'
    )
    return output


def find_circle_hough_method(data: npt.NDArray) -> Circle:
    logging.info(
        f'Detector circle not provided, calculating with Hough method')
    hough_results = cv2.HoughCircles(data,
                                     cv2.HOUGH_GRADIENT,
                                     dp=1,
                                     minDist=10000,
                                     param1=10,
                                     param2=0.9,
                                     minRadius=100,
                                     maxRadius=200)
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
