"""Foil image matching and merging utilities."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi

from src.data.analysis import Circle


@dataclass
class EdgeLine:
    """Line fitted to edge points: y = slope * x + intercept."""
    slope: float
    intercept: float
    slope_deg: float
    points: List[Tuple[float, float]]  # (x, y) pairs used for fitting


def find_edge_crossings(
    image: np.ndarray,
    x_positions: List[int],
    threshold: float,
    stripe_width: int = 10,
    search_from_bottom: bool = True
) -> List[Tuple[int, Optional[float]]]:
    """Find vertical edge crossings at specified x positions.
    
    Args:
        image: Input image array
        x_positions: X coordinates to sample
        threshold: Intensity threshold for edge detection
        stripe_width: Width of vertical stripe to average
        search_from_bottom: If True, search from bottom to top
        
    Returns:
        List of (x, y_crossing) tuples. y_crossing is None if not found.
    """
    h, w = image.shape
    half = stripe_width // 2
    results = []
    
    for x in x_positions:
        x0 = max(0, x - half)
        x1 = min(w, x + half + 1)
        profile = np.nanmean(image[:, x0:x1].astype(float), axis=1)
        
        # Find threshold crossing with interpolation
        above = profile >= threshold
        if not np.any(above):
            results.append((x, None))
            continue
            
        indices = np.where(above)[0]
        if search_from_bottom:
            idx = int(indices.max())
        else:
            idx = int(indices.min())
            
        # Interpolate crossing position
        if idx == 0 or idx == len(profile) - 1:
            results.append((x, float(idx)))
        else:
            prev = idx - 1 if search_from_bottom else idx + 1
            v_low, v_high = float(profile[prev]), float(profile[idx])
            if v_high != v_low:
                frac = (threshold - v_low) / (v_high - v_low)
                y_cross = prev + frac if search_from_bottom else idx - frac
            else:
                y_cross = float(idx)
            results.append((x, y_cross))
    
    return results


def fit_edge_line(crossings: List[Tuple[int, Optional[float]]], robust: bool = True) -> Optional[EdgeLine]:
    """Fit a line through edge crossing points.
    
    Args:
        crossings: List of (x, y_crossing) tuples from find_edge_crossings
        robust: If True, use RANSAC for outlier rejection
        
    Returns:
        EdgeLine object or None if insufficient valid points
    """
    valid = [(float(x), y) for x, y in crossings if y is not None]
    if len(valid) < 2:
        return None
    
    xs = np.array([p[0] for p in valid])
    ys = np.array([p[1] for p in valid])
    
    if robust:
        try:
            from sklearn.linear_model import RANSACRegressor, LinearRegression
            model = RANSACRegressor(LinearRegression(), min_samples=max(2, len(xs) // 2))
            model.fit(xs.reshape(-1, 1), ys)
            slope = float(model.estimator_.coef_[0])
            intercept = float(model.estimator_.intercept_)
            mask = model.inlier_mask_
            points = [(xs[i], ys[i]) for i in range(len(xs)) if mask[i]]
        except Exception:
            robust = False
    
    if not robust:
        p = np.polyfit(xs, ys, 1)
        slope, intercept = float(p[0]), float(p[1])
        points = list(zip(xs.tolist(), ys.tolist()))
    
    return EdgeLine(
        slope=slope,
        intercept=intercept,
        slope_deg=float(np.degrees(np.arctan(slope))),
        points=points
    )


def rotate_image(image: np.ndarray, angle_deg: float, cval: float = 0.0) -> np.ndarray:
    """Rotate image by specified angle."""
    return ndi.rotate(image, angle_deg, reshape=False, order=3, cval=cval)


def rotate_point(point: Tuple[float, float], img_shape: tuple, angle_deg: float) -> Tuple[float, float]:
    """Rotate a point around image center."""
    h, w = img_shape
    cx, cy = w / 2.0, h / 2.0
    x, y = point[0] - cx, point[1] - cy
    
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    
    x_rot = cos_a * x - sin_a * y + cx
    y_rot = sin_a * x + cos_a * y + cy
    return (x_rot, y_rot)


def rotate_circle(circle: Circle, img_shape: tuple, angle_deg: float) -> Circle:
    """Rotate circle center around image center."""
    x_new, y_new = rotate_point((circle.x, circle.y), img_shape, angle_deg)
    return Circle(x=x_new, y=y_new, r=circle.r)


def rotate_edge_line(line: EdgeLine, img_shape: tuple, angle_deg: float) -> EdgeLine:
    """Rotate edge line points around image center."""
    rotated_points = [rotate_point(p, img_shape, angle_deg) for p in line.points]
    
    # Refit line through rotated points
    if len(rotated_points) >= 2:
        xs = np.array([p[0] for p in rotated_points])
        ys = np.array([p[1] for p in rotated_points])
        p = np.polyfit(xs, ys, 1)
        new_slope = float(p[0])
        new_intercept = float(p[1])
    else:
        new_slope, new_intercept = line.slope, line.intercept
    
    return EdgeLine(
        slope=new_slope,
        intercept=new_intercept,
        slope_deg=float(np.degrees(np.arctan(new_slope))),
        points=rotated_points
    )


def transform_edge_line_for_crop(line: EdgeLine, offset: Tuple[int, int]) -> EdgeLine:
    """Transform edge line points after cropping (subtract offset)."""
    x_off, y_off = offset
    transformed_points = [(x - x_off, y - y_off) for x, y in line.points]
    
    # Recalculate intercept for new coordinate system
    new_intercept = line.intercept - line.slope * x_off - y_off
    
    return EdgeLine(
        slope=line.slope,
        intercept=new_intercept,
        slope_deg=line.slope_deg,
        points=transformed_points
    )


def crop_centered(image: np.ndarray, center_x: float, center_y: float, size: int = 300) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop image centered on specified point.
    
    Returns:
        Tuple of (cropped_image, (x_offset, y_offset))
    """
    half = size // 2
    x_start = int(center_x - half)
    y_start = int(center_y - half)
    
    cropped = image[y_start:y_start + size, x_start:x_start + size]
    return cropped, (x_start, y_start)


def merge_images(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_edge_x: float,
    right_edge_x: float,
    margin: float = 0.0
) -> Tuple[np.ndarray, float]:
    """Merge two images at their edge positions.
    
    Args:
        left_img: Left image (will use left part up to edge)
        right_img: Right image (will use right part from edge)
        left_edge_x: X position of edge in left image
        right_edge_x: X position of edge in right image
        margin: Adjustment to edge positions (negative = overlap, positive = gap)
        
    Returns:
        Tuple of (merged_image, x_offset) where x_offset is shift applied to right coordinates
    """
    left_crop_x = int(left_edge_x + margin)
    right_crop_x = int(right_edge_x - margin)
    
    left_part = left_img[:, :left_crop_x]
    right_part = right_img[:, right_crop_x:]
    
    merged = np.concatenate([left_part, right_part], axis=1)
    x_offset = left_crop_x - right_crop_x
    
    return merged, x_offset


def transform_circle_for_merge(circle: Circle, x_offset: float) -> Circle:
    """Transform circle coordinates after merge."""
    return Circle(x=circle.x + x_offset, y=circle.y, r=circle.r)
