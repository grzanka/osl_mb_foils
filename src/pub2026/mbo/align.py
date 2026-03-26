"""MBO foil alignment: circle detection, edge finding, rotation, cropping.

For each foil loaded from a survey NPZ:
1. Detect the detector circle (Hough on LV, refined via gradient on raw).
2. Find the straight edge via threshold crossings.
3. Rotate so the edge is horizontal at the bottom.
4. Crop to a square centred on the circle.
5. Plot the aligned image and a horizontal profile through the centre.

Falls back to live-view for edge detection when raw fails.
"""

import io
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data.analysis import Circle
from src.data.detector import (
    find_circle_hough_method,
    img_for_circle_detection,
)
from src.data.matching import (
    find_edge_crossings,
    fit_edge_line,
    EdgeLine,
)
from src.pub2026.config import MBOAlignConfig, resolve_file
from src.pub2026.mbo.match import (
    find_gradient_edge_points,
    filter_edge_points_near_circle,
    fit_circle_to_points,
)
from src.pub2026.mbo.survey_raw import _percentile_clim
from src.pub2026.pdf_report import PDFReport

import scipy.ndimage as ndi

# ---------------------------------------------------------------------------
# Data container for one aligned foil
# ---------------------------------------------------------------------------


@dataclass
class AlignedFoil:
    det_id: int
    circle: Circle  # Hough circle in original image coords
    edge_line: Optional[EdgeLine]
    edge_source: str  # "raw", "lv", or "failed"
    rotation_deg: float
    rotated_image: np.ndarray
    rotated_lv: np.ndarray  # rotated live-view image
    rotated_circle: Circle  # Hough circle after rotation
    cropped_image: np.ndarray
    crop_offset: Tuple[int, int]  # (x0, y0) of crop in rotated image
    refined_circle_in_crop: Circle  # gradient-refined circle in crop coords


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _detect_circle_hough(lv_img: np.ndarray, n_sigma: float,
                         probe_radius: int) -> Circle:
    """Detect circle using Hough transform on the live-view image."""
    lv_for_det = img_for_circle_detection(lv_img,
                                          r=probe_radius,
                                          nsigma=n_sigma)
    circle = find_circle_hough_method(lv_for_det)
    if not circle.proper:
        h, w = lv_img.shape
        circle = Circle(x=w / 2, y=h / 2, r=min(h, w) / 3)
    return circle


def _refine_circle_gradient(
    raw_img: np.ndarray,
    hough_circle: Circle,
    gradient_high: float,
    gradient_low: float,
    edge_distance_mm: float,
    radius_tolerance: float,
    pixel_size_mm: float,
) -> Tuple[Circle, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Refine circle on full image by pre-cropping around the Hough centre.

    The gradient radial scan starts from the image centre, so we first crop
    a generous region centred on the Hough circle, run the scan there, then
    map the refined circle back to full-image coordinates.

    Returns (refined_circle_full, all_edge_pts_full, filtered_pts_full)
    where point coordinates are in full-image pixel space.
    """
    # Pre-crop: circle diameter + generous margin so the whole edge is visible
    margin_px = int(round(2.0 / pixel_size_mm))  # 2 mm margin
    crop_half = int(round(hough_circle.r)) + margin_px
    crop_size = 2 * crop_half
    crop, (x0, y0) = _crop_centered(raw_img, hough_circle.x, hough_circle.y,
                                    crop_size)

    # Circle in crop coords
    circle_crop = Circle(x=hough_circle.x - x0,
                         y=hough_circle.y - y0,
                         r=hough_circle.r)

    # Gradient edge detection on crop
    max_dist_px = edge_distance_mm / pixel_size_mm
    all_pts = find_gradient_edge_points(crop, gradient_high, gradient_low)
    filtered_pts = filter_edge_points_near_circle(all_pts, circle_crop,
                                                  max_dist_px)
    refined_crop = fit_circle_to_points(filtered_pts, circle_crop,
                                        radius_tolerance)

    # Map back to full-image coords
    refined_full = Circle(x=refined_crop.x + x0,
                          y=refined_crop.y + y0,
                          r=refined_crop.r)
    all_pts_full = [(px_x + x0, px_y + y0) for px_x, px_y in all_pts]
    filt_pts_full = [(px_x + x0, px_y + y0) for px_x, px_y in filtered_pts]

    return refined_full, all_pts_full, filt_pts_full


def _rotate_image_around(image: np.ndarray, angle_deg: float, cx: float,
                         cy: float) -> np.ndarray:
    """Rotate *image* by *angle_deg* around pixel (cx, cy).

    Positive angle = counter-clockwise (same convention as scipy.ndimage.rotate).
    Uses an affine transform: translate so (cx, cy) is the origin, rotate,
    translate back.
    """
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # scipy.ndimage.affine_transform uses the *inverse* mapping (output→input).
    # To rotate the image CCW by θ, the inverse mapping rotates CW by θ,
    # which in (row, col) = (y, x) space is:
    matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    # offset = center - matrix @ center
    center = np.array([cy, cx])  # (row, col)
    offset = center - matrix @ center

    return ndi.affine_transform(image.astype(np.float64),
                                matrix,
                                offset,
                                order=3,
                                mode='constant',
                                cval=0.0)


def _rotate_circle_around(circle: Circle, angle_deg: float, cx: float,
                          cy: float) -> Circle:
    """Rotate circle centre around (cx, cy) by *angle_deg*."""
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx, dy = circle.x - cx, circle.y - cy
    x_new = cos_a * dx - sin_a * dy + cx
    y_new = sin_a * dx + cos_a * dy + cy
    return Circle(x=x_new, y=y_new, r=circle.r)


def _load_survey_npz(npz_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Load smoothed images from a raw-survey NPZ.

    Returns dict mapping detector_id -> {'raw': array, 'lv': array}.
    """
    data = np.load(npz_path)
    det_ids = data['detector_ids']
    result = {}
    for did in det_ids:
        result[int(did)] = {
            'raw': data[f'foil_{did}_raw'],
            'lv': data[f'foil_{did}_lv'],
        }
    return result


def _find_edge(
    image: np.ndarray, edge_x_px: List[int], threshold: float,
    stripe_width: int
) -> Tuple[List[Tuple[int, Optional[float]]], Optional[EdgeLine]]:
    """Find edge line in an image.

    Returns (crossings, edge_line). edge_line is None on failure.
    """
    crossings = find_edge_crossings(image, edge_x_px, threshold, stripe_width)
    valid_count = sum(1 for _, y in crossings if y is not None)
    if valid_count < 2:
        return crossings, None
    line = fit_edge_line(crossings, robust=True)
    return crossings, line


def _rotation_angle_edge_bottom(edge_line: EdgeLine) -> float:
    """Return rotation angle (degrees) that places the edge at the bottom.

    The edge line slope gives the tilt. We want the edge horizontal at the
    bottom of the image. ``find_edge_crossings`` searches from the bottom, so
    the edge is already near the bottom — we only need to cancel the tilt.
    """
    return edge_line.slope_deg


def _crop_centered(image: np.ndarray, cx: float, cy: float,
                   size_px: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop *size_px × size_px* region centred on (cx, cy)."""
    half = size_px // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    h, w = image.shape

    # Pad with zeros if crop extends outside the image
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, (x0 + size_px) - w)
    pad_bottom = max(0, (y0 + size_px) - h)

    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(w, x0 + size_px)
    y1c = min(h, y0 + size_px)

    region = image[y0c:y1c, x0c:x1c]
    if pad_left or pad_top or pad_right or pad_bottom:
        region = np.pad(region, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant',
                        constant_values=0)
    return region, (x0, y0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_circle_detection(raw_img: np.ndarray, lv_img: np.ndarray,
                           circle: Circle, det_id: int, px: float,
                           clim: Tuple[float, float]) -> plt.Figure:
    """Side-by-side raw + LV with detected circle overlay."""
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, img, label in [(ax_r, raw_img, 'Raw'), (ax_l, lv_img, 'LV')]:
        v0, v1 = _percentile_clim(img, clim[0], clim[1])
        ext = [0, img.shape[1] * px, img.shape[0] * px, 0]
        im = ax.imshow(img, extent=ext, vmin=v0, vmax=v1, cmap='hot')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.add_patch(
            plt.Circle((circle.x * px, circle.y * px),
                       circle.r * px,
                       color='cyan',
                       fill=False,
                       lw=1.5))
        ax.plot(circle.x * px, circle.y * px, 'c+', ms=10, mew=1.5)
        ax.set(xlabel='X [mm]',
               ylabel='Y [mm]',
               title=f'Foil {det_id} — {label}')

    fig.suptitle(
        f'Foil {det_id}: Circle Detection '
        f'(r={circle.r * px:.2f} mm)',
        fontsize=13)
    fig.tight_layout()
    return fig


def _plot_gradient_refinement(
    raw_img: np.ndarray,
    hough_circle: Circle,
    refined_circle: Circle,
    all_pts: List[Tuple[int, int]],
    filtered_pts: List[Tuple[int, int]],
    det_id: int,
    px: float,
    clim: Tuple[float, float],
) -> plt.Figure:
    """Gradient edge detection visualization (notebook style).

    Shows the raw image with:
    - Blue dashed circle: initial Hough circle
    - Red dots: all gradient edge points
    - Green circles: filtered edge points (near initial circle)
    - Cyan solid circle: refined circle after fitting
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    v0, v1 = _percentile_clim(raw_img, clim[0], clim[1])
    ext = [0, raw_img.shape[1] * px, raw_img.shape[0] * px, 0]
    im = ax.imshow(raw_img, extent=ext, vmin=v0, vmax=v1, cmap='hot')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')

    # Initial Hough circle (blue dashed)
    ax.add_patch(
        plt.Circle((hough_circle.x * px, hough_circle.y * px),
                   hough_circle.r * px,
                   color='blue',
                   fill=False,
                   lw=1,
                   ls='--',
                   label=f'Hough (r={hough_circle.r * px:.2f} mm)'))

    # All gradient edge points (red, small)
    if all_pts:
        ax.plot([p[0] * px for p in all_pts], [p[1] * px for p in all_pts],
                'r.',
                ms=1,
                alpha=0.3,
                label=f'All edge pts ({len(all_pts)})')

    # Filtered edge points (green, larger)
    if filtered_pts:
        ax.plot([p[0] * px for p in filtered_pts],
                [p[1] * px for p in filtered_pts],
                'go',
                ms=3,
                alpha=0.7,
                label=f'Filtered pts ({len(filtered_pts)})')

    # Refined circle (cyan solid)
    ax.add_patch(
        plt.Circle((refined_circle.x * px, refined_circle.y * px),
                   refined_circle.r * px,
                   color='cyan',
                   fill=False,
                   lw=1.5,
                   label=f'Refined (r={refined_circle.r * px:.2f} mm)'))
    ax.plot(refined_circle.x * px, refined_circle.y * px, 'c+', ms=10, mew=1.5)

    dx = (refined_circle.x - hough_circle.x) * px
    dy = (refined_circle.y - hough_circle.y) * px
    dr = (refined_circle.r - hough_circle.r) * px
    ax.set(xlabel='X [mm]',
           ylabel='Y [mm]',
           title=f'Foil {det_id} — Gradient Circle Refinement\n'
           f'Δcenter=({dx:+.3f},{dy:+.3f}) mm  Δr={dr:+.3f} mm')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig


def _plot_edge_detection(
        image: np.ndarray,
        crossings: List[Tuple[int, Optional[float]]],
        edge_line: Optional[EdgeLine],
        circle: Circle,
        det_id: int,
        edge_source: str,
        px: float,
        clim: Tuple[float, float],
        edge_x_mm: Optional[List[float]] = None) -> plt.Figure:
    """Image with all crossing points and fitted edge line (match.py style)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    v0, v1 = _percentile_clim(image, clim[0], clim[1])
    ext = [0, image.shape[1] * px, image.shape[0] * px, 0]
    im = ax.imshow(image, extent=ext, vmin=v0, vmax=v1, cmap='hot')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')

    # Circle outline
    ax.add_patch(
        plt.Circle((circle.x * px, circle.y * px),
                   circle.r * px,
                   color='cyan',
                   fill=False,
                   lw=1,
                   ls='--'))

    # Vertical lines at edge search positions
    if edge_x_mm:
        for x_mm in edge_x_mm:
            ax.axvline(x_mm, color='yellow', lw=0.5, ls=':', alpha=0.7)

    # All crossing points (including outliers)
    for pt_x, pt_y in crossings:
        if pt_y is not None:
            ax.plot(pt_x * px, pt_y * px, 'r+', ms=8, mew=2)

    # Fitted line through inlier points
    status = 'FAILED'
    if edge_line is not None and edge_line.points:
        xs = [p[0] * px for p in edge_line.points]
        ys = [p[1] * px for p in edge_line.points]
        ax.plot(xs, ys, 'r-', lw=1.5)
        status = f'{edge_line.slope_deg:.2f}°'

    ax.set(xlabel='X [mm]',
           ylabel='Y [mm]',
           title=f'Foil {det_id} — Edge Detection ({edge_source})\n'
           f'Edge angle: {status}')
    fig.tight_layout()
    return fig


def _plot_aligned_foil(
        cropped: np.ndarray,
        det_id: int,
        circle_in_crop: Circle,
        px: float,
        clim: Tuple[float, float],
        contour_levels: Optional[List[float]] = None) -> plt.Figure:
    """Rotated+cropped foil with circle, contours, and center crosshair."""
    h, w = cropped.shape
    ext = [0, w * px, h * px, 0]
    v0, v1 = _percentile_clim(cropped, clim[0], clim[1])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cropped, extent=ext, vmin=v0, vmax=v1, cmap='hot')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')

    ax.add_patch(
        plt.Circle((circle_in_crop.x * px, circle_in_crop.y * px),
                   circle_in_crop.r * px,
                   color='cyan',
                   fill=False,
                   lw=1,
                   ls='--'))
    ax.axhline(circle_in_crop.y * px, color='cyan', lw=0.5, ls=':')
    ax.axvline(circle_in_crop.x * px, color='cyan', lw=0.5, ls=':')

    if contour_levels:
        x_arr = np.linspace(ext[0], ext[1], w)
        y_arr = np.linspace(ext[3], ext[2], h)
        cs = ax.contour(x_arr,
                        y_arr,
                        cropped,
                        levels=contour_levels,
                        colors='cyan',
                        linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

    ax.set(xlabel='X [mm]',
           ylabel='Y [mm]',
           title=f'Foil {det_id} — Aligned (edge at bottom)')
    fig.tight_layout()
    return fig


def _plot_profiles(cropped: np.ndarray,
                   circle_in_crop: Circle,
                   det_id: int,
                   px: float,
                   strip_half_px: int = 5) -> plt.Figure:
    """Horizontal and vertical profiles through the circle centre."""
    cy_px = int(round(circle_in_crop.y))
    cx_px = int(round(circle_in_crop.x))
    strip_width_px = 2 * strip_half_px + 1
    strip_width_mm = strip_width_px * px

    # Horizontal profile (average over row strip)
    y0 = max(0, cy_px - strip_half_px)
    y1 = min(cropped.shape[0], cy_px + strip_half_px + 1)
    h_profile = np.nanmean(cropped[y0:y1, :].astype(float), axis=0)
    h_mm = np.arange(len(h_profile)) * px

    # Vertical profile (average over column strip)
    x0 = max(0, cx_px - strip_half_px)
    x1 = min(cropped.shape[1], cx_px + strip_half_px + 1)
    v_profile = np.nanmean(cropped[:, x0:x1].astype(float), axis=1)
    v_mm = np.arange(len(v_profile)) * px

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h_mm, h_profile, 'r-', lw=1.2, label='Horizontal')
    ax.plot(v_mm, v_profile, 'b-', lw=1.2, label='Vertical')
    ax.axvline(circle_in_crop.x * px, color='red', ls='--', lw=0.6, alpha=0.5)
    ax.axvline(circle_in_crop.y * px, color='blue', ls='--', lw=0.6, alpha=0.5)
    ax.set(
        xlabel='Position [mm]',
        ylabel='Intensity',
        title=(f'Foil {det_id} — Profiles Through Centre\n'
               f'Strip width: {strip_width_px} px ({strip_width_mm:.2f} mm)'))
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_all_profiles(aligned_foils: Dict[int, AlignedFoil],
                       px: float,
                       strip_half_px: int = 5) -> plt.Figure:
    """Overlay horizontal and vertical profiles of all aligned foils."""
    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(14, 6))
    for det_id in sorted(aligned_foils.keys()):
        af = aligned_foils[det_id]
        crop = af.cropped_image
        cy_px_c = int(round(af.refined_circle_in_crop.y))
        cx_px_c = int(round(af.refined_circle_in_crop.x))

        # Horizontal profile
        y0 = max(0, cy_px_c - strip_half_px)
        y1 = min(crop.shape[0], cy_px_c + strip_half_px + 1)
        h_profile = np.nanmean(crop[y0:y1, :].astype(float), axis=0)
        cx_mm = af.refined_circle_in_crop.x * px
        h_mm = np.arange(len(h_profile)) * px - cx_mm
        ax_h.plot(h_mm, h_profile, lw=1.2, label=f'Foil {det_id}')

        # Vertical profile
        x0 = max(0, cx_px_c - strip_half_px)
        x1 = min(crop.shape[1], cx_px_c + strip_half_px + 1)
        v_profile = np.nanmean(crop[:, x0:x1].astype(float), axis=1)
        cy_mm = af.refined_circle_in_crop.y * px
        v_mm = np.arange(len(v_profile)) * px - cy_mm
        ax_v.plot(v_mm, v_profile, lw=1.2, label=f'Foil {det_id}')

    ax_h.set(xlabel='X relative to centre [mm]',
             ylabel='Intensity',
             title='Horizontal Profiles — All Foils')
    ax_h.legend()
    ax_v.set(xlabel='Y relative to centre [mm]',
             ylabel='Intensity',
             title='Vertical Profiles — All Foils')
    ax_v.legend()
    fig.tight_layout()
    return fig


def _plot_foils_overview(
    raw_images: Dict[int, np.ndarray],
    lv_images: Dict[int, np.ndarray],
    circles: Optional[Dict[int, Circle]],
    px: float,
    clim: Tuple[float, float],
    suptitle: str,
) -> plt.Figure:
    """2-row overview: top row = raw data, bottom row = live-view."""
    det_ids = sorted(raw_images.keys())
    n = len(det_ids)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10), squeeze=False)

    for col, det_id in enumerate(det_ids):
        for row, (imgs, label) in enumerate([(raw_images, 'Raw'),
                                             (lv_images, 'LV')]):
            ax = axes[row][col]
            img = imgs[det_id]
            v0, v1 = _percentile_clim(img, clim[0], clim[1])
            ext = [0, img.shape[1] * px, img.shape[0] * px, 0]
            im = ax.imshow(img, extent=ext, vmin=v0, vmax=v1, cmap='hot')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if circles and det_id in circles:
                c = circles[det_id]
                ax.add_patch(
                    plt.Circle((c.x * px, c.y * px),
                               c.r * px,
                               color='cyan',
                               fill=False,
                               lw=1.5))
                ax.plot(c.x * px, c.y * px, 'c+', ms=10, mew=1.5)

            ax.set(xlabel='X [mm]',
                   ylabel='Y [mm]',
                   title=f'Foil {det_id} — {label}')

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Parallel-mode helpers
# ---------------------------------------------------------------------------

_FIG_DPI = 200


def _fig_to_png(fig: plt.Figure, dpi: int = _FIG_DPI) -> bytes:
    """Render figure to PNG bytes with grid, then close it."""
    for ax in fig.get_axes():
        ax.grid(True, which='major', alpha=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


@dataclass
class _FoilResult:
    """Per-foil worker result (analysis + pre-rendered figures)."""
    det_id: int
    aligned_foil: AlignedFoil
    figure_pages: List[Tuple[str, bytes]]  # (caption, png_bytes)
    pre_rot_deg: Optional[float]
    post_rot_deg: Optional[float]
    messages: List[Tuple[str,
                         str]] = field(default_factory=list)  # (level, msg)


def _process_single_foil(params: dict) -> _FoilResult:
    """Worker: full processing + plotting for one foil.

    Designed to run in a subprocess via ProcessPoolExecutor.
    Receives a pre-computed Hough circle.  Flow:
      1. Gradient refinement on full raw image (pre-crop around Hough centre)
      2. Edge detection → rotation around refined circle centre
      3. Crop + aligned plots
    """
    import matplotlib
    matplotlib.use('Agg')

    det_id = params['det_id']
    raw_sm = params['raw_sm']
    lv_sm = params['lv_sm']
    hough_circle = params['hough_circle']
    edge_x_px = params['edge_x_px']
    stripe_width_px = params['stripe_width_px']
    edge_threshold = params['edge_threshold']
    edge_x_mm = params['edge_x_mm']
    px = params['px']
    clim = params['clim']
    crop_size_px = params['crop_size_px']
    contour_levels = params['contour_levels']

    figures: List[Tuple[str, bytes]] = []
    messages: List[Tuple[str, str]] = []

    # 1. Hough circle plot
    fig = _plot_circle_detection(raw_sm, lv_sm, hough_circle, det_id, px, clim)
    figures.append(
        (f"Foil {det_id}: Hough circle detection", _fig_to_png(fig)))

    # 2. Gradient circle refinement on full raw image
    refined, all_pts, filt_pts = _refine_circle_gradient(
        raw_sm, hough_circle, params['gradient_high'], params['gradient_low'],
        params['edge_distance_mm'], params['radius_tolerance'], px)
    dx = (refined.x - hough_circle.x) * px
    dy = (refined.y - hough_circle.y) * px
    dr = (refined.r - hough_circle.r) * px
    messages.append(('info', f"Foil {det_id}: circle refined "
                     f"\u0394center=({dx:+.3f},{dy:+.3f}) mm "
                     f"\u0394r={dr:+.3f} mm  "
                     f"({len(all_pts)} edge pts, {len(filt_pts)} filtered)"))

    fig = _plot_gradient_refinement(raw_sm, hough_circle, refined, all_pts,
                                    filt_pts, det_id, px, clim)
    figures.append(
        (f"Foil {det_id}: gradient circle refinement", _fig_to_png(fig)))

    # 3. Edge detection (using refined circle for display)
    crossings, edge_line = _find_edge(raw_sm, edge_x_px, edge_threshold,
                                      stripe_width_px)
    edge_source = "raw"
    if edge_line is None:
        messages.append(
            ('warning', f"Foil {det_id}: edge not found on raw, trying LV"))
        crossings, edge_line = _find_edge(lv_sm, edge_x_px, edge_threshold,
                                          stripe_width_px)
        edge_source = "lv" if edge_line is not None else "failed"

    pre_rot_deg = edge_line.slope_deg if edge_line else None

    fig = _plot_edge_detection(raw_sm,
                               crossings,
                               edge_line,
                               refined,
                               det_id,
                               edge_source,
                               px,
                               clim,
                               edge_x_mm=edge_x_mm)
    figures.append(
        (f"Foil {det_id}: edge detection ({edge_source})", _fig_to_png(fig)))

    # 4. Rotate around refined circle centre
    if edge_line is not None:
        rot_deg = _rotation_angle_edge_bottom(edge_line)
    else:
        rot_deg = 0.0
        messages.append(
            ('warning', f"Foil {det_id}: no edge found, skipping rotation"))

    rotated = _rotate_image_around(raw_sm, rot_deg, refined.x, refined.y)
    rotated_lv = _rotate_image_around(lv_sm, rot_deg, refined.x, refined.y)
    # Circle centre stays at (refined.x, refined.y) since we rotate around it
    rot_circle = Circle(x=refined.x, y=refined.y, r=refined.r)

    # 4b. Verify rotation
    post_crossings, post_edge = _find_edge(rotated, edge_x_px, edge_threshold,
                                           stripe_width_px)
    post_rot_deg = post_edge.slope_deg if post_edge else None

    if pre_rot_deg is not None and post_rot_deg is not None:
        msg = (
            f"Foil {det_id}: edge angle before={pre_rot_deg:.3f}\u00b0, "
            f"after={post_rot_deg:.3f}\u00b0 (rotation={rot_deg:.3f}\u00b0)")
        if abs(post_rot_deg) >= abs(pre_rot_deg):
            messages.append(
                ('warning',
                 f"{msg} \u2014 rotation did NOT reduce edge angle!"))
        else:
            messages.append(('info', msg))

    caption_post = (
        f"Foil {det_id}: post-rotation edge check "
        f"(before={pre_rot_deg:.2f}\u00b0, after={post_rot_deg:.2f}\u00b0)"
        if pre_rot_deg is not None and post_rot_deg is not None else
        f"Foil {det_id}: post-rotation edge check")
    fig = _plot_edge_detection(rotated,
                               post_crossings,
                               post_edge,
                               rot_circle,
                               det_id,
                               "post-rotation verification",
                               px,
                               clim,
                               edge_x_mm=edge_x_mm)
    figures.append((caption_post, _fig_to_png(fig)))

    # 5. Crop centred on (refined) circle
    cropped, offset = _crop_centered(rotated, rot_circle.x, rot_circle.y,
                                     crop_size_px)
    circle_in_crop = Circle(x=rot_circle.x - offset[0],
                            y=rot_circle.y - offset[1],
                            r=rot_circle.r)

    af = AlignedFoil(det_id=det_id,
                     circle=hough_circle,
                     edge_line=edge_line,
                     edge_source=edge_source,
                     rotation_deg=rot_deg,
                     rotated_image=rotated,
                     rotated_lv=rotated_lv,
                     rotated_circle=rot_circle,
                     cropped_image=cropped,
                     crop_offset=offset,
                     refined_circle_in_crop=circle_in_crop)

    # 6. Plot aligned foil
    fig = _plot_aligned_foil(cropped, det_id, circle_in_crop, px, clim,
                             contour_levels)
    figures.append((f"Foil {det_id}: aligned (rot {rot_deg:.2f}\u00b0, "
                    f"edge src: {edge_source})", _fig_to_png(fig)))

    # 7. Profile
    fig = _plot_profiles(cropped, circle_in_crop, det_id, px)
    figures.append((f"Foil {det_id}: H+V profiles", _fig_to_png(fig)))

    return _FoilResult(det_id=det_id,
                       aligned_foil=af,
                       figure_pages=figures,
                       pre_rot_deg=pre_rot_deg,
                       post_rot_deg=post_rot_deg,
                       messages=messages)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def align_mbo(config: MBOAlignConfig,
              output_dir: str = ".",
              pdf_path: Optional[str] = None,
              timing: bool = False,
              parallel: bool = False) -> Dict[int, AlignedFoil]:
    """Align all MBO foils from a survey NPZ.

    Parameters
    ----------
    config : MBOAlignConfig
    output_dir : str
    pdf_path : str, optional
    timing : bool
        If True, print per-step timing information.
    parallel : bool
        If True, process foils in parallel using multiprocessing.

    Returns
    -------
    dict mapping detector id → AlignedFoil
    """
    timings: List[Tuple[str, float]] = []

    def _tick(label: str, t0: float) -> float:
        elapsed = time.perf_counter() - t0
        timings.append((label, elapsed))
        return time.perf_counter()

    t0 = time.perf_counter()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    px = config.pixel_size_mm
    crop_size_px = int(round(config.crop_size_mm / px))
    stripe_width_px = max(1, int(round(config.edge_stripe_width_mm / px)))
    clim = (config.clip_percentile_low, config.clip_percentile_high)

    # Resolve input NPZ (may live in sibling data/ dir from a survey run)
    input_npz = resolve_file(config.input_npz, out)

    if pdf_path is None:
        pdf_path = str(out / f"mbo_align_{config.facility}.pdf")

    report = PDFReport(
        pdf_path,
        title=f"MBO Alignment: {config.facility.upper()}",
        config_path=str(input_npz),
    )

    report.add_text(
        f"Input survey NPZ:\n{input_npz}\n\n"
        f"Facility: {config.facility}\n"
        f"Pixel size: {config.pixel_size_mm} mm\n"
        f"Hough n_sigma: {config.n_sigma}\n"
        f"Hough probe_radius: {config.probe_radius}\n"
        f"Edge threshold: {config.edge_threshold}\n"
        f"Edge X positions [mm]: {config.edge_x_positions_mm}\n"
        f"Edge stripe width: {config.edge_stripe_width_mm} mm ({stripe_width_px} px)\n"
        f"Circle gradient high: {config.circle_gradient_high}\n"
        f"Circle gradient low: {config.circle_gradient_low}\n"
        f"Circle edge distance: {config.circle_edge_distance_mm} mm\n"
        f"Circle radius tolerance: {config.circle_radius_tolerance}\n"
        f"Crop size: {config.crop_size_mm} mm ({crop_size_px} px)\n"
        f"Percentile clipping: [{clim[0]}, {clim[1]}]\n"
        f"Contour levels: {config.contour_levels}",
        title="Data Source & Parameters",
    )

    # ---- Load smoothed images from survey NPZ ----
    foils = _load_survey_npz(str(input_npz))
    t0 = _tick("Load NPZ", t0)

    edge_x_px = [int(round(x_mm / px)) for x_mm in config.edge_x_positions_mm]
    aligned: Dict[int, AlignedFoil] = {}

    # ---- Hough circle detection for all foils ----
    hough_circles: Dict[int, Circle] = {}
    for det_id in sorted(foils.keys()):
        hough_circles[det_id] = _detect_circle_hough(foils[det_id]['lv'],
                                                     config.n_sigma,
                                                     config.probe_radius)
    t0 = _tick("Hough circle detection (all foils)", t0)

    # ---- Starting phase overview (raw + LV, all foils, Hough circles) ----
    raw_all = {did: foils[did]['raw'] for did in foils}
    lv_all = {did: foils[did]['lv'] for did in foils}
    fig_start = _plot_foils_overview(
        raw_all, lv_all, hough_circles, px, clim,
        "Starting Phase: All Foils (Hough circles)")
    report.add_figure(
        fig_start, caption="Starting phase overview (raw + LV, Hough circles)")
    t0 = _tick("Starting phase overview", t0)

    if parallel:
        # ---- Parallel mode: process foils in subprocesses ----
        foil_params = [{
            'det_id': det_id,
            'raw_sm': foils[det_id]['raw'],
            'lv_sm': foils[det_id]['lv'],
            'hough_circle': hough_circles[det_id],
            'edge_x_px': edge_x_px,
            'stripe_width_px': stripe_width_px,
            'edge_threshold': config.edge_threshold,
            'edge_x_mm': config.edge_x_positions_mm,
            'px': px,
            'clim': clim,
            'crop_size_px': crop_size_px,
            'contour_levels': config.contour_levels,
            'gradient_high': config.circle_gradient_high,
            'gradient_low': config.circle_gradient_low,
            'edge_distance_mm': config.circle_edge_distance_mm,
            'radius_tolerance': config.circle_radius_tolerance,
        } for det_id in sorted(foils.keys())]

        n_workers = min(len(foil_params), os.cpu_count() or 4)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_process_single_foil, foil_params))
        t0 = _tick(f"Parallel foil processing ({n_workers} workers)", t0)

        for result in results:
            for level, msg in result.messages:
                print(msg)
                getattr(logging, level)(msg)
            aligned[result.det_id] = result.aligned_foil
            for caption, png_bytes in result.figure_pages:
                report.add_png_bytes(png_bytes, caption)
        t0 = _tick("Add figures to PDF", t0)

    else:
        # ---- Sequential mode: process foils in main process ----
        for det_id in sorted(foils.keys()):
            raw_sm = foils[det_id]['raw']
            lv_sm = foils[det_id]['lv']
            hough = hough_circles[det_id]

            # -- 1. Hough circle plot --
            fig_circ = _plot_circle_detection(raw_sm, lv_sm, hough, det_id, px,
                                              clim)
            report.add_figure(fig_circ,
                              caption=f"Foil {det_id}: Hough circle detection")
            t0 = _tick(f"Foil {det_id}: plot Hough circle", t0)

            # -- 2. Gradient circle refinement on full raw image --
            refined, all_pts, filt_pts = _refine_circle_gradient(
                raw_sm, hough, config.circle_gradient_high,
                config.circle_gradient_low, config.circle_edge_distance_mm,
                config.circle_radius_tolerance, px)
            dx = (refined.x - hough.x) * px
            dy = (refined.y - hough.y) * px
            dr = (refined.r - hough.r) * px
            logging.info(
                f"Foil {det_id}: circle refined "
                f"Δcenter=({dx:+.3f},{dy:+.3f}) mm Δr={dr:+.3f} mm "
                f"({len(all_pts)} edge pts, {len(filt_pts)} filtered)")
            print(f"Foil {det_id}: circle refined "
                  f"Δcenter=({dx:+.3f},{dy:+.3f}) mm Δr={dr:+.3f} mm "
                  f"({len(all_pts)} edge pts, {len(filt_pts)} filtered)")
            t0 = _tick(f"Foil {det_id}: gradient circle refinement", t0)

            fig_grad = _plot_gradient_refinement(raw_sm, hough, refined,
                                                 all_pts, filt_pts, det_id, px,
                                                 clim)
            report.add_figure(
                fig_grad, caption=f"Foil {det_id}: gradient circle refinement")
            t0 = _tick(f"Foil {det_id}: plot gradient refinement", t0)

            # -- 3. Edge detection (try raw first, fall back to LV) --
            crossings, edge_line = _find_edge(raw_sm, edge_x_px,
                                              config.edge_threshold,
                                              stripe_width_px)
            edge_source = "raw"

            if edge_line is None:
                logging.warning(
                    f"Foil {det_id}: edge not found on raw, trying LV")
                crossings, edge_line = _find_edge(lv_sm, edge_x_px,
                                                  config.edge_threshold,
                                                  stripe_width_px)
                edge_source = "lv" if edge_line is not None else "failed"
            t0 = _tick(f"Foil {det_id}: edge detection", t0)

            pre_rot_deg = edge_line.slope_deg if edge_line else None

            fig_edge = _plot_edge_detection(
                raw_sm,
                crossings,
                edge_line,
                refined,
                det_id,
                edge_source,
                px,
                clim,
                edge_x_mm=config.edge_x_positions_mm)
            report.add_figure(
                fig_edge,
                caption=f"Foil {det_id}: edge detection ({edge_source})")
            t0 = _tick(f"Foil {det_id}: plot edge", t0)

            # -- 4. Rotate around refined circle centre --
            if edge_line is not None:
                rot_deg = _rotation_angle_edge_bottom(edge_line)
            else:
                rot_deg = 0.0
                logging.warning(
                    f"Foil {det_id}: no edge found, skipping rotation")

            rotated = _rotate_image_around(raw_sm, rot_deg, refined.x,
                                           refined.y)
            rotated_lv = _rotate_image_around(lv_sm, rot_deg, refined.x,
                                              refined.y)
            # Circle centre stays fixed (we rotate around it)
            rot_circle = Circle(x=refined.x, y=refined.y, r=refined.r)

            # -- 4b. Verify rotation: re-run edge on full rotated image --
            post_crossings, post_edge = _find_edge(rotated, edge_x_px,
                                                   config.edge_threshold,
                                                   stripe_width_px)
            post_rot_deg = post_edge.slope_deg if post_edge else None

            if pre_rot_deg is not None and post_rot_deg is not None:
                msg = (f"Foil {det_id}: edge angle before={pre_rot_deg:.3f}°, "
                       f"after={post_rot_deg:.3f}° (rotation={rot_deg:.3f}°)")
                if abs(post_rot_deg) >= abs(pre_rot_deg):
                    logging.warning(
                        f"{msg} — rotation did NOT reduce edge angle!")
                else:
                    logging.info(msg)
                print(msg)

            fig_post = _plot_edge_detection(
                rotated,
                post_crossings,
                post_edge,
                rot_circle,
                det_id,
                f"post-rotation verification",
                px,
                clim,
                edge_x_mm=config.edge_x_positions_mm)
            report.add_figure(
                fig_post,
                caption=f"Foil {det_id}: post-rotation edge check "
                f"(before={pre_rot_deg:.2f}°, after={post_rot_deg:.2f}°)"
                if pre_rot_deg is not None and post_rot_deg is not None else
                f"Foil {det_id}: post-rotation edge check")
            t0 = _tick(f"Foil {det_id}: rotation + verify", t0)

            # -- 5. Crop centred on (refined) circle --
            cropped, offset = _crop_centered(rotated, rot_circle.x,
                                             rot_circle.y, crop_size_px)
            circle_in_crop = Circle(x=rot_circle.x - offset[0],
                                    y=rot_circle.y - offset[1],
                                    r=rot_circle.r)

            aligned[det_id] = AlignedFoil(
                det_id=det_id,
                circle=hough,
                edge_line=edge_line,
                edge_source=edge_source,
                rotation_deg=rot_deg,
                rotated_image=rotated,
                rotated_lv=rotated_lv,
                rotated_circle=rot_circle,
                cropped_image=cropped,
                crop_offset=offset,
                refined_circle_in_crop=circle_in_crop,
            )

            # -- 6. Plot aligned foil --
            fig_al = _plot_aligned_foil(cropped, det_id, circle_in_crop, px,
                                        clim, config.contour_levels)
            report.add_figure(fig_al,
                              caption=f"Foil {det_id}: aligned "
                              f"(rot {rot_deg:.2f}°, edge src: {edge_source})")
            t0 = _tick(f"Foil {det_id}: plot aligned", t0)

            # -- 7. Profiles through centre --
            fig_prof = _plot_profiles(cropped, circle_in_crop, det_id, px)
            report.add_figure(fig_prof, caption=f"Foil {det_id}: H+V profiles")
            t0 = _tick(f"Foil {det_id}: plot profile", t0)

    # -- end of sequential / parallel branches --

    # -- Rotated phase overview (raw + LV, all foils) --
    if aligned:
        rot_raw = {did: af.rotated_image for did, af in aligned.items()}
        rot_lv = {did: af.rotated_lv for did, af in aligned.items()}
        rot_circles = {did: af.rotated_circle for did, af in aligned.items()}
        fig_rot = _plot_foils_overview(rot_raw, rot_lv, rot_circles, px, clim,
                                       "Rotated Phase: All Foils")
        report.add_figure(fig_rot, caption="Rotated phase overview (raw + LV)")
        t0 = _tick("Rotated phase overview", t0)

    # -- Overview: all profiles --
    if aligned:
        fig_all = _plot_all_profiles(aligned, px)
        report.add_figure(fig_all, caption="All foils: H+V profiles")
        t0 = _tick("Plot all profiles", t0)

    report.save()
    t0 = _tick("PDF save", t0)
    print(f"PDF report saved to: {pdf_path}")

    # -- Save aligned data (using refined circle) --
    npz_name = config.output_npz or f"mbo_align_{config.facility}.npz"
    npz_path = out / npz_name
    save_dict = {}
    for det_id, af in aligned.items():
        save_dict[f"foil_{det_id}_image"] = af.cropped_image
        save_dict[f"foil_{det_id}_circle_x"] = af.refined_circle_in_crop.x
        save_dict[f"foil_{det_id}_circle_y"] = af.refined_circle_in_crop.y
        save_dict[f"foil_{det_id}_circle_r"] = af.refined_circle_in_crop.r
        save_dict[f"foil_{det_id}_rotation_deg"] = af.rotation_deg
        save_dict[f"foil_{det_id}_edge_source"] = af.edge_source
    np.savez_compressed(npz_path, **save_dict)
    t0 = _tick("NPZ save", t0)
    print(f"Aligned data saved to: {npz_path}")

    # -- Print timing summary --
    if timing:
        total = sum(t for _, t in timings)
        print(f"\n{'─'*50}")
        print(f"{'Step':<40} {'Time':>8}")
        print(f"{'─'*50}")
        for label, dt in timings:
            print(f"{label:<40} {dt:>7.2f}s")
        print(f"{'─'*50}")
        print(f"{'TOTAL':<40} {total:>7.2f}s")

    return aligned
