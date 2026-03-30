"""Merge foils after background subtraction with 90-degree rotation.

Left foil is rotated 90 degrees counter-clockwise and right foil 90 degrees
clockwise around their respective circle centres.  The foil edges that were
previously at the bottom of each image face each other and are merged
into a panoramic view.  The result is cropped to the configured output
size (default 25 x 40 mm).

Two merges are produced for each scenario: a simple merge (margin = 0) and
a shift-optimised merge where the inter-foil distance is adjusted to best
match a smooth depth-dose profile shape.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d

from src.data.analysis import Circle
from src.data.matching import merge_images
from src.pub2026.config import MBOBackgroundSubtractionMergeConfig, resolve_file
from src.pub2026.mbo.match import _fit_and_evaluate
from src.pub2026.pdf_report import PDFReport


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

def _white_green_red_cmap():
    return LinearSegmentedColormap.from_list('white_green_red', [
        (0.0, (1, 1, 1, 1)),
        (0.5, (0, 1, 0, 1)),
        (1.0, (1, 0, 0, 1)),
    ])


def _ratio_cmap():
    """Blue-white-red diverging colour map centred at 1.0."""
    return LinearSegmentedColormap.from_list('bwr_ratio', [
        (0.0, (0.0, 0.0, 0.8, 1.0)),
        (0.5, (1.0, 1.0, 1.0, 1.0)),
        (1.0, (0.8, 0.0, 0.0, 1.0)),
    ])


# ---------------------------------------------------------------------------
# Rotation & geometry
# ---------------------------------------------------------------------------

def _rotate_ccw(image: np.ndarray, circle: Circle):
    """Rotate image 90 degrees counter-clockwise and transform circle centre.

    For a square image of side *S*, point (x, y) maps to (y, S-1-x).
    """
    S = image.shape[0]
    rotated = np.rot90(image, k=1)
    new_circle = Circle(x=circle.y, y=S - 1 - circle.x, r=circle.r)
    return rotated, new_circle


def _rotate_cw(image: np.ndarray, circle: Circle):
    """Rotate image 90 degrees clockwise and transform circle centre.

    For a square image of side *S*, point (x, y) maps to (S-1-y, x).
    """
    S = image.shape[0]
    rotated = np.rot90(image, k=3)
    new_circle = Circle(x=S - 1 - circle.y, y=circle.x, r=circle.r)
    return rotated, new_circle


def _shift_vertically(image: np.ndarray, circle: Circle,
                      target_y: float):
    """Shift image so circle centre sits at *target_y*, NaN-filling gaps."""
    h, w = image.shape
    shift = int(round(target_y - circle.y))
    if shift == 0:
        return image.copy(), circle
    shifted = np.full_like(image, np.nan, dtype=np.float64)
    if shift > 0:
        shifted[shift:, :] = image[:h - shift, :]
    else:
        shifted[:h + shift, :] = image[-shift:, :]
    return shifted, Circle(x=circle.x, y=circle.y + shift, r=circle.r)


def _rightmost_valid_col(image: np.ndarray) -> float:
    """Last column where at least 50 % of rows are finite.  Returns exclusive index."""
    valid_per_col = np.sum(np.isfinite(image), axis=0)
    threshold = image.shape[0] * 0.5
    active = np.where(valid_per_col >= threshold)[0]
    if len(active) == 0:
        return float(image.shape[1])
    return float(active[-1] + 1)


def _leftmost_valid_col(image: np.ndarray) -> float:
    """First column where at least 50 % of rows are finite."""
    valid_per_col = np.sum(np.isfinite(image), axis=0)
    threshold = image.shape[0] * 0.5
    active = np.where(valid_per_col >= threshold)[0]
    if len(active) == 0:
        return 0.0
    return float(active[0])


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_bg_sub(npz_path, foil_id, scenario: str):
    d = np.load(npz_path)
    if scenario == 'a':
        key = f'scenario_a_foil_{foil_id}_difference'
    else:
        key = f'scenario_b_foil_{foil_id}_ratio'
    image = d[key].astype(np.float64)
    circle = Circle(
        x=float(d[f'scenario_b_foil_{foil_id}_circle_x']),
        y=float(d[f'scenario_b_foil_{foil_id}_circle_y']),
        r=float(d[f'scenario_b_foil_{foil_id}_circle_r']),
    )
    return image, circle


# ---------------------------------------------------------------------------
# Output cropping
# ---------------------------------------------------------------------------

def _crop_output(merged: np.ndarray, left_c: Circle, right_c: Circle,
                 px: float, out_w_mm: float, out_h_mm: float):
    """Crop *merged* to *out_w_mm x out_h_mm* centred on mid-point of circles."""
    mid_x = (left_c.x + right_c.x) / 2
    mid_y = (left_c.y + right_c.y) / 2
    h, w = merged.shape
    ow = min(int(round(out_w_mm / px)), w)
    oh = min(int(round(out_h_mm / px)), h)
    x0 = int(np.clip(mid_x - ow / 2, 0, max(0, w - ow)))
    y0 = int(np.clip(mid_y - oh / 2, 0, max(0, h - oh)))
    cropped = merged[y0:y0 + oh, x0:x0 + ow]
    lc = Circle(x=left_c.x - x0, y=left_c.y - y0, r=left_c.r)
    rc = Circle(x=right_c.x - x0, y=right_c.y - y0, r=right_c.r)
    return cropped, lc, rc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_merged_2d(merged, lc, rc, px, cmap, vmin, vmax, title,
                    profile_y_offsets_mm=None, strip_w_mm=1.0):
    """Plot the merged 2D image with circle overlays and profile markers."""
    h, w = merged.shape
    extent = [0, w * px, h * px, 0]
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(np.ma.masked_invalid(merged), cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=extent, aspect='equal')
    for c in (lc, rc):
        ax.add_patch(plt.Circle((c.x * px, c.y * px), c.r * px,
                                color='black', fill=False, lw=1))
        ax.plot(c.x * px, c.y * px, 'k+', ms=12, mew=1.5)
    if profile_y_offsets_mm is not None:
        cy_mm = (lc.y + rc.y) / 2 * px
        shp_mm = strip_w_mm / 2
        for off in profile_y_offsets_mm:
            y_mm = cy_mm + off
            ax.axhline(y=y_mm - shp_mm, color='blue', linewidth=1, alpha=0.7)
            ax.axhline(y=y_mm + shp_mm, color='blue', linewidth=1, alpha=0.7)
    ax.set(title=title, xlabel='X [mm]', ylabel='Y [mm]')
    ax.grid(True, color='k', alpha=0.2, lw=0.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Signal')
    fig.tight_layout()
    return fig


def _plot_profiles(merged, lc, rc, px, offsets_mm, strip_w_mm=1.0,
                   vmin=None, vmax=None, title=''):
    """Horizontal profiles at circle-centre Y plus offsets."""
    cy_mm = (lc.y + rc.y) / 2 * px
    shp = max(1, int((strip_w_mm / 2) / px))
    x_mm = np.arange(merged.shape[1]) * px
    fig, ax = plt.subplots(figsize=(14, 6))
    for off in offsets_mm:
        y_mm = cy_mm + off
        ypx = int(y_mm / px)
        y0 = max(0, ypx - shp)
        y1 = min(merged.shape[0], ypx + shp)
        prof = np.nanmean(merged[y0:y1, :].astype(float), axis=0)
        lbl = 'Centre' if off == 0 else f'Centre {off:+.0f} mm'
        ax.plot(x_mm, prof, lw=1.5, label=f'{lbl} (Y={y_mm:.1f} mm)')
    ax.set_xlim(0, merged.shape[1] * px)
    if vmin is not None and vmax is not None:
        ax.set_ylim(vmin, vmax)
    ax.set(xlabel='X [mm]', ylabel='Signal',
           title=title or 'Horizontal profiles at different Y positions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_merged_contours(merged, lc, rc, px, cmap, vmin, vmax,
                          contour_levels, title,
                          profile_y_offsets_mm=None, strip_w_mm=1.0):
    """2D image with filled contours and isolines at given levels."""
    h, w = merged.shape
    extent = [0, w * px, h * px, 0]
    x_mm = np.linspace(0, w * px, w)
    y_mm = np.linspace(0, h * px, h)
    data = np.nan_to_num(merged, nan=0.0)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(np.ma.masked_invalid(merged), cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=extent, aspect='equal')
    cs = ax.contour(x_mm, y_mm, data, levels=contour_levels,
                    colors='black', linewidths=0.8)
    # Use appropriate precision: integer format for large levels, 2 decimals otherwise
    fmt = '%.0f' if all(abs(l) >= 10 for l in contour_levels) else '%.2f'
    ax.clabel(cs, inline=True, fontsize=7, fmt=fmt)
    for c in (lc, rc):
        ax.add_patch(plt.Circle((c.x * px, c.y * px), c.r * px,
                                color='black', fill=False, lw=1))
        ax.plot(c.x * px, c.y * px, 'k+', ms=12, mew=1.5)
    if profile_y_offsets_mm is not None:
        cy_mm = (lc.y + rc.y) / 2 * px
        shp_mm = strip_w_mm / 2
        for off in profile_y_offsets_mm:
            y_pos = cy_mm + off
            ax.axhline(y=y_pos - shp_mm, color='blue', linewidth=1, alpha=0.7)
            ax.axhline(y=y_pos + shp_mm, color='blue', linewidth=1, alpha=0.7)
    ax.set(title=title, xlabel='X [mm]', ylabel='Y [mm]')
    ax.grid(True, color='k', alpha=0.2, lw=0.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Signal')
    fig.tight_layout()
    return fig

def _merge_scenario(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_circle: Circle,
    right_circle: Circle,
    left_edge_y: float,
    right_edge_y: float,
    config: MBOBackgroundSubtractionMergeConfig,
    cmap,
    vmin: float,
    vmax: Optional[float],
    contour_levels: List[float],
    scenario_label: str,
    report: PDFReport,
) -> Dict:
    """Rotate, merge, optimise, and report for one scenario.

    Left foil is rotated 90 degrees CCW, right foil 90 degrees CW.  The
    bottom edges (at *left_edge_y* and *right_edge_y*) become vertical edges
    that face each other after rotation and are merged.

    After CCW rotation the bottom edge at Y maps to a vertical edge at X = Y.
    After CW rotation the bottom edge at Y maps to X = S - 1 - Y.
    """
    px = config.pixel_size_mm
    S = left_img.shape[0]

    # -- Rotate images and circles ----------------------------------------
    left_rot, left_c_rot = _rotate_ccw(left_img, left_circle)
    right_rot, right_c_rot = _rotate_cw(right_img, right_circle)

    # -- Edge X in rotated coordinates ------------------------------------
    left_edge_x = left_edge_y
    right_edge_x = float(S) - right_edge_y

    # Clamp edges to the valid-data boundary so NaN columns from the
    # bg-sub centering shift are excluded from the merge.
    left_edge_x = min(left_edge_x, _rightmost_valid_col(left_rot))
    right_edge_x = max(right_edge_x, _leftmost_valid_col(right_rot))

    # -- Vertical shift to align circle centres at image midpoint ---------
    target_y = S / 2.0
    left_sh, left_c_sh = _shift_vertically(left_rot, left_c_rot, target_y)
    right_sh, right_c_sh = _shift_vertically(right_rot, right_c_rot, target_y)

    # ================================================================
    # SIMPLE MERGE (margin = config.merge_margin)
    # ================================================================
    merged, x_off = merge_images(left_sh, right_sh, left_edge_x,
                                 right_edge_x, config.merge_margin)
    lc_m = left_c_sh
    rc_m = Circle(x=right_c_sh.x + x_off, y=right_c_sh.y, r=right_c_sh.r)

    cropped, lc_c, rc_c = _crop_output(
        merged, lc_m, rc_m, px,
        config.output_width_mm, config.output_height_mm)

    if vmax is None:
        vmax = float(np.nanmax(cropped))

    fig = _plot_merged_2d(cropped, lc_c, rc_c, px, cmap, vmin, vmax,
                          f'{scenario_label}: Simple Merge '
                          f'(margin={config.merge_margin}px)',
                          profile_y_offsets_mm=config.profile_y_offsets_mm,
                          strip_w_mm=config.profile_strip_width_mm)
    report.add_figure(fig, caption=f'{scenario_label}: merged foils (simple)')
    plt.close(fig)

    fig = _plot_merged_contours(
        cropped, lc_c, rc_c, px, cmap, vmin, vmax, contour_levels,
        f'{scenario_label}: Isolines — Simple Merge',
        profile_y_offsets_mm=config.profile_y_offsets_mm,
        strip_w_mm=config.profile_strip_width_mm)
    report.add_figure(fig,
                      caption=f'{scenario_label}: isolines (simple)')
    plt.close(fig)

    fig = _plot_profiles(cropped, lc_c, rc_c, px,
                         config.profile_y_offsets_mm,
                         config.profile_strip_width_mm,
                         title=f'{scenario_label}: Profiles - Simple Merge')
    report.add_figure(fig,
                      caption=f'{scenario_label}: horizontal profiles (simple)')
    plt.close(fig)

    # ================================================================
    # SPLINE SHIFT OPTIMISATION
    # ================================================================
    py_min = int(config.profile_y_min_mm / px)
    py_max = int(config.profile_y_max_mm / px)
    py_min = max(0, min(py_min, cropped.shape[0] - 1))
    py_max = max(py_min + 1, min(py_max, cropped.shape[0]))

    x_mm = np.arange(cropped.shape[1]) * px
    prof_raw = np.nanmean(cropped[py_min:py_max, :].astype(float), axis=0)
    prof = gaussian_filter1d(np.nan_to_num(prof_raw, nan=0.0),
                             sigma=config.smoothing_sigma_px)

    m1 = (x_mm >= config.section1_x_min) & (x_mm <= config.section1_x_max)
    m2 = (x_mm >= config.section2_x_min) & (x_mm <= config.section2_x_max)
    x1, y1 = x_mm[m1], prof[m1]
    x2_orig, y2 = x_mm[m2], prof[m2]

    if len(x1) > 3 and len(x2_orig) > 3:
        shifts = np.linspace(-2, 2, 401)
        rss_arr = np.array(
            [_fit_and_evaluate(x1, y1, x2_orig, y2, s)[1] for s in shifts])
        opt_idx = int(np.argmin(rss_arr))
        opt_shift = float(shifts[opt_idx])
    else:
        opt_shift = 0.0
        rss_arr = np.array([0.0])
        shifts = np.array([0.0])

    # RSS vs shift plot
    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    ax_s.plot(shifts, rss_arr)
    ax_s.axvline(opt_shift, color='r', ls='--',
                 label=f'Optimal: {opt_shift:.2f} mm')
    ax_s.axvline(0, color='k', ls=':', alpha=0.5, label='No shift')
    ax_s.set(xlabel='Shift [mm]', ylabel='RSS',
             title=f'{scenario_label}: Spline alignment optimisation')
    ax_s.legend()
    ax_s.grid(True, alpha=0.3)
    fig_s.tight_layout()
    report.add_figure(fig_s,
                      caption=f'{scenario_label}: RSS vs shift '
                              f'(optimal {opt_shift:.2f} mm)')
    plt.close(fig_s)

    # ================================================================
    # OPTIMISED MERGE
    # ================================================================
    opt_margin = int(round(opt_shift / px))
    merged_opt, x_off_opt = merge_images(
        left_sh, right_sh, left_edge_x, right_edge_x, opt_margin)
    lc_mo = left_c_sh
    rc_mo = Circle(x=right_c_sh.x + x_off_opt, y=right_c_sh.y,
                   r=right_c_sh.r)

    cropped_opt, lc_co, rc_co = _crop_output(
        merged_opt, lc_mo, rc_mo, px,
        config.output_width_mm, config.output_height_mm)

    fig = _plot_merged_2d(
        cropped_opt, lc_co, rc_co, px, cmap, vmin, vmax,
        f'{scenario_label}: Optimised Merge '
        f'(margin={opt_margin}px, shift={opt_shift:.2f} mm)',
        profile_y_offsets_mm=config.profile_y_offsets_mm,
        strip_w_mm=config.profile_strip_width_mm)
    report.add_figure(fig,
                      caption=f'{scenario_label}: merged foils (optimised)')
    plt.close(fig)

    fig = _plot_merged_contours(
        cropped_opt, lc_co, rc_co, px, cmap, vmin, vmax, contour_levels,
        f'{scenario_label}: Isolines — Optimised Merge '
        f'(shift={opt_shift:.2f} mm)',
        profile_y_offsets_mm=config.profile_y_offsets_mm,
        strip_w_mm=config.profile_strip_width_mm)
    report.add_figure(fig,
                      caption=f'{scenario_label}: isolines (optimised)')
    plt.close(fig)

    fig = _plot_profiles(
        cropped_opt, lc_co, rc_co, px,
        config.profile_y_offsets_mm, config.profile_strip_width_mm,
        title=f'{scenario_label}: Profiles - Optimised Merge '
              f'(shift={opt_shift:.2f} mm)')
    report.add_figure(fig,
                      caption=f'{scenario_label}: horizontal profiles '
                              '(optimised)')
    plt.close(fig)

    # Profile comparison: simple vs optimised
    x_mm_opt = np.arange(cropped_opt.shape[1]) * px
    py_min_opt = max(0, min(py_min, cropped_opt.shape[0] - 1))
    py_max_opt = max(py_min_opt + 1, min(py_max, cropped_opt.shape[0]))
    prof_opt_raw = np.nanmean(
        cropped_opt[py_min_opt:py_max_opt, :].astype(float), axis=0)
    prof_opt = gaussian_filter1d(np.nan_to_num(prof_opt_raw, nan=0.0),
                                 sigma=config.smoothing_sigma_px)

    fig_cmp, ax_cmp = plt.subplots(figsize=(14, 6))
    ax_cmp.plot(x_mm, prof, 'b--', lw=1, alpha=0.6, label='Simple merge')
    ax_cmp.plot(x_mm_opt, prof_opt, 'r-', lw=1.5, label='Optimised merge')
    ax_cmp.set(xlabel='X [mm]', ylabel='Signal',
               title=f'{scenario_label}: Profile comparison '
                     f'(Y={config.profile_y_min_mm}-'
                     f'{config.profile_y_max_mm} mm)')
    ax_cmp.legend()
    ax_cmp.grid(True, alpha=0.3)
    fig_cmp.tight_layout()
    report.add_figure(fig_cmp,
                      caption=f'{scenario_label}: profile before/after '
                              'optimisation')
    plt.close(fig_cmp)

    return {
        'merged': cropped,
        'merged_opt': cropped_opt,
        'left_circle': lc_c,
        'right_circle': rc_c,
        'left_circle_opt': lc_co,
        'right_circle_opt': rc_co,
        'optimal_shift_mm': opt_shift,
        'optimised_margin_px': opt_margin,
        'x_mm_opt': x_mm_opt,
        'profile_opt_smooth': prof_opt,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def merge_background_subtraction(
        config: MBOBackgroundSubtractionMergeConfig,
        output_dir: str = '.',
        pdf_path: Optional[str] = None) -> Dict:
    """Merge foils after background subtraction for scenarios A and B.

    Each foil is rotated (left 90 degrees CCW, right 90 degrees CW) so that
    the bottom edges face each other.  Edge positions are detected on the
    aligned target images (strong signal) and mapped to the background-
    subtracted coordinate system before merging.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bg_sub_npz = Path(resolve_file(config.bg_subtraction_npz, out))

    if pdf_path is None:
        pdf_path = str(out / f'mbo_bg_sub_merge_{config.facility}.pdf')

    report = PDFReport(
        pdf_path,
        title=f'MBO BG-Sub Merge: {config.facility.upper()}',
        config_path=str(bg_sub_npz),
    )

    px = config.pixel_size_mm
    left_id, right_id = config.foil_ids

    report.add_text(
        f'Background subtraction NPZ: {bg_sub_npz}\n\n'
        f'Foil IDs: left={left_id}, right={right_id}\n'
        f'Pixel size: {px} mm\n'
        f'Output size: {config.output_width_mm} x '
        f'{config.output_height_mm} mm\n'
        f'Profile Y offsets: {config.profile_y_offsets_mm}\n'
        f'Section 1: X = {config.section1_x_min}-'
        f'{config.section1_x_max} mm\n'
        f'Section 2: X = {config.section2_x_min}-'
        f'{config.section2_x_max} mm',
        title='Inputs and parameters')

    # -- Detect bottom edge on aligned target images -------------------------
    # Use circle geometry: the bottom of the circular detector is at
    # circle.y + circle.r.  This is more robust than threshold-based
    # scanning which fails for difference images (no NaN mask).
    left_bg_a, left_bg_circle = _load_bg_sub(bg_sub_npz, left_id, 'a')
    right_bg_a, right_bg_circle = _load_bg_sub(bg_sub_npz, right_id, 'a')

    left_edge_y_bgsub = left_bg_circle.y + left_bg_circle.r
    right_edge_y_bgsub = right_bg_circle.y + right_bg_circle.r

    S = left_bg_a.shape[0]
    report.add_text(
        f'Left foil circle bottom: Y = {left_edge_y_bgsub:.1f} px '
        f'({left_edge_y_bgsub * px:.2f} mm)\n'
        f'Right foil circle bottom: Y = {right_edge_y_bgsub:.1f} px '
        f'({right_edge_y_bgsub * px:.2f} mm)\n'
        f'Image size: {S}x{S} px\n\n'
        f'After rotation:\n'
        f'  Left (CCW):  edge at X = {left_edge_y_bgsub:.1f} px\n'
        f'  Right (CW):  edge at X = {S - right_edge_y_bgsub:.1f} px',
        title='Edge detection results (circle geometry)')

    # ======================================================================
    # SCENARIO A -- difference images
    # ======================================================================
    report.add_text(
        'Merging Scenario A (raw signal subtraction) difference images.\n'
        'Left foil rotated 90 degrees CCW, right foil 90 degrees CW.',
        title='===  SCENARIO A MERGE  ===')

    cmap_a = _white_green_red_cmap()
    result_a = _merge_scenario(
        left_bg_a, right_bg_a,
        left_bg_circle, right_bg_circle,
        left_edge_y_bgsub, right_edge_y_bgsub,
        config, cmap_a,
        vmin=0.0, vmax=None,
        contour_levels=config.scenario_a_contour_levels,
        scenario_label='Scenario A', report=report)

    # ======================================================================
    # SCENARIO B -- ratio images
    # ======================================================================
    report.add_text(
        'Merging Scenario B (background-subtracted ratio) images.\n'
        'Left foil rotated 90 degrees CCW, right foil 90 degrees CW.',
        title='===  SCENARIO B MERGE  ===')

    left_ratio, left_ratio_circle = _load_bg_sub(bg_sub_npz, left_id, 'b')
    right_ratio, right_ratio_circle = _load_bg_sub(bg_sub_npz, right_id, 'b')

    cmap_b = _white_green_red_cmap()
    result_b = _merge_scenario(
        left_ratio, right_ratio,
        left_ratio_circle, right_ratio_circle,
        left_edge_y_bgsub, right_edge_y_bgsub,
        config, cmap_b,
        vmin=0.0, vmax=None,
        contour_levels=config.scenario_b_contour_levels,
        scenario_label='Scenario B', report=report)

    # -- Save PDF ------------------------------------------------------------
    report.save()
    print(f'PDF report saved to: {pdf_path}')

    # -- Save NPZ ------------------------------------------------------------
    npz_name = (config.output_npz
                or f'mbo_bg_sub_merge_{config.facility}.npz')
    output_npz = out / npz_name

    save_dict = {
        'pixel_size_mm': px,
        'left_foil_id': left_id,
        'right_foil_id': right_id,
    }

    for label, res in [('scenario_a', result_a), ('scenario_b', result_b)]:
        save_dict[f'{label}_merged'] = res['merged']
        save_dict[f'{label}_merged_opt'] = res['merged_opt']
        save_dict[f'{label}_left_circle_x'] = res['left_circle'].x
        save_dict[f'{label}_left_circle_y'] = res['left_circle'].y
        save_dict[f'{label}_left_circle_r'] = res['left_circle'].r
        save_dict[f'{label}_right_circle_x'] = res['right_circle'].x
        save_dict[f'{label}_right_circle_y'] = res['right_circle'].y
        save_dict[f'{label}_right_circle_r'] = res['right_circle'].r
        save_dict[f'{label}_left_circle_opt_x'] = res['left_circle_opt'].x
        save_dict[f'{label}_left_circle_opt_y'] = res['left_circle_opt'].y
        save_dict[f'{label}_left_circle_opt_r'] = res['left_circle_opt'].r
        save_dict[f'{label}_right_circle_opt_x'] = res['right_circle_opt'].x
        save_dict[f'{label}_right_circle_opt_y'] = res['right_circle_opt'].y
        save_dict[f'{label}_right_circle_opt_r'] = res['right_circle_opt'].r
        save_dict[f'{label}_optimal_shift_mm'] = res['optimal_shift_mm']
        save_dict[f'{label}_optimised_margin_px'] = res['optimised_margin_px']

    np.savez_compressed(output_npz, **save_dict)
    print(f'Merged data saved to: {output_npz}')

    # -- Save scenario B optimised profile NPZ (compatible with match step) --
    profile_name = (config.output_profile_npz
                    or f'mbo_bg_sub_profile_{config.facility}.npz')
    profile_npz = out / profile_name
    np.savez_compressed(
        profile_npz,
        x_mm=result_b['x_mm_opt'],
        profile_opt_smooth=result_b['profile_opt_smooth'],
        profile_y_min_mm=config.profile_y_min_mm,
        profile_y_max_mm=config.profile_y_max_mm,
        smoothing_sigma_px=config.smoothing_sigma_px,
        pixel_size_mm=px,
    )
    print(f'Scenario B profile saved to: {profile_npz}')

    return {
        'scenario_a': result_a,
        'scenario_b': result_b,
        'output_npz': output_npz,
        'output_profile_npz': profile_npz,
        'pdf_path': pdf_path,
    }
