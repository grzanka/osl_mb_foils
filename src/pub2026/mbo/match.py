"""MBO foil matching: edge detection, rotation, merging, spline optimization.

Ported from notebooks: 0.2-match-aic144.ipynb, 0.2-match-ccb.ipynb
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from src.data.analysis import Circle
from src.data.matching import (
    find_edge_crossings,
    fit_edge_line,
    rotate_image,
    rotate_circle,
    rotate_edge_line,
    transform_edge_line_for_crop,
    crop_centered,
    merge_images,
)
from src.pub2026.config import MBOMatchConfig, resolve_file
from src.pub2026.pdf_report import PDFReport


def _white_green_red_cmap(white_threshold: float = 0.05):
    return LinearSegmentedColormap.from_list('white_green_red', [
        (0.0, (1, 1, 1, 1)),
        (white_threshold, (1, 1, 1, 1)),
        (0.5, (0, 1, 0, 1)),
        (1.0, (1, 0, 0, 1)),
    ])


# -- Gradient-based circle refinement --


def find_gradient_edge_points(image: np.ndarray, high_thresh: float,
                              low_thresh: float) -> List[Tuple[int, int]]:
    """Find edge points via radial scanning where intensity transitions
    from *high_thresh* down to *low_thresh*."""
    h, w = image.shape
    cy, cx = h // 2, w // 2
    edge_points = []
    n_angles = 360

    for ai in range(n_angles):
        angle = 2 * np.pi * ai / n_angles
        max_r = min(h, w) // 2
        prev_val = None
        in_trans = False
        trans_start = None

        for r in range(10, max_r):
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:
                val = image[y, x]
                if prev_val is not None:
                    if not in_trans and prev_val > high_thresh and val <= high_thresh:
                        in_trans = True
                        trans_start = (x, y)
                    if in_trans and val < low_thresh:
                        if trans_start is not None:
                            mx = (trans_start[0] + x) // 2
                            my = (trans_start[1] + y) // 2
                            edge_points.append((mx, my))
                        break
                prev_val = val
    return edge_points


def filter_edge_points_near_circle(edge_points, circle: Circle,
                                   max_distance_px: float):
    filtered = []
    for x, y in edge_points:
        dist = abs(np.sqrt((x - circle.x)**2 + (y - circle.y)**2) - circle.r)
        if dist <= max_distance_px:
            filtered.append((x, y))
    return filtered


def fit_circle_to_points(points, initial: Circle,
                         radius_tolerance: float) -> Circle:
    if len(points) < 10:
        return initial
    pts = np.array(points, dtype=float)
    xp, yp = pts[:, 0], pts[:, 1]
    min_r = initial.r * (1 - radius_tolerance)
    max_r = initial.r * (1 + radius_tolerance)

    def obj(p):
        return np.sum((np.sqrt((xp - p[0])**2 + (yp - p[1])**2) - p[2])**2)

    res = minimize(obj, [initial.x, initial.y, initial.r],
                   method='L-BFGS-B',
                   bounds=[(None, None), (None, None), (min_r, max_r)])
    return Circle(x=res.x[0], y=res.x[1], r=res.x[2])


def _shift_image_vertically(image: np.ndarray, circle: Circle,
                            target_y: float):
    h, w = image.shape
    shift = int(round(target_y - circle.y))
    if shift == 0:
        return image.copy(), circle
    shifted = np.zeros_like(image)
    if shift > 0:
        shifted[shift:, :] = image[:h - shift, :]
    else:
        shifted[:h + shift, :] = image[-shift:, :]
    return shifted, Circle(x=circle.x, y=circle.y + shift, r=circle.r)


def _fit_and_evaluate(x1, y1, x2, y2, shift, spline_s_factor: float = 100):
    x2s = x2 + shift
    xc = np.concatenate([x1, x2s])
    yc = np.concatenate([y1, y2])
    idx = np.argsort(xc)
    xc, yc = xc[idx], yc[idx]
    spl = UnivariateSpline(xc, yc, s=len(xc) * spline_s_factor)
    rss = float(np.sum((yc - spl(xc))**2))
    mae = float(np.mean(np.abs(yc - spl(xc))))
    return spl, rss, mae, x2s


# -- 2D visualisation helpers --


def _plot_isosignal_contours(merged_image: np.ndarray,
                             left_circle,
                             right_circle,
                             px: float,
                             cmap,
                             vmax: float,
                             contour_levels=None,
                             crop_half_x: float = 20.0,
                             crop_half_y: float = 10.0,
                             title: str = '') -> plt.Figure:
    """Create 2D signal visualisation with isosignal contour lines.

    Crops around the midpoint between circle centres and overlays contour
    lines and circle outlines.
    """
    if contour_levels is None:
        contour_levels = [600, 700, 800, 900, 1000, 1200, 1400, 1600]

    mid_x_mm = (left_circle.x + right_circle.x) / 2 * px
    mid_y_mm = (left_circle.y + right_circle.y) / 2 * px

    h, w = merged_image.shape
    px_x_min = max(0, int((mid_x_mm - crop_half_x) / px))
    px_x_max = min(w, int((mid_x_mm + crop_half_x) / px))
    px_y_min = max(0, int((mid_y_mm - crop_half_y) / px))
    px_y_max = min(h, int((mid_y_mm + crop_half_y) / px))

    cropped = merged_image[px_y_min:px_y_max, px_x_min:px_x_max].astype(float)
    crop_extent = [px_x_min * px, px_x_max * px, px_y_max * px, px_y_min * px]
    h_crop, w_crop = cropped.shape
    x_arr = np.linspace(crop_extent[0], crop_extent[1], w_crop)
    y_arr = np.linspace(crop_extent[3], crop_extent[2], h_crop)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(cropped, cmap=cmap, vmin=0, vmax=vmax, extent=crop_extent)

    cs = ax.contour(x_arr,
                    y_arr,
                    cropped,
                    levels=contour_levels,
                    colors='black',
                    linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

    for circle in [left_circle, right_circle]:
        cx, cy, r = circle.x * px, circle.y * px, circle.r * px
        ax.add_patch(
            plt.Circle((cx, cy), r, color='black', fill=False, lw=1, ls='--'))
        ax.plot(cx, cy, 'k+', ms=12, mew=1.5)

    ax.plot(mid_x_mm, mid_y_mm, 'kx', ms=10, mew=2)
    ax.set(xlabel='X [mm]',
           ylabel='Y [mm]',
           title=title or '2D Signal with Isosignal Lines')
    fig.colorbar(ax.images[0],
                 ax=ax,
                 fraction=0.03,
                 pad=0.02,
                 label='Intensity')
    fig.tight_layout()
    return fig


def _plot_horizontal_profiles_at_y_positions(
        merged_image: np.ndarray,
        left_circle,
        right_circle,
        px: float,
        offsets_mm=(-4, 0, 4),
        strip_width_mm: float = 1.0,
        vmax: float = 3000.0,
        crop_half_x: float = 20.0) -> plt.Figure:
    """Plot horizontal profiles at the circle-centre Y and offsets."""
    center_y_mm = (left_circle.y + right_circle.y) / 2 * px
    mid_x_mm = (left_circle.x + right_circle.x) / 2 * px
    strip_half_px = max(1, int((strip_width_mm / 2) / px))

    x_mm = np.arange(merged_image.shape[1]) * px
    fig, ax = plt.subplots(figsize=(14, 6))

    for offset in offsets_mm:
        y_mm_pos = center_y_mm + offset
        y_px = int(y_mm_pos / px)
        y0 = max(0, y_px - strip_half_px)
        y1 = min(merged_image.shape[0], y_px + strip_half_px)
        profile = np.nanmean(merged_image[y0:y1, :].astype(float), axis=0)
        lbl = 'Center' if offset == 0 else f'Center {offset:+.0f} mm'
        ax.plot(x_mm, profile, lw=1.5, label=f'{lbl} (Y={y_mm_pos:.1f} mm)')

    ax.set_xlim(mid_x_mm - crop_half_x, mid_x_mm + crop_half_x)
    ax.set_ylim(0, vmax)
    ax.set(
        xlabel='X [mm]',
        ylabel='Intensity',
        title=
        f'Horizontal profiles ({strip_width_mm:.0f} mm strips) at circle centre and offsets'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# -- Main pipeline --


def match_mbo(config: MBOMatchConfig,
              output_dir: str = ".",
              pdf_path: Optional[str] = None) -> Dict:
    """Run MBO foil matching pipeline for one facility / foil pair.

    Returns dict with merged images, profile data, optimal shift.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(config.data_dir).resolve()

    if pdf_path is None:
        pdf_path = str(
            out /
            f"mbo_match_{config.facility}_{config.left_foil_id}_{config.right_foil_id}.pdf"
        )

    report = PDFReport(
        pdf_path,
        title=
        f"MBO Match: {config.facility.upper()} foils {config.left_foil_id}&{config.right_foil_id}",
        config_path=str(data_path))

    report.add_text(
        f"Input TIFF directory (from explore):\n{data_path}\n\n"
        f"Input pickle: {config.input_pkl or f'mbo_{config.facility}_processed.pkl'}",
        title="Data Source Paths")

    cmap = _white_green_red_cmap(config.white_threshold)
    px = config.pixel_size_mm

    # 1. Load pickle ---------------------------------------------------------
    pkl_name = config.input_pkl or f"mbo_{config.facility}_processed.pkl"
    pkl_path = resolve_file(
        str(data_path /
            pkl_name) if not Path(pkl_name).is_absolute() else pkl_name, out)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    left_img = np.fliplr(data.data[config.left_foil_id].raw.image)
    right_img = np.fliplr(data.data[config.right_foil_id].raw.image)

    def _flip_circle(det_id, img):
        c = data.data[det_id].circle
        return Circle(x=img.shape[1] - c.x, y=c.y, r=c.r)

    left_circle = _flip_circle(config.left_foil_id, left_img)
    right_circle = _flip_circle(config.right_foil_id, right_img)

    # Report: raw foils with initial circles
    fig_raw, axes_raw = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, circ, lbl in [
        (axes_raw[0], left_img, left_circle,
         f'Left foil {config.left_foil_id}'),
        (axes_raw[1], right_img, right_circle,
         f'Right foil {config.right_foil_id}'),
    ]:
        ext = [0, img.shape[1] * px, img.shape[0] * px, 0]
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=config.vmax, extent=ext)
        ax.add_patch(
            plt.Circle((circ.x * px, circ.y * px),
                       circ.r * px,
                       color='blue',
                       fill=False,
                       lw=1))
        ax.set(title=lbl, xlabel='X [mm]', ylabel='Y [mm]')
        fig_raw.colorbar(im,
                         ax=ax,
                         fraction=0.046,
                         pad=0.04,
                         label='Intensity')
    fig_raw.tight_layout()
    report.add_figure(fig_raw,
                      caption='Step 1: Raw foils (flipped) with Hough circles',
                      source_paths=[str(data_path)])

    # 2. Edge detection & line fitting ---------------------------------------
    edge_x_px = [int(round(x_mm / px)) for x_mm in config.edge_x_positions_mm]
    left_xr = find_edge_crossings(left_img, edge_x_px, config.edge_threshold,
                                  config.edge_stripe_width)
    right_xr = find_edge_crossings(right_img, edge_x_px, config.edge_threshold,
                                   config.edge_stripe_width)
    left_line = fit_edge_line(left_xr, robust=True)
    right_line = fit_edge_line(right_xr, robust=True)

    # Report: edge detection
    fig_edge, axes_edge = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, xr, line, circ, lbl in [
        (axes_edge[0], left_img, left_xr, left_line, left_circle,
         f'Left foil {config.left_foil_id}'),
        (axes_edge[1], right_img, right_xr, right_line, right_circle,
         f'Right foil {config.right_foil_id}'),
    ]:
        ext = [0, img.shape[1] * px, img.shape[0] * px, 0]
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=config.vmax, extent=ext)
        for pt_x, pt_y in xr:
            if pt_y is not None:
                ax.plot(pt_x * px, pt_y * px, 'r+', ms=8, mew=2)
        if hasattr(line, 'points') and line.points:
            xs = [p[0] * px for p in line.points]
            ys = [p[1] * px for p in line.points]
            ax.plot(xs, ys, 'r-', lw=1.5)
        ax.set(title=f'{lbl}\nEdge angle: {line.slope_deg:.2f}°',
               xlabel='X [mm]',
               ylabel='Y [mm]')
        fig_edge.colorbar(im,
                          ax=ax,
                          fraction=0.046,
                          pad=0.04,
                          label='Intensity')
    fig_edge.tight_layout()
    report.add_figure(fig_edge,
                      caption='Step 2: Edge detection and line fitting',
                      source_paths=[str(data_path)])

    # 3. Rotate to make edges vertical ----------------------------------------
    la = left_line.slope_deg + 90
    ra = right_line.slope_deg - 90
    left_rot = rotate_image(left_img, la)
    right_rot = rotate_image(right_img, ra)
    left_c_rot = rotate_circle(left_circle, left_img.shape, -la)
    right_c_rot = rotate_circle(right_circle, right_img.shape, -ra)
    left_l_rot = rotate_edge_line(left_line, left_img.shape, -la)
    right_l_rot = rotate_edge_line(right_line, right_img.shape, -ra)

    # Report: rotated images
    fig_rot, axes_rot = plt.subplots(1, 2, figsize=(14, 6))
    for ax, img, circ, lbl, angle in [
        (axes_rot[0], left_rot, left_c_rot, f'Left (rot {la:.1f}°)', la),
        (axes_rot[1], right_rot, right_c_rot, f'Right (rot {ra:.1f}°)', ra),
    ]:
        ext = [0, img.shape[1] * px, img.shape[0] * px, 0]
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=config.vmax, extent=ext)
        ax.add_patch(
            plt.Circle((circ.x * px, circ.y * px),
                       circ.r * px,
                       color='blue',
                       fill=False,
                       lw=1))
        ax.set(title=lbl, xlabel='X [mm]', ylabel='Y [mm]')
        fig_rot.colorbar(im,
                         ax=ax,
                         fraction=0.046,
                         pad=0.04,
                         label='Intensity')
    fig_rot.tight_layout()
    report.add_figure(fig_rot,
                      caption='Step 3: Rotated to vertical edges',
                      source_paths=[str(data_path)])

    # 4. Crop centered -------------------------------------------------------
    cs = config.crop_size
    left_crop, left_off = crop_centered(left_rot, left_c_rot.x, left_c_rot.y,
                                        cs)
    right_crop, right_off = crop_centered(right_rot, right_c_rot.x,
                                          right_c_rot.y, cs)

    left_c_crop = Circle(x=left_c_rot.x - left_off[0],
                         y=left_c_rot.y - left_off[1],
                         r=left_c_rot.r)
    right_c_crop = Circle(x=right_c_rot.x - right_off[0],
                          y=right_c_rot.y - right_off[1],
                          r=right_c_rot.r)
    left_l_crop = transform_edge_line_for_crop(left_l_rot, left_off)
    right_l_crop = transform_edge_line_for_crop(right_l_rot, right_off)

    # 5. Gradient-based circle refinement ------------------------------------
    max_dist_px = config.circle_edge_distance_mm / px
    for side, img, circ in [('left', left_crop, left_c_crop),
                            ('right', right_crop, right_c_crop)]:
        pts = find_gradient_edge_points(img, config.circle_gradient_high,
                                        config.circle_gradient_low)
        filt = filter_edge_points_near_circle(pts, circ, max_dist_px)
        refined = fit_circle_to_points(filt, circ,
                                       config.circle_radius_tolerance)
        if side == 'left':
            left_c_final = refined
        else:
            right_c_final = refined

    # Report: cropped images with circles
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    vmax = config.vmax
    for ax, img, ic, rc, lbl in [
        (axes[0], left_crop, left_c_crop, left_c_final,
         f'Left (foil {config.left_foil_id})'),
        (axes[1], right_crop, right_c_crop, right_c_final,
         f'Right (foil {config.right_foil_id})'),
    ]:
        ext = [0, img.shape[1] * px, img.shape[0] * px, 0]
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, extent=ext)
        ax.add_patch(
            plt.Circle((ic.x * px, ic.y * px),
                       ic.r * px,
                       color='blue',
                       fill=False,
                       lw=1,
                       ls='--'))
        ax.add_patch(
            plt.Circle((rc.x * px, rc.y * px),
                       rc.r * px,
                       color='black',
                       fill=False,
                       lw=1))
        ax.set(title=f'{lbl}\nBlue=initial, Black=refined',
               xlabel='X [mm]',
               ylabel='Y [mm]')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
    fig.tight_layout()
    report.add_figure(
        fig,
        caption=
        'Step 5: Cropped foils — initial (blue dashed) vs refined (black) circles',
        source_paths=[str(data_path)])

    # 6. Vertical shift & initial merge --------------------------------------
    left_edge_x = np.mean([x for x, _ in left_l_crop.points])
    right_edge_x = np.mean([x for x, _ in right_l_crop.points])
    target_y = cs / 2.0
    left_sh, left_c_sh = _shift_image_vertically(left_crop, left_c_final,
                                                 target_y)
    right_sh, right_c_sh = _shift_image_vertically(right_crop, right_c_final,
                                                   target_y)

    merged, x_off = merge_images(left_sh, right_sh, left_edge_x, right_edge_x,
                                 config.merge_margin)
    left_c_m = left_c_sh
    right_c_m = Circle(x=right_c_sh.x + x_off, y=right_c_sh.y, r=right_c_sh.r)

    # Report: initial merged image
    fig_m, ax_m = plt.subplots(figsize=(14, 6))
    hm, wm = merged.shape
    im_m = ax_m.imshow(merged,
                       cmap=cmap,
                       vmin=0,
                       vmax=vmax,
                       extent=[0, wm * px, hm * px, 0])
    for c in (left_c_m, right_c_m):
        ax_m.add_patch(
            plt.Circle((c.x * px, c.y * px),
                       c.r * px,
                       color='black',
                       fill=False,
                       lw=1))
        ax_m.plot(c.x * px, c.y * px, 'k+', ms=10, mew=1.5)
    ax_m.set(title='Initial merge', xlabel='X [mm]', ylabel='Y [mm]')
    ax_m.grid(True, color='k', alpha=0.2, lw=0.5)
    fig_m.colorbar(im_m, ax=ax_m, fraction=0.03, pad=0.02, label='Intensity')
    fig_m.tight_layout()
    report.add_figure(fig_m,
                      caption='Step 6: Merged foils (initial margin)',
                      source_paths=[str(data_path)])

    # 7. Spline shift optimisation -------------------------------------------
    py_min = int(config.profile_y_min_mm / px)
    py_max = int(config.profile_y_max_mm / px)
    x_mm = np.arange(merged.shape[1]) * px
    prof_raw = np.nanmean(merged[py_min:py_max, :].astype(float), axis=0)
    prof = gaussian_filter1d(prof_raw, sigma=config.smoothing_sigma_px)

    m1 = (x_mm >= config.section1_x_min) & (x_mm <= config.section1_x_max)
    m2 = (x_mm >= config.section2_x_min) & (x_mm <= config.section2_x_max)
    x1, y1 = x_mm[m1], prof[m1]
    x2_orig, y2 = x_mm[m2], prof[m2]

    shifts = np.linspace(-2, 2, 401)
    rss_arr = np.array(
        [_fit_and_evaluate(x1, y1, x2_orig, y2, s)[1] for s in shifts])
    opt_idx = int(np.argmin(rss_arr))
    opt_shift = float(shifts[opt_idx])

    # Report: RSS vs shift
    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    ax_s.plot(shifts, rss_arr)
    ax_s.axvline(opt_shift,
                 color='r',
                 ls='--',
                 label=f'Optimal: {opt_shift:.2f} mm')
    ax_s.axvline(0, color='k', ls=':', alpha=0.5, label='No shift')
    ax_s.set(xlabel='Shift [mm]',
             ylabel='RSS',
             title='Spline alignment optimisation')
    ax_s.legend()
    ax_s.grid(True, alpha=0.3)
    fig_s.tight_layout()
    report.add_figure(
        fig_s, caption=f'Step 7: RSS vs shift — optimal {opt_shift:.2f} mm')

    # 8. Re-merge with optimised margin --------------------------------------
    opt_margin = int(round(opt_shift / px))
    merged_opt, x_off_opt = merge_images(left_sh, right_sh, left_edge_x,
                                         right_edge_x, opt_margin)
    left_c_mo = left_c_sh
    right_c_mo = Circle(x=right_c_sh.x + x_off_opt,
                        y=right_c_sh.y,
                        r=right_c_sh.r)

    # Report: optimised merged image
    fig_o, ax_o = plt.subplots(figsize=(14, 6))
    ho, wo = merged_opt.shape
    im_o = ax_o.imshow(merged_opt,
                       cmap=cmap,
                       vmin=0,
                       vmax=vmax,
                       extent=[0, wo * px, ho * px, 0])
    for c in (left_c_mo, right_c_mo):
        ax_o.add_patch(
            plt.Circle((c.x * px, c.y * px),
                       c.r * px,
                       color='black',
                       fill=False,
                       lw=1))
        ax_o.plot(c.x * px, c.y * px, 'k+', ms=10, mew=1.5)
    ax_o.set(
        title=
        f'Optimised merge (margin={opt_margin}px, shift={opt_shift:.2f}mm)',
        xlabel='X [mm]',
        ylabel='Y [mm]')
    ax_o.grid(True, color='k', alpha=0.2, lw=0.5)
    fig_o.colorbar(im_o, ax=ax_o, fraction=0.03, pad=0.02, label='Intensity')
    fig_o.tight_layout()
    report.add_figure(fig_o,
                      caption='Step 8: Merged foils (optimised margin)',
                      source_paths=[str(data_path)])

    # Profile comparison
    x_mm_opt = np.arange(merged_opt.shape[1]) * px
    prof_opt_raw = np.nanmean(merged_opt[py_min:py_max, :].astype(float),
                              axis=0)
    prof_opt = gaussian_filter1d(prof_opt_raw, sigma=config.smoothing_sigma_px)

    fig_p, ax_p = plt.subplots(figsize=(14, 6))
    x_mm_orig = np.arange(merged.shape[1]) * px
    prof_orig = gaussian_filter1d(np.nanmean(
        merged[py_min:py_max, :].astype(float), axis=0),
                                  sigma=config.smoothing_sigma_px)
    ax_p.plot(x_mm_orig, prof_orig, 'b--', lw=1, alpha=0.6, label='Original')
    ax_p.plot(x_mm_opt, prof_opt, 'r-', lw=1.5, label='Optimised')
    ax_p.set(xlabel='X [mm]',
             ylabel='Intensity',
             title='Profile comparison: original vs optimised')
    ax_p.legend()
    ax_p.grid(True, alpha=0.3)
    fig_p.tight_layout()
    report.add_figure(
        fig_p, caption='Step 9: Horizontal profile before/after optimisation')
    # 10. 2D Signal Visualization — isosignal contour lines ------------------
    fig_contour = _plot_isosignal_contours(
        merged_opt,
        left_c_mo,
        right_c_mo,
        px,
        cmap,
        vmax,
        title=
        f'Foils {config.left_foil_id} & {config.right_foil_id} — 2D Signal with Isosignal Lines'
    )
    report.add_figure(
        fig_contour,
        caption='Step 10: 2D signal with isosignal contour lines',
        source_paths=[str(data_path)])

    # 11. Horizontal profiles at different Y positions -----------------------
    fig_hprof = _plot_horizontal_profiles_at_y_positions(merged_opt,
                                                         left_c_mo,
                                                         right_c_mo,
                                                         px,
                                                         vmax=vmax)
    report.add_figure(
        fig_hprof,
        caption='Step 11: Horizontal profiles at circle centre ± 4 mm',
        source_paths=[str(data_path)])
    report.save()
    print(f"PDF saved to: {pdf_path}")

    # 9. Save outputs --------------------------------------------------------
    merged_npz = out / (
        config.output_merged_npz
        or f"merged_foils_{config.left_foil_id}_{config.right_foil_id}.npz")
    np.savez_compressed(
        merged_npz,
        merged_image=merged,
        merged_image_opt=merged_opt,
        left_circle_x=left_c_m.x,
        left_circle_y=left_c_m.y,
        left_circle_r=left_c_m.r,
        right_circle_x=right_c_m.x,
        right_circle_y=right_c_m.y,
        right_circle_r=right_c_m.r,
        left_circle_opt_x=left_c_mo.x,
        left_circle_opt_y=left_c_mo.y,
        left_circle_opt_r=left_c_mo.r,
        right_circle_opt_x=right_c_mo.x,
        right_circle_opt_y=right_c_mo.y,
        right_circle_opt_r=right_c_mo.r,
        merge_margin=config.merge_margin,
        optimized_merge_margin=opt_margin,
        optimal_shift_mm=opt_shift,
        left_foil_id=config.left_foil_id,
        right_foil_id=config.right_foil_id,
        pixel_size_mm=px,
    )

    profile_npz = out / (
        config.output_profile_npz
        or f"profile_foils_{config.left_foil_id}_{config.right_foil_id}.npz")
    np.savez_compressed(
        profile_npz,
        x_mm=x_mm_opt,
        profile_opt_smooth=prof_opt,
        profile_y_min_mm=config.profile_y_min_mm,
        profile_y_max_mm=config.profile_y_max_mm,
        smoothing_sigma_px=config.smoothing_sigma_px,
        pixel_size_mm=px,
    )
    print(f"Saved: {merged_npz}, {profile_npz}")

    return {
        'merged_image': merged,
        'merged_image_opt': merged_opt,
        'profile_x_mm': x_mm_opt,
        'profile_smooth': prof_opt,
        'optimal_shift_mm': opt_shift,
    }
