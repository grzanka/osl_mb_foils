"""Post-alignment MBO background-subtraction scenarios for March 2026 CCB foils.

This stage consumes aligned foil crops and explores two background-subtraction
approaches using the 03-17 and 03-19 datasets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.data.analysis import Circle
from src.pub2026.config import MBOBackgroundSubtractionConfig, resolve_file
from src.pub2026.pdf_report import PDFReport

# Custom colormap: white (0) → green (0.5×max) → red (max)
_WGR_CMAP = LinearSegmentedColormap.from_list(
    'white_green_red',
    [(0.0, (1.0, 1.0, 1.0)),
     (0.5, (0.0, 0.75, 0.0)),
     (1.0, (0.9, 0.0, 0.0))],
)


@dataclass
class _AlignedFoil:
    image: np.ndarray
    circle: Circle


def _load_aligned_npz(npz_path: Path) -> Dict[int, _AlignedFoil]:
    data = np.load(npz_path)
    foils: Dict[int, _AlignedFoil] = {}
    for key in data.files:
        if not key.startswith("foil_") or not key.endswith("_image"):
            continue
        foil_id = int(key.split("_")[1])
        foils[foil_id] = _AlignedFoil(
            image=data[key].astype(np.float64),
            circle=Circle(
                x=float(data[f"foil_{foil_id}_circle_x"]),
                y=float(data[f"foil_{foil_id}_circle_y"]),
                r=float(data[f"foil_{foil_id}_circle_r"]),
            ),
        )
    return foils


def _common_circle(circle_a: Circle, circle_b: Circle) -> Circle:
    return Circle(x=0.5 * (circle_a.x + circle_b.x),
                  y=0.5 * (circle_a.y + circle_b.y),
                  r=0.5 * (circle_a.r + circle_b.r))


def _shift_to_circle(image: np.ndarray, circle: Circle,
                     target: Circle) -> Tuple[np.ndarray, Circle]:
    shift_y = target.y - circle.y
    shift_x = target.x - circle.x
    shifted = ndi.shift(image.astype(np.float64),
                        shift=(shift_y, shift_x),
                        order=1,
                        mode='constant',
                        cval=np.nan,
                        prefilter=False)
    return shifted, target


def _radius_mask(shape: Tuple[int, int], circle: Circle,
                 radius_fraction: float) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    radius = circle.r * radius_fraction
    return ((xx - circle.x)**2 + (yy - circle.y)**2) <= radius**2


def _circle_mean(image: np.ndarray, circle: Circle,
                 radius_fraction: float) -> float:
    mask = _radius_mask(image.shape, circle, radius_fraction)
    return float(np.nanmean(image[mask]))


def _relative_axes(image: np.ndarray, circle: Circle,
                   px: float) -> Tuple[List[float], np.ndarray, np.ndarray]:
    height, width = image.shape
    extent = [(-circle.x) * px, (width - circle.x) * px,
              (height - circle.y) * px, (-circle.y) * px]
    x_arr = np.linspace(extent[0], extent[1], width)
    y_arr = np.linspace(extent[3], extent[2], height)
    return extent, x_arr, y_arr


def _plot_image(ax: plt.Axes,
                image: np.ndarray,
                circle: Circle,
                px: float,
                title: str,
                contour_levels: Optional[Iterable[float]] = None,
                cmap: str = 'viridis',
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                norm: Optional[TwoSlopeNorm] = None,
                draw_radius_fraction: float = 1.0) -> None:
    extent, x_arr, y_arr = _relative_axes(image, circle, px)
    masked = np.ma.masked_invalid(image)
    im = ax.imshow(masked,
                   extent=extent,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   norm=norm)
    radius = circle.r * draw_radius_fraction * px
    ax.add_patch(
        plt.Circle((0.0, 0.0), radius, color='white', fill=False, lw=1.0,
                   ls='--'))
    if contour_levels:
        contours = ax.contour(x_arr,
                              y_arr,
                              masked,
                              levels=list(contour_levels),
                              colors='white',
                              linewidths=0.6)
        ax.clabel(contours, inline=True, fontsize=6, fmt='%.2f')
    ax.set(xlabel='X relative to center [mm]',
           ylabel='Y relative to center [mm]',
           title=title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _safe_nanmean(data: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(data)
    counts = valid.sum(axis=axis)
    sums = np.where(valid, data, 0.0).sum(axis=axis)
    result = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def _clamp_nonneg(image: np.ndarray) -> np.ndarray:
    """Return a copy of image with negative values replaced by 0."""
    result = image.copy()
    finite = np.isfinite(result)
    result[finite & (result < 0.0)] = 0.0
    return result



def _extract_profiles(image: np.ndarray,
                      circle: Circle,
                      px: float,
                      strip_half_width_px: int) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    center_y = int(round(circle.y))
    center_x = int(round(circle.x))

    y0 = max(0, center_y - strip_half_width_px)
    y1 = min(image.shape[0], center_y + strip_half_width_px + 1)
    horizontal = _safe_nanmean(image[y0:y1, :].astype(np.float64), axis=0)
    horizontal_x = (np.arange(image.shape[1]) - circle.x) * px

    x0 = max(0, center_x - strip_half_width_px)
    x1 = min(image.shape[1], center_x + strip_half_width_px + 1)
    vertical = _safe_nanmean(image[:, x0:x1].astype(np.float64), axis=1)
    vertical_y = (np.arange(image.shape[0]) - circle.y) * px
    return horizontal_x, horizontal, vertical_y, vertical


def _plot_profile(ax: plt.Axes,
                  positions: np.ndarray,
                  values: np.ndarray,
                  title: str,
                  axis_label: str,
                  reference_line: Optional[float] = None) -> None:
    ax.plot(positions, values, lw=1.2)
    ax.axvline(0.0, color='black', lw=0.8, ls='--', alpha=0.5)
    if reference_line is not None:
        ax.axhline(reference_line, color='tab:red', lw=0.8, ls=':')
    ax.set(xlabel=axis_label, ylabel='Signal', title=title)


def _plot_profiles_combined(ax: plt.Axes,
                            hx: np.ndarray,
                            hv: np.ndarray,
                            vx: np.ndarray,
                            vv: np.ndarray,
                            title: str,
                            reference_line: Optional[float] = None) -> None:
    """Plot horizontal and vertical profiles as two series on one axes."""
    ax.plot(hx, hv, lw=1.2, label='Horizontal')
    ax.plot(vx, vv, lw=1.2, label='Vertical')
    ax.axvline(0.0, color='black', lw=0.8, ls='--', alpha=0.5)
    if reference_line is not None:
        ax.axhline(reference_line, color='tab:red', lw=0.8, ls=':')
    ax.set(xlabel='Distance from center [mm]', ylabel='Signal', title=title)
    ax.legend()


def _plot_background_reference_rows(reference_label: str,
                                    foils: Dict[int, _AlignedFoil],
                                    foil_ids: List[int],
                                    means: Dict[int, float],
                                    px: float,
                                    strip_half_width_px: int,
                                    radius_fraction: float) -> plt.Figure:
    fig, axes = plt.subplots(len(foil_ids),
                             3,
                             figsize=(15, 5 * len(foil_ids)),
                             squeeze=False)
    for row, foil_id in enumerate(foil_ids):
        foil = foils[foil_id]
        _plot_image(axes[row][0],
                    foil.image,
                    foil.circle,
                    px,
                    title=(f'{reference_label} foil {foil_id}\n'
                           f'mean inside {radius_fraction:.0%} r = {means[foil_id]:.3f}'),
                    draw_radius_fraction=radius_fraction)
        hx, hv, vx, vv = _extract_profiles(foil.image, foil.circle, px,
                                           strip_half_width_px)
        _plot_profile(axes[row][1], hx, hv,
                      title=f'Foil {foil_id} horizontal profile',
                      axis_label='X relative to center [mm]')
        _plot_profile(axes[row][2], vx, vv,
                      title=f'Foil {foil_id} vertical profile',
                      axis_label='Y relative to center [mm]')
    fig.tight_layout()
    return fig


def _scenario_a_diff_norm(diff_img: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (vmax, clamped_diff) for scenario A difference image."""
    clamped = _clamp_nonneg(diff_img)
    finite = clamped[np.isfinite(clamped)]
    vmax = float(np.percentile(finite, 99)) if finite.size > 0 else 1.0
    return max(vmax, 1.0), clamped


def _plot_scenario_a_page1(reference_label: str,
                            target_label: str,
                            rows: Dict[int, Dict[str, object]],
                            px: float,
                            contour_levels: List[float]) -> plt.Figure:
    """Page 1: original foils side-by-side with the subtraction result."""
    foil_ids = sorted(rows)
    fig, axes = plt.subplots(len(foil_ids), 3,
                             figsize=(18, 5 * len(foil_ids)),
                             squeeze=False)
    for row, foil_id in enumerate(foil_ids):
        result = rows[foil_id]
        circle = result['circle']
        vmax, diff_img = _scenario_a_diff_norm(result['difference'])
        _plot_image(axes[row][0],
                    result['reference_centered'],
                    circle, px,
                    title=f'{reference_label} foil {foil_id}',
                    cmap='magma')
        _plot_image(axes[row][1],
                    result['target_centered'],
                    circle, px,
                    title=f'{target_label} foil {foil_id}',
                    cmap='magma')
        _plot_image(axes[row][2],
                    diff_img,
                    circle, px,
                    title=f'Diff ({target_label} \u2212 {reference_label}), foil {foil_id}',
                    contour_levels=[l for l in contour_levels if l > 0],
                    cmap=_WGR_CMAP, vmin=0.0, vmax=vmax)
    fig.suptitle('Scenario A \u2014 Page 1: foils to be subtracted and subtraction result',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def _plot_scenario_a_page2(reference_label: str,
                            target_label: str,
                            rows: Dict[int, Dict[str, object]],
                            px: float,
                            strip_half_width_px: int,
                            contour_levels: List[float]) -> plt.Figure:
    """Page 2: subtraction result and combined H+V profiles."""
    foil_ids = sorted(rows)
    fig, axes = plt.subplots(len(foil_ids), 2,
                             figsize=(12, 5 * len(foil_ids)),
                             squeeze=False)
    for row, foil_id in enumerate(foil_ids):
        result = rows[foil_id]
        circle = result['circle']
        vmax, diff_img = _scenario_a_diff_norm(result['difference'])
        _plot_image(axes[row][0],
                    diff_img,
                    circle, px,
                    title=f'Diff ({target_label} \u2212 {reference_label}), foil {foil_id}',
                    contour_levels=[l for l in contour_levels if l > 0],
                    cmap=_WGR_CMAP, vmin=0.0, vmax=vmax)
        hx, hv, vx, vv = _extract_profiles(diff_img, circle, px,
                                            strip_half_width_px)
        _plot_profiles_combined(axes[row][1], hx, hv, vx, vv,
                                title=f'Foil {foil_id} diff profiles',
                                reference_line=0.0)
    fig.suptitle('Scenario A \u2014 Page 2: subtraction result and profiles',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def _plot_bg_subtracted_rows(dataset_label: str,
                             rows: Dict[int, Dict[str, object]],
                             px: float,
                             strip_half_width_px: int) -> plt.Figure:
    foil_ids = sorted(rows)
    fig, axes = plt.subplots(len(foil_ids),
                             2,
                             figsize=(12, 5 * len(foil_ids)),
                             squeeze=False)
    for row, foil_id in enumerate(foil_ids):
        result = rows[foil_id]
        image = _clamp_nonneg(result['image'])
        circle = result['circle']
        finite = image[np.isfinite(image)]
        vmax = float(np.percentile(finite, 99)) if finite.size > 0 else 1.0
        vmax = max(vmax, 1.0)
        _plot_image(axes[row][0],
                    image,
                    circle,
                    px,
                    title=f'{dataset_label} foil {foil_id} after bg subtraction',
                    cmap=_WGR_CMAP, vmin=0.0, vmax=vmax)
        hx, hv, vx, vv = _extract_profiles(image, circle, px,
                                           strip_half_width_px)
        _plot_profiles_combined(axes[row][1], hx, hv, vx, vv,
                                title=f'Foil {foil_id} profiles after bg subtraction',
                                reference_line=0.0)
    fig.tight_layout()
    return fig


def _two_slope_around_one(image: np.ndarray) -> Optional[TwoSlopeNorm]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return None
    p01 = float(np.percentile(finite, 1))
    p99 = float(np.percentile(finite, 99))
    if p01 < 1.0 < p99:
        return TwoSlopeNorm(vcenter=1.0, vmin=p01, vmax=p99)
    return None



def _plot_scenario_b_rows(reference_label: str,
                          target_label: str,
                          rows: Dict[int, Dict[str, object]],
                          px: float,
                          strip_half_width_px: int,
                          normalized_contours: List[float],
                          ratio_contours: List[float]) -> plt.Figure:
    foil_ids = sorted(rows)
    fig, axes = plt.subplots(len(foil_ids),
                             5,
                             figsize=(24, 5 * len(foil_ids)),
                             squeeze=False)
    ref_norm = TwoSlopeNorm(vcenter=1.0, vmin=0.9, vmax=1.1)
    for row, foil_id in enumerate(foil_ids):
        result = rows[foil_id]
        circle = result['circle']
        ratio_img = result['ratio']
        ratio_norm = _two_slope_around_one(ratio_img)

        ref_bg_sub = _clamp_nonneg(result['reference_bg_sub_centered'])
        tgt_bg_sub = _clamp_nonneg(result['target_bg_sub_centered'])
        ref_finite = ref_bg_sub[np.isfinite(ref_bg_sub)]
        tgt_finite = tgt_bg_sub[np.isfinite(tgt_bg_sub)]
        all_finite = np.concatenate([ref_finite, tgt_finite])
        bg_vmax = float(np.percentile(all_finite, 99)) if all_finite.size > 0 else 1.0
        bg_vmax = max(bg_vmax, 1.0)

        _plot_image(axes[row][0],
                    ref_bg_sub,
                    circle, px,
                    title=f'{reference_label} foil {foil_id} bg-sub',
                    cmap=_WGR_CMAP, vmin=0.0, vmax=bg_vmax)
        _plot_image(
            axes[row][1],
            result['reference_normalized_smoothed'],
            circle, px,
            title=(f'{reference_label} foil {foil_id} normalized×smoothed\n'
                   f'mean ≤ {result["normalization_radius_fraction"]:.0%} r = '
                   f'{result["normalization_mean"]:.3f}, σ={result["smoothing_sigma_px"]:.1f}px'),
            contour_levels=normalized_contours,
            cmap='coolwarm',
            norm=ref_norm)
        _plot_image(axes[row][2],
                    tgt_bg_sub,
                    circle, px,
                    title=f'{target_label} foil {foil_id} bg-sub',
                    cmap=_WGR_CMAP, vmin=0.0, vmax=bg_vmax)
        _plot_image(axes[row][3],
                    ratio_img,
                    circle, px,
                    title=f'{target_label} / norm×smoothed {reference_label}\nfoil {foil_id}',
                    contour_levels=ratio_contours,
                    cmap='coolwarm',
                    norm=ratio_norm)
        hx, hv, vx, vv = _extract_profiles(ratio_img, circle, px,
                                            strip_half_width_px)
        _plot_profiles_combined(axes[row][4], hx, hv, vx, vv,
                                title=f'Foil {foil_id} ratio profiles',
                                reference_line=1.0)
    fig.suptitle(f'Scenario B — Ratio: {target_label} / normalized {reference_label}',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def explore_background_subtraction(
        config: MBOBackgroundSubtractionConfig,
        output_dir: str = '.',
        pdf_path: Optional[str] = None) -> Dict[str, object]:
    """Run the post-alignment MBO background-subtraction exploration."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference_npz = resolve_file(config.reference_npz, out)
    target_npz = resolve_file(config.target_npz, out)
    if pdf_path is None:
        pdf_path = str(out / f'mbo_background_subtraction_{config.facility}.pdf')

    report = PDFReport(
        pdf_path,
        title=f'MBO Background Subtraction: {config.facility.upper()}',
        config_path=str(reference_npz),
    )

    reference_foils = _load_aligned_npz(reference_npz)
    target_foils = _load_aligned_npz(target_npz)
    px = config.pixel_size_mm

    report.add_text(
        f'Reference aligned NPZ: {reference_npz}\n'
        f'Target aligned NPZ: {target_npz}\n\n'
        f'Reference label: {config.reference_label}\n'
        f'Target label: {config.target_label}\n'
        f'Reference foils: {config.reference_foil_ids}\n'
        f'Background foils: {config.background_foil_ids}\n'
        f'Background radius fraction: {config.background_radius_fraction}\n'
        f'Normalization radius fraction: {config.normalization_radius_fraction}\n'
        f'Smoothing sigma: {config.smoothing_sigma_px} px\n'
        f'Scenario A contour levels: {config.scenario_a_contour_levels}\n'
        f'Normalized contour levels: {config.normalized_contour_levels}\n'
        f'Ratio contour levels: {config.ratio_contour_levels}',
        title='Inputs and parameters',
        source_paths=[str(reference_npz), str(target_npz)])

    background_means = {
        foil_id: _circle_mean(reference_foils[foil_id].image,
                              reference_foils[foil_id].circle,
                              config.background_radius_fraction)
        for foil_id in config.background_foil_ids
    }
    background_level = float(np.mean(list(background_means.values())))

    # Computed here, added to the report in the Scenario B section
    fig_bg_ref = _plot_background_reference_rows(
        config.reference_label,
        reference_foils,
        config.background_foil_ids,
        background_means,
        px,
        config.profile_strip_half_width_px,
        config.background_radius_fraction)

    scenario_a_rows: Dict[int, Dict[str, object]] = {}
    scenario_b_rows: Dict[int, Dict[str, object]] = {}
    save_dict: Dict[str, object] = {'background_level': background_level}

    for foil_id in config.reference_foil_ids:
        reference_foil = reference_foils[foil_id]
        target_foil = target_foils[foil_id]
        common_circle = _common_circle(reference_foil.circle, target_foil.circle)
        ref_centered, common_circle = _shift_to_circle(reference_foil.image,
                                                       reference_foil.circle,
                                                       common_circle)
        tgt_centered, common_circle = _shift_to_circle(target_foil.image,
                                                       target_foil.circle,
                                                       common_circle)

        diff_image = tgt_centered - ref_centered
        scenario_a_rows[foil_id] = {
            'reference_centered': ref_centered,
            'target_centered': tgt_centered,
            'difference': diff_image,
            'circle': common_circle,
        }
        save_dict[f'scenario_a_foil_{foil_id}_reference_centered'] = ref_centered
        save_dict[f'scenario_a_foil_{foil_id}_target_centered'] = tgt_centered
        save_dict[f'scenario_a_foil_{foil_id}_difference'] = diff_image

        ref_bg_sub = reference_foil.image - background_level
        tgt_bg_sub = target_foil.image - background_level
        ref_bg_sub_centered, common_circle = _shift_to_circle(
            ref_bg_sub, reference_foil.circle, common_circle)
        tgt_bg_sub_centered, common_circle = _shift_to_circle(
            tgt_bg_sub, target_foil.circle, common_circle)

        normalization_mean = _circle_mean(ref_bg_sub_centered, common_circle,
                                          config.normalization_radius_fraction)
        normalized = ref_bg_sub_centered / normalization_mean
        normalized_smoothed = ndi.gaussian_filter(normalized,
                                                  sigma=config.smoothing_sigma_px)
        ratio = np.full_like(tgt_bg_sub_centered, np.nan, dtype=np.float64)
        valid = np.isfinite(normalized_smoothed) & (
            np.abs(normalized_smoothed) >= config.minimum_divisor)
        np.divide(tgt_bg_sub_centered,
                  normalized_smoothed,
                  out=ratio,
                  where=valid)
        ratio[~_radius_mask(ratio.shape, common_circle,
                            config.ratio_mask_radius_fraction)] = np.nan

        scenario_b_rows[foil_id] = {
            'reference_bg_sub_centered': ref_bg_sub_centered,
            'target_bg_sub_centered': tgt_bg_sub_centered,
            'reference_normalized_smoothed': normalized_smoothed,
            'ratio': ratio,
            'circle': common_circle,
            'normalization_mean': normalization_mean,
            'normalization_radius_fraction': config.normalization_radius_fraction,
            'smoothing_sigma_px': config.smoothing_sigma_px,
        }
        save_dict[f'scenario_b_foil_{foil_id}_reference_bg_sub_centered'] = ref_bg_sub_centered
        save_dict[f'scenario_b_foil_{foil_id}_target_bg_sub_centered'] = tgt_bg_sub_centered
        save_dict[f'scenario_b_foil_{foil_id}_reference_normalized'] = normalized
        save_dict[f'scenario_b_foil_{foil_id}_reference_normalized_smoothed'] = normalized_smoothed
        save_dict[f'scenario_b_foil_{foil_id}_ratio'] = ratio
        save_dict[f'scenario_b_foil_{foil_id}_circle_x'] = common_circle.x
        save_dict[f'scenario_b_foil_{foil_id}_circle_y'] = common_circle.y
        save_dict[f'scenario_b_foil_{foil_id}_circle_r'] = common_circle.r
        save_dict[f'scenario_b_foil_{foil_id}_normalization_mean'] = normalization_mean

    # ── Scenario A ────────────────────────────────────────────────────────
    report.add_text(
        f'Scenario A: raw signal subtraction\n\n'
        f'{config.target_label} foil \u2212 {config.reference_label} foil, '
        f'both centered on their common circle.\n\n'
        f'Foils: {config.reference_foil_ids}\n'
        f'Contour levels (positive only): '
        f'{[l for l in config.scenario_a_contour_levels if l > 0]}',
        title='\u2501\u2501\u2501  SCENARIO A  \u2501\u2501\u2501',
        source_paths=[str(reference_npz), str(target_npz)])

    fig_a1 = _plot_scenario_a_page1(config.reference_label,
                                    config.target_label,
                                    scenario_a_rows,
                                    px,
                                    config.scenario_a_contour_levels)
    report.add_figure(fig_a1,
                      caption=('Scenario A \u2014 Page 1: foils to be subtracted and subtraction result\n'
                               f'({config.target_label} \u2212 {config.reference_label}) '
                               f'for foils {config.reference_foil_ids}'),
                      source_paths=[str(reference_npz), str(target_npz)])
    plt.close(fig_a1)

    fig_a2 = _plot_scenario_a_page2(config.reference_label,
                                    config.target_label,
                                    scenario_a_rows,
                                    px,
                                    config.profile_strip_half_width_px,
                                    config.scenario_a_contour_levels)
    report.add_figure(fig_a2,
                      caption=('Scenario A \u2014 Page 2: subtraction result and profiles\n'
                               f'({config.target_label} \u2212 {config.reference_label}) '
                               f'for foils {config.reference_foil_ids}'),
                      source_paths=[str(reference_npz), str(target_npz)])
    plt.close(fig_a2)

    # ── Scenario B ────────────────────────────────────────────────────────
    report.add_text(
        f'Scenario B: background subtraction using a fixed scalar level\n\n'
        f'Background level = average of means from '
        f'{config.reference_label} foils {config.background_foil_ids} '
        f'inside {config.background_radius_fraction:.0%} of radius.\n\n'
        + '\n'.join(
            f'  foil {foil_id}: mean \u2264 '
            f'{config.background_radius_fraction:.0%} r = {background_means[foil_id]:.6f}'
            for foil_id in config.background_foil_ids)
        + f'\n\nBackground level used for subtraction: {background_level:.6f}\n\n'
        f'Steps:\n'
        f'  1. Subtract {background_level:.4f} from all foils.\n'
        f'  2. Normalize reference foils (divide by mean inside '
        f'{config.normalization_radius_fraction:.0%} r).\n'
        f'  3. Apply Gaussian smoothing (\u03c3 = {config.smoothing_sigma_px} px).\n'
        f'  4. Divide target foil by the normalized, smoothed reference foil.',
        title='\u2501\u2501\u2501  SCENARIO B  \u2501\u2501\u2501',
        source_paths=[str(reference_npz), str(target_npz)])

    report.add_figure(fig_bg_ref,
                      caption='Scenario B: background-source foils used to estimate the fixed scalar background level',
                      source_paths=[str(reference_npz)])
    plt.close(fig_bg_ref)

    reference_bg_rows = {
        foil_id: {
            'image': reference_foils[foil_id].image - background_level,
            'circle': reference_foils[foil_id].circle,
        }
        for foil_id in config.reference_foil_ids
    }
    fig_ref_bg = _plot_bg_subtracted_rows(config.reference_label,
                                          reference_bg_rows,
                                          px,
                                          config.profile_strip_half_width_px)
    report.add_figure(fig_ref_bg,
                      caption=(f'Scenario B: {config.reference_label} background-subtracted foils '
                               f'{config.reference_foil_ids} (fixed scalar background level)'),
                      source_paths=[str(reference_npz)])
    plt.close(fig_ref_bg)

    target_bg_rows = {
        foil_id: {
            'image': target_foils[foil_id].image - background_level,
            'circle': target_foils[foil_id].circle,
        }
        for foil_id in sorted(target_foils)
    }
    for foil_id, row in target_bg_rows.items():
        save_dict[f'scenario_b_target_foil_{foil_id}_bg_sub'] = row['image']
    fig_tgt_bg = _plot_bg_subtracted_rows(config.target_label,
                                          target_bg_rows,
                                          px,
                                          config.profile_strip_half_width_px)
    report.add_figure(fig_tgt_bg,
                      caption=(f'Scenario B: {config.target_label} background-subtracted foils '
                               '(same fixed scalar background level)'),
                      source_paths=[str(target_npz)])
    plt.close(fig_tgt_bg)

    fig_b = _plot_scenario_b_rows(config.reference_label,
                                  config.target_label,
                                  scenario_b_rows,
                                  px,
                                  config.profile_strip_half_width_px,
                                  config.normalized_contour_levels,
                                  config.ratio_contour_levels)
    report.add_figure(fig_b,
                      caption=('Scenario B: normalized\u00d7smoothed reference foils from '
                               f'{config.reference_label}, ratio with '
                               f'{config.target_label} foils'),
                      source_paths=[str(reference_npz), str(target_npz)])
    plt.close(fig_b)

    normalization_lines = '\n'.join(
        f'foil {foil_id}: normalization mean \u2264 {config.normalization_radius_fraction:.0%} r = '
        f'{scenario_b_rows[foil_id]["normalization_mean"]:.6f}'
        for foil_id in sorted(scenario_b_rows))
    report.add_text(
        normalization_lines +
        f'\n\nFinal ratio images masked outside {config.ratio_mask_radius_fraction:.0%} of the aligned foil radius.',
        title='Scenario B normalization details',
        source_paths=[str(reference_npz), str(target_npz)])

    report.save()
    print(f'PDF report saved to: {pdf_path}')

    npz_name = (
        config.output_npz
        or f'mbo_background_subtraction_{config.facility}.npz')
    output_npz = out / npz_name
    np.savez_compressed(output_npz, **save_dict)
    print(f'Background subtraction data saved to: {output_npz}')

    return {
        'background_level': background_level,
        'scenario_a': scenario_a_rows,
        'scenario_b': scenario_b_rows,
        'output_npz': output_npz,
        'pdf_path': pdf_path,
    }
