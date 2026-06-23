"""EBT film analysis: process TIFF pairs, dose calculation, pencil mark detection.

Ported from notebooks: 1.0_ebt_aic144_analysis.ipynb, 1.0_ebt_ccb_analysis.ipynb
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Circle
from scipy import ndimage

from src.ebt import (
    FilePairData,
    auto_crop_dose,
    ebt3_dose_Gy,
    get_px_to_mm,
    get_red_channel,
    get_tiff_dpi,
    rotate_dose,
)
from src.ebt.dosimetry import DEFAULT_BG_G, DEFAULT_BG_B
from src.pub2026.config import EBTAnalysisConfig
from src.pub2026.pdf_report import PDFReport
from src.visualise.dose import (
    create_dose_colormap,
    plot_dose_with_background,
    plot_raw_signal_comparison,
)


@dataclass
class PencilMark:
    center_x_mm: float
    center_y_mm: float
    area_px: int
    label_id: int


@dataclass
class PencilMarkLine:
    a: float
    b: float
    angle_deg: float
    n_points: int
    y_range_mm: Tuple[float, float]


def _load_pair(signal_path: Path, bg_path: Path) -> FilePairData:
    """Load a signal/background TIFF pair and convert to dose."""
    import tifffile

    signal_dpi = get_tiff_dpi(signal_path)
    signal_px_to_mm = get_px_to_mm(signal_dpi)
    signal_raw = tifffile.imread(signal_path)

    bg_raw = tifffile.imread(bg_path)

    dose_signal = ebt3_dose_Gy(signal_raw)
    dose_bg = ebt3_dose_Gy(bg_raw)

    return FilePairData(
        signal_path=signal_path,
        background_path=bg_path,
        dpi=signal_dpi,
        px_to_mm=signal_px_to_mm,
        signal_raw=signal_raw,
        background_raw=bg_raw,
        dose_full=dose_signal,
        background_dose_full=dose_bg,
        shape=signal_raw.shape,
    )


def _crop_and_rotate(pair: FilePairData,
                     config: EBTAnalysisConfig) -> FilePairData:
    """Auto-crop and rotate a pair's dose (and background) arrays."""
    px_to_mm = pair.px_to_mm

    dose_cropped, bbox = auto_crop_dose(
        pair.dose_full,
        px_to_mm=px_to_mm,
        dose_threshold_gy=config.crop_dose_threshold_gy,
        smooth_sigma_mm=config.crop_smooth_sigma_mm,
        min_diameter_mm=config.crop_min_diameter_mm,
        margin_mm=config.crop_margin_mm,
    )

    y_start, y_end, x_start, x_end = bbox

    signal_red = get_red_channel(pair.signal_raw)
    signal_raw_cropped = signal_red[y_start:y_end, x_start:x_end]

    bg_red = get_red_channel(pair.background_raw)
    if pair.background_raw.shape[:2] == pair.signal_raw.shape[:2]:
        bg_raw_cropped = bg_red[y_start:y_end, x_start:x_end]
    else:
        bg_raw_cropped = bg_red

    if pair.background_dose_full.shape == pair.dose_full.shape:
        bg_dose_cropped = pair.background_dose_full[y_start:y_end,
                                                    x_start:x_end]
    else:
        bg_dose_cropped = pair.background_dose_full

    k = config.rotation_k
    dose_cropped = rotate_dose(dose_cropped, k=k)
    signal_raw_cropped = rotate_dose(signal_raw_cropped, k=k)
    bg_raw_cropped = rotate_dose(bg_raw_cropped, k=k)
    bg_dose_cropped = rotate_dose(bg_dose_cropped, k=k)

    pair.dose_cropped = dose_cropped
    pair.signal_cropped = signal_raw_cropped
    pair.background_cropped = bg_raw_cropped
    pair.background_dose_cropped = bg_dose_cropped
    pair.crop_bbox = bbox

    return pair


def _circular_mask(height: int, width: int, radius_px: float) -> npt.NDArray:
    """Create a circular boolean mask centred in an image."""
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    return np.sqrt((y - cy)**2 + (x - cx)**2) <= radius_px


def _bg_stats(bg_raw_cropped: npt.NDArray, mask: npt.NDArray) -> Dict:
    vals = bg_raw_cropped[mask]
    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals))
    return {
        "mean": mean_val,
        "std": std_val,
        "std_pct": (std_val / mean_val * 100) if mean_val else 0.0,
        "n_pixels": int(np.sum(mask))
    }


def _recalibrate(pair: FilePairData, bg_mean_red: float,
                 config: EBTAnalysisConfig) -> npt.NDArray:
    """Recalculate dose using measured background value."""
    new_bg = (bg_mean_red, DEFAULT_BG_G, DEFAULT_BG_B)
    dose_full_recal = ebt3_dose_Gy(pair.signal_raw, bg_values=new_bg)

    y_start, y_end, x_start, x_end = pair.crop_bbox
    dose_cropped = dose_full_recal[y_start:y_end, x_start:x_end]
    dose_cropped = rotate_dose(dose_cropped, k=config.rotation_k)
    return dose_cropped


def _detect_pencil_marks(
    dose_array: npt.NDArray, px_to_mm: float, config: EBTAnalysisConfig
) -> Tuple[Optional[PencilMark], Optional[PencilMark]]:
    """Detect pencil marks in two expected regions."""
    height, width = dose_array.shape
    binary_mask = dose_array > config.pencil_threshold_gy
    labeled_array, _ = ndimage.label(binary_mask)

    regions = [
        (config.pencil_mark1_y_range_mm, config.pencil_mark1_x_range_mm),
        (config.pencil_mark2_y_range_mm, config.pencil_mark2_x_range_mm),
    ]

    marks = []
    for y_range_mm, x_range_mm in regions:
        y_start_px = max(0, int(y_range_mm[0] / px_to_mm))
        y_end_px = min(height, int(y_range_mm[1] / px_to_mm))
        x_start_px = max(0, int(x_range_mm[0] / px_to_mm))
        x_end_px = min(width, int(x_range_mm[1] / px_to_mm))

        region_labels = labeled_array[y_start_px:y_end_px, x_start_px:x_end_px]
        unique_labels = np.unique(region_labels)
        unique_labels = unique_labels[unique_labels > 0]

        best_mark, best_area = None, 0
        for lid in unique_labels:
            blob = labeled_array == lid
            area = int(np.sum(blob))
            if area < 5:
                continue
            ys, xs = np.where(blob)
            cx_mm = float(np.mean(xs)) * px_to_mm
            cy_mm = float(np.mean(ys)) * px_to_mm
            if (x_range_mm[0] <= cx_mm <= x_range_mm[1]
                    and y_range_mm[0] <= cy_mm <= y_range_mm[1]):
                if area > best_area:
                    best_area = area
                    best_mark = PencilMark(cx_mm, cy_mm, area, lid)
        marks.append(best_mark)

    return marks[0], marks[1]


def _fit_pencil_line(dose_array: npt.NDArray, px_to_mm: float,
                     config: EBTAnalysisConfig) -> Optional[PencilMarkLine]:
    """Fit X = a*Y + b through both pencil mark regions, weighted by dose."""
    height, width = dose_array.shape
    regions = [
        (config.pencil_mark1_y_range_mm, config.pencil_mark1_x_range_mm),
        (config.pencil_mark2_y_range_mm, config.pencil_mark2_x_range_mm),
    ]

    all_y, all_x, all_w = [], [], []
    for y_range_mm, x_range_mm in regions:
        y0 = max(0, int(y_range_mm[0] / px_to_mm))
        y1 = min(height, int(y_range_mm[1] / px_to_mm))
        x0 = max(0, int(x_range_mm[0] / px_to_mm))
        x1 = min(width, int(x_range_mm[1] / px_to_mm))

        region = dose_array[y0:y1, x0:x1]
        yi, xi = np.where(region > config.line_fit_dose_threshold_gy)
        if len(yi) > 0:
            all_y.extend((yi + y0) * px_to_mm)
            all_x.extend((xi + x0) * px_to_mm)
            all_w.extend(region[yi, xi])

    if len(all_y) < 10:
        return None

    y_arr, x_arr, w_arr = np.array(all_y), np.array(all_x), np.array(all_w)
    coeffs = np.polyfit(y_arr, x_arr, 1, w=w_arr)
    a, b = coeffs[0], coeffs[1]
    return PencilMarkLine(a=a,
                          b=b,
                          angle_deg=float(np.degrees(np.arctan(a))),
                          n_points=len(y_arr),
                          y_range_mm=(float(y_arr.min()), float(y_arr.max())))


# --- Plotting helpers ---


def _plot_dose_map_with_profiles(dose: npt.NDArray, px_to_mm: float, key: str,
                                 hw: int, cmap_threshold: float) -> plt.Figure:
    height, width = dose.shape
    vmax = np.percentile(dose[dose > 0], 99) if (dose
                                                 > 0).any() else dose.max()
    cmap = create_dose_colormap(cmap_threshold)

    fig = plt.figure(figsize=(16, 5))
    ax1 = plt.subplot(1, 3, 1)
    extent = [0, width * px_to_mm, height * px_to_mm, 0]
    im = ax1.imshow(dose,
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    extent=extent,
                    aspect='equal')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title(f'Recalibrated Dose - {key}')
    plt.colorbar(im, ax=ax1, label='Dose [Gy]', fraction=0.046)

    y_pos, x_pos = height // 2, width // 2
    ax1.axhline(y=y_pos * px_to_mm, color='blue', ls='--', lw=1.5, alpha=0.8)
    ax1.axvline(x=x_pos * px_to_mm, color='red', ls='--', lw=1.5, alpha=0.8)

    ax2 = plt.subplot(1, 3, 2)
    h_prof = dose[max(0, y_pos - hw):min(height, y_pos + hw), :].mean(axis=0)
    ax2.plot(np.arange(len(h_prof)) * px_to_mm, h_prof, 'b-', lw=2)
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Dose [Gy]')
    ax2.set_title(f'Horizontal Profile')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(1, 3, 3)
    v_prof = dose[:, max(0, x_pos - hw):min(width, x_pos + hw)].mean(axis=1)
    ax3.plot(np.arange(len(v_prof)) * px_to_mm, v_prof, 'r-', lw=2)
    ax3.set_xlabel('Y [mm]')
    ax3.set_ylabel('Dose [Gy]')
    ax3.set_title(f'Vertical Profile')
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_dose_with_marks(dose: npt.NDArray, px_to_mm: float,
                          mark1: Optional[PencilMark],
                          mark2: Optional[PencilMark],
                          line: Optional[PencilMarkLine],
                          cmap_threshold: float, key: str) -> plt.Figure:
    height, width = dose.shape
    vmax = np.percentile(dose[dose > 0], 99) if (dose
                                                 > 0).any() else dose.max()
    cmap = create_dose_colormap(cmap_threshold)
    extent = [0, width * px_to_mm, height * px_to_mm, 0]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(dose,
                   cmap=cmap,
                   vmin=0,
                   vmax=vmax,
                   extent=extent,
                   aspect='equal')

    for mark, name in [(mark1, 'Mark 1'), (mark2, 'Mark 2')]:
        if mark:
            ax.plot(mark.center_x_mm,
                    mark.center_y_mm,
                    'bo',
                    ms=10,
                    markeredgecolor='white',
                    markeredgewidth=1.5)
            offset_y = 3 if mark.center_y_mm < height * px_to_mm / 2 else -3
            ax.annotate(f'({mark.center_x_mm:.1f}, {mark.center_y_mm:.1f}) mm',
                        xy=(mark.center_x_mm, mark.center_y_mm),
                        xytext=(mark.center_x_mm + 2,
                                mark.center_y_mm + offset_y),
                        fontsize=9,
                        color='blue',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white',
                                  alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1))

    if line:
        y_plot = np.linspace(line.y_range_mm[0], line.y_range_mm[1], 100)
        x_plot = line.a * y_plot + line.b
        ax.plot(x_plot,
                y_plot,
                'b-',
                lw=2,
                label=f'X = {line.a:.4f}*Y + {line.b:.2f}\n'
                f'θ = {line.angle_deg:.2f}°')
        ax.legend(loc='upper right', fontsize=9)

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_title(f'{key}')
    plt.colorbar(im, ax=ax, label='Dose [Gy]', fraction=0.046)
    fig.tight_layout()
    return fig


def _plot_combined_profiles(processed: Dict[str, dict], hw: int) -> plt.Figure:
    fig, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
    for key, data in processed.items():
        dose = data['dose_recalibrated']
        px_to_mm = data['px_to_mm']
        h, w = dose.shape
        yc, xc = h // 2, w // 2

        h_prof = dose[max(0, yc - hw):min(h, yc + hw), :].mean(axis=0)
        v_prof = dose[:, max(0, xc - hw):min(w, xc + hw)].mean(axis=1)

        ax_h.plot(np.arange(len(h_prof)) * px_to_mm, h_prof, lw=2, label=key)
        ax_v.plot(np.arange(len(v_prof)) * px_to_mm, v_prof, lw=2, label=key)

    for ax, xlabel, title in [(ax_h, 'X [mm]', 'Horizontal Profiles'),
                              (ax_v, 'Y [mm]', 'Vertical Profiles')]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Dose [Gy]')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    return fig


# --- Main pipeline ---


def analyze_ebt(config: EBTAnalysisConfig,
                output_dir: str = ".",
                pdf_path: Optional[str] = None) -> Dict:
    """Run full EBT analysis pipeline for one facility.

    Parameters
    ----------
    config : EBTAnalysisConfig
        Configuration with file pairs and processing parameters.
    output_dir : str
        Directory for output files.
    pdf_path : str, optional
        Path for PDF report.

    Returns
    -------
    dict with processed data and pencil mark results.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(output_dir_path / f"ebt_analysis_{config.facility}.pdf")

    source_paths = []
    for pair_cfg in config.file_pairs:
        source_paths.extend([pair_cfg['signal'], pair_cfg['background']])

    report = PDFReport(pdf_path,
                       title=f"EBT Analysis: {config.facility.upper()}",
                       config_path=str(source_paths[:2]))

    # --- Step 1: Load and process pairs ---
    processed: Dict[str, dict] = {}

    for pair_cfg in config.file_pairs:
        signal_path = Path(pair_cfg['signal'])
        bg_path = Path(pair_cfg['background'])
        key = pair_cfg.get('key', signal_path.parent.name)

        pair = _load_pair(signal_path, bg_path)
        pair = _crop_and_rotate(pair, config)

        # Background statistics
        bg_red = get_red_channel(pair.background_raw)
        y0, y1, x0, x1 = pair.crop_bbox
        if pair.background_raw.shape[:2] == pair.signal_raw.shape[:2]:
            bg_cropped = bg_red[y0:y1, x0:x1]
        else:
            bg_cropped = bg_red
        bg_cropped = rotate_dose(bg_cropped, k=config.rotation_k)

        radius_px = config.circle_radius_mm / pair.px_to_mm
        mask = _circular_mask(*bg_cropped.shape, radius_px)
        stats = _bg_stats(bg_cropped, mask)

        # Recalibrate
        dose_recal = _recalibrate(pair, stats['mean'], config)

        # Pencil marks + line
        mark1, mark2 = _detect_pencil_marks(dose_recal, pair.px_to_mm, config)
        line = _fit_pencil_line(dose_recal, pair.px_to_mm, config)

        processed[key] = {
            'pair': pair,
            'dose_recalibrated': dose_recal,
            'bg_stats': stats,
            'mark1': mark1,
            'mark2': mark2,
            'line': line,
            'px_to_mm': pair.px_to_mm,
        }

        # --- Report pages ---
        # Raw comparison
        fig_raw = plot_raw_signal_comparison(
            signal_image=get_red_channel(pair.signal_raw),
            background_image=get_red_channel(pair.background_raw),
            px_to_mm=pair.px_to_mm,
            title=f'Raw Signal - {key}')
        report.add_figure(fig_raw,
                          caption=f"Raw signal comparison: {key}",
                          source_paths=[str(signal_path),
                                        str(bg_path)])

        # Dose comparison
        fig_dose = plot_dose_with_background(
            dose_array=pair.dose_cropped,
            background_dose_array=pair.background_dose_cropped,
            px_to_mm=pair.px_to_mm,
            title=f'Dose (Cropped) - {key}',
            white_threshold_percent=config.cmap_white_threshold_percent)
        report.add_figure(
            fig_dose,
            caption=f"Dose comparison (cropped & rotated): {key}",
            source_paths=[str(signal_path)])

        # Recalibrated dose + profiles
        fig_map = _plot_dose_map_with_profiles(
            dose_recal, pair.px_to_mm, key, config.profile_half_width_px,
            config.cmap_white_threshold_percent)
        report.add_figure(fig_map,
                          caption=f"Recalibrated dose with profiles: {key}",
                          source_paths=[str(signal_path)])

        # Pencil marks + line
        fig_marks = _plot_dose_with_marks(dose_recal, pair.px_to_mm, mark1,
                                          mark2, line,
                                          config.cmap_white_threshold_percent,
                                          key)
        report.add_figure(fig_marks,
                          caption=f"Pencil marks and fitted line: {key}",
                          source_paths=[str(signal_path)])

    # --- Combined profiles ---
    fig_combined = _plot_combined_profiles(processed,
                                           config.profile_half_width_px)
    report.add_figure(fig_combined,
                      caption="Combined profiles (all pairs)",
                      source_paths=source_paths)

    # --- Summary tables ---
    import pandas as pd

    stats_rows = []
    for key, data in processed.items():
        s = data['bg_stats']
        stats_rows.append({
            'Pair': key,
            'BG Mean': f"{s['mean']:.2f}",
            'BG Std': f"{s['std']:.2f}",
            'Std/Mean %': f"{s['std_pct']:.2f}"
        })
    report.add_table(pd.DataFrame(stats_rows),
                     title="Background Statistics",
                     source_paths=source_paths)

    mark_rows = []
    for key, data in processed.items():
        m1, m2, ln = data['mark1'], data['mark2'], data['line']
        mark_rows.append({
            'Pair': key,
            'Mark1 X': f"{m1.center_x_mm:.2f}" if m1 else 'N/A',
            'Mark1 Y': f"{m1.center_y_mm:.2f}" if m1 else 'N/A',
            'Mark2 X': f"{m2.center_x_mm:.2f}" if m2 else 'N/A',
            'Mark2 Y': f"{m2.center_y_mm:.2f}" if m2 else 'N/A',
            'Line slope': f"{ln.a:.6f}" if ln else 'N/A',
            'Angle [°]': f"{ln.angle_deg:.2f}" if ln else 'N/A',
        })
    report.add_table(pd.DataFrame(mark_rows),
                     title="Pencil Mark Positions & Line Fits",
                     source_paths=source_paths)

    report.save()
    print(f"PDF report saved to: {pdf_path}")

    # Save NPZ for the key pair
    npz_key = config.output_pair_key or list(processed.keys())[0]
    npz_path = output_dir_path / (config.output_npz
                                  or f"ebt_2D_{config.facility}.npz")
    np.savez_compressed(str(npz_path), processed[npz_key]['dose_recalibrated'])
    print(f"NPZ saved to: {npz_path}")

    return processed
