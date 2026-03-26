"""Raw MBO foil survey: display all foils without background subtraction.

Shows each detector's raw and LV images with colorbars and isosignal contour
lines.  No denoising, no circle detection — just a plain visual survey of the
raw data to support an alternative analysis approach.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from src.data.analysis import read_tiff_img
from src.data.detector import DetectorDataCollection
from src.pub2026.config import MBORawSurveyConfig
from src.pub2026.pdf_report import PDFReport


def _smooth(image: np.ndarray, method: str, kernel: int) -> np.ndarray:
    """Apply optional smoothing to an image."""
    if method == "gaussian":
        return ndi.gaussian_filter(image.astype(np.float64), sigma=kernel)
    if method == "median":
        return ndi.median_filter(image, size=kernel).astype(np.float64)
    return image.astype(np.float64)


def _percentile_clim(image: np.ndarray, low: float,
                     high: float) -> Tuple[float, float]:
    """Return (vmin, vmax) from percentiles, ignoring NaNs."""
    return float(np.nanpercentile(image,
                                  low)), float(np.nanpercentile(image, high))


def _plot_foil(image: np.ndarray,
               pixel_size_mm: float,
               vmin: float,
               vmax: float,
               title: str,
               contour_levels: Optional[List[float]] = None) -> plt.Figure:
    """Single-foil 2D image with colorbar and optional isosignal contour lines."""
    h, w = image.shape
    extent = [0, w * pixel_size_mm, h * pixel_size_mm, 0]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(image, extent=extent, vmin=vmin, vmax=vmax, cmap='hot')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')

    if contour_levels:
        x_arr = np.linspace(extent[0], extent[1], w)
        y_arr = np.linspace(extent[3], extent[2], h)
        cs = ax.contour(x_arr,
                        y_arr,
                        image,
                        levels=contour_levels,
                        colors='cyan',
                        linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

    ax.set(xlabel='X [mm]', ylabel='Y [mm]', title=title)
    fig.tight_layout()
    return fig


def _plot_overview_grid(
        images: Dict[int, np.ndarray],
        pixel_size_mm: float,
        clim: Tuple[float, float],
        suptitle: str,
        contour_levels: Optional[List[float]] = None) -> plt.Figure:
    """Grid of all foils for a quick side-by-side comparison."""
    n = len(images)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows,
                             ncols,
                             figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)

    sorted_ids = sorted(images.keys())
    for idx, det_id in enumerate(sorted_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        img = images[det_id]
        h, w = img.shape
        extent = [0, w * pixel_size_mm, h * pixel_size_mm, 0]

        v0, v1 = _percentile_clim(img, clim[0], clim[1])
        im = ax.imshow(img, extent=extent, vmin=v0, vmax=v1, cmap='hot')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if contour_levels:
            x_arr = np.linspace(extent[0], extent[1], w)
            y_arr = np.linspace(extent[3], extent[2], h)
            cs = ax.contour(x_arr,
                            y_arr,
                            img,
                            levels=contour_levels,
                            colors='cyan',
                            linewidths=0.6)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%d')
        ax.set(xlabel='X [mm]', ylabel='Y [mm]', title=f'Foil {det_id}')

    # hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def survey_raw_mbo(config: MBORawSurveyConfig,
                   output_dir: str = ".",
                   pdf_path: Optional[str] = None) -> None:
    """Run a raw survey of MBO foils for one dataset.

    Parameters
    ----------
    config : MBORawSurveyConfig
    output_dir : str
    pdf_path : str, optional
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(output_dir_path /
                       f"mbo_raw_survey_{config.facility}.pdf")

    data_path = Path(config.data_dir).resolve()

    report = PDFReport(
        pdf_path,
        title=f"MBO Raw Survey: {config.facility.upper()}",
        config_path=str(data_path),
    )

    report.add_text(
        f"Input TIFF directory:\n{data_path}\n\n"
        f"No background subtraction applied.\n"
        f"Smoothing: {config.smooth_method} (kernel={config.smooth_kernel})\n"
        f"Percentile clipping: [{config.clip_percentile_low}, "
        f"{config.clip_percentile_high}]\n"
        f"cut_px={config.cut_px}, pixel_size_mm={config.pixel_size_mm}\n"
        f"contour_levels={config.contour_levels}",
        title="Data Source & Parameters",
    )

    # Load all detectors
    collection = DetectorDataCollection(path=data_path, cut_px=config.cut_px)

    # Apply smoothing
    raw_images = {
        did: _smooth(det.raw.image, config.smooth_method, config.smooth_kernel)
        for did, det in collection.data.items()
    }
    lv_images = {
        did: _smooth(det.lv.image, config.smooth_method, config.smooth_kernel)
        for did, det in collection.data.items()
    }
    clim = (config.clip_percentile_low, config.clip_percentile_high)

    # -- Overview grids (raw + LV) --
    fig_raw_grid = _plot_overview_grid(raw_images,
                                       config.pixel_size_mm,
                                       clim,
                                       'All Foils — Raw Signal',
                                       contour_levels=config.contour_levels)
    report.add_figure(fig_raw_grid,
                      caption="Overview of all foils (raw)",
                      source_paths=[str(data_path)])

    fig_lv_grid = _plot_overview_grid(lv_images, config.pixel_size_mm, clim,
                                      'All Foils — Live View')
    report.add_figure(fig_lv_grid,
                      caption="Overview of all foils (live view)",
                      source_paths=[str(data_path)])

    # -- Individual foil pages (raw + LV) --
    for det_id in sorted(raw_images.keys()):
        raw_img = raw_images[det_id]
        lv_img = lv_images[det_id]
        det = collection.data[det_id]
        v0, v1 = _percentile_clim(raw_img, clim[0], clim[1])

        fig_raw = _plot_foil(raw_img,
                             config.pixel_size_mm,
                             v0,
                             v1,
                             title=f'Foil {det_id} — Raw Signal',
                             contour_levels=config.contour_levels)
        report.add_figure(fig_raw,
                          caption=f"Foil {det_id} raw image",
                          source_paths=[str(det.raw.path)])

        v0_lv, v1_lv = _percentile_clim(lv_img, clim[0], clim[1])
        fig_lv = _plot_foil(lv_img,
                            config.pixel_size_mm,
                            v0_lv,
                            v1_lv,
                            title=f'Foil {det_id} — Live View')
        report.add_figure(fig_lv,
                          caption=f"Foil {det_id} live-view image",
                          source_paths=[str(det.lv.path)])

    # -- Horizontal profiles (central strip) --
    fig_prof, ax = plt.subplots(figsize=(12, 6))
    for det_id in sorted(raw_images.keys()):
        img = raw_images[det_id]
        mid = img.shape[0] // 2
        strip = img[mid - 10:mid + 10, :].mean(axis=0)
        x_mm = np.arange(len(strip)) * config.pixel_size_mm
        ax.plot(x_mm, strip, label=f'Foil {det_id}')
    ax.set(xlabel='X [mm]',
           ylabel='Intensity',
           title='Horizontal Profiles (central ±10 px strip)')
    ax.legend()
    fig_prof.tight_layout()
    report.add_figure(fig_prof,
                      caption="Horizontal profiles — raw",
                      source_paths=[str(data_path)])

    report.save()
    print(f"PDF report saved to: {pdf_path}")

    # -- Save smoothed images as NPZ for downstream (align) --
    npz_name = config.output_npz or f"mbo_survey_{config.facility}.npz"
    npz_path = output_dir_path / npz_name
    save_dict = {'detector_ids': np.array(sorted(raw_images.keys()))}
    for did in sorted(raw_images.keys()):
        save_dict[f'foil_{did}_raw'] = raw_images[did]
        save_dict[f'foil_{did}_lv'] = lv_images[did]
    np.savez_compressed(npz_path, **save_dict)
    print(f"Survey NPZ saved to: {npz_path}")
