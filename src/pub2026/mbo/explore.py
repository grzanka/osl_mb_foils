"""MBO foil exploration: background subtraction, denoising, circle detection.

Ported from notebooks: 0.1-explore-aic144.ipynb, 0.1-explore-ccb.ipynb
"""

import pickle
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.colors import LinearSegmentedColormap

from src.data.analysis import Circle, read_tiff_img, create_circular_mask
from src.data.detector import (
    DetectorData,
    DetectorDataCollection,
    DetectorImage,
    find_circle_hough_method,
    img_for_circle_detection,
)
from src.pub2026.config import MBOExploreConfig
from src.pub2026.pdf_report import PDFReport


def _compute_noise_stats(image: np.ndarray, strip_width: int = 20):
    strips = np.concatenate([
        image[:, :strip_width].flatten(),
        image[:, -strip_width:].flatten(),
        image[:strip_width, :].flatten(),
        image[-strip_width:, :].flatten(),
    ])
    return float(np.mean(strips)), float(np.std(strips))


def _green_red_cmap():
    cmap = LinearSegmentedColormap.from_list('green_red', [(0, 1, 0),
                                                           (1, 0, 0)])
    cmap.set_bad(color='white')
    return cmap


def _subtract_background(image: np.ndarray, bg: np.ndarray) -> np.ndarray:
    result = image.copy()
    offset = bg.max()
    np.add(result, offset, out=result)
    np.subtract(result, bg, out=result)
    np.clip(result, a_min=offset, a_max=None, out=result)
    np.subtract(result, offset, out=result)
    return result


def _plot_detector(lv: np.ndarray, raw: np.ndarray, title: str, sigma: int,
                   n_sigma: int, probe_r: float, strip_w: int,
                   clip_pct: float) -> plt.Figure:
    cmap = _green_red_cmap()
    lv_m, lv_s = _compute_noise_stats(lv, strip_w)
    raw_m, raw_s = _compute_noise_stats(raw, strip_w)

    lv_sm = ndi.gaussian_filter(lv, sigma=sigma).astype(np.float32)
    raw_sm = ndi.gaussian_filter(raw, sigma=sigma).astype(np.float32)
    mask = lv_sm > lv_m + n_sigma * lv_s
    lv_sm[~mask] = 0
    raw_sm[~mask] = 0

    lv_cm = ndi.center_of_mass(lv_sm)
    lv_probe = create_circular_mask(lv_sm,
                                    Circle(x=lv_cm[1], y=lv_cm[0], r=probe_r))
    raw_cm = ndi.center_of_mass(raw_sm)
    raw_probe = create_circular_mask(
        raw_sm, Circle(x=raw_cm[1], y=raw_cm[0], r=probe_r))

    lv_pm, lv_ps = np.nanmean(lv_sm[lv_probe]), np.nanstd(lv_sm[lv_probe])
    raw_pm, raw_ps = np.nanmean(raw_sm[raw_probe]), np.nanstd(
        raw_sm[raw_probe])
    lv_sm[(lv_sm < lv_pm - n_sigma * lv_ps) |
          (lv_sm > lv_pm + n_sigma * lv_ps)] = np.nan
    raw_sm[(raw_sm < raw_pm - n_sigma * raw_ps) |
           (raw_sm > raw_pm + n_sigma * raw_ps)] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, img, sm, cm, pm, ps, pr, label in [
        (axes[0], lv, lv_sm, lv_cm, lv_pm, lv_ps, lv_probe, 'LV'),
        (axes[1], raw, raw_sm, raw_cm, raw_pm, raw_ps, raw_probe, 'Raw'),
    ]:
        upper = np.percentile(img, clip_pct)
        ax[0].hist(img.flatten(),
                   bins=100,
                   range=(0, upper),
                   density=True,
                   histtype='step')
        ax[0].hist(sm[pr].flatten(),
                   bins=100,
                   range=(0, upper),
                   density=True,
                   histtype='step',
                   color='blue')
        ax[0].set(yscale='log', title=f'{label} Histogram')
        ax[0].grid(True, alpha=0.3)

        im = ax[1].imshow(sm,
                          cmap=cmap,
                          vmin=pm - n_sigma * ps,
                          vmax=pm + n_sigma * ps)
        ax[1].scatter(*cm[::-1], color='blue', s=20)
        ax[1].add_patch(
            plt.Circle(cm[::-1], probe_r, color='blue', fill=False, lw=1.5))
        ax[1].set_title(f'{label} Image')
        plt.colorbar(im, ax=ax[1], shrink=0.8)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def explore_mbo(config: MBOExploreConfig,
                output_dir: str = ".",
                pdf_path: Optional[str] = None) -> Dict:
    """Run MBO exploration pipeline for one facility.

    Parameters
    ----------
    config : MBOExploreConfig
    output_dir : str
    pdf_path : str, optional

    Returns
    -------
    dict with processed detector data.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(output_dir_path / f"mbo_explore_{config.facility}.pdf")

    data_path = Path(config.data_dir)
    bg_path = data_path / "tlo"

    report = PDFReport(pdf_path,
                       title=f"MBO Exploration: {config.facility.upper()}",
                       config_path=str(data_path))

    # Load data
    proton_data = DetectorDataCollection(path=data_path, cut_px=config.cut_px)

    bg_raw_path = next(bg_path.glob('**/*tif'))
    bg_lv_path = next(bg_path.with_name('tlo_lv').glob('**/*tif'))
    bg_data = DetectorData(
        raw=DetectorImage(path=bg_raw_path,
                          image=read_tiff_img(bg_raw_path,
                                              cut_px=config.cut_px)),
        lv=DetectorImage(path=bg_lv_path,
                         image=read_tiff_img(bg_lv_path,
                                             cut_px=config.cut_px)),
        det_no=-1,
    )

    # Process: bg subtraction + denoise + circle detection
    final_data = {}
    for det_id, det in proton_data.data.items():
        bg_sub = _subtract_background(det.raw.image, bg_data.raw.image)
        denoised = ndi.median_filter(bg_sub, size=config.median_filter_size)

        lv_for_det = img_for_circle_detection(det.lv.image)
        circle = find_circle_hough_method(lv_for_det)

        final_data[det_id] = DetectorData(
            raw=DetectorImage(path=det.raw.path, image=denoised),
            lv=det.lv,
            det_no=det_id,
            circle=circle,
        )

    # Report — first 3 foils
    for det_id in sorted(final_data.keys())[:3]:
        det = final_data[det_id]
        fig = _plot_detector(
            det.lv.image.copy(),
            det.raw.image.copy(),
            f'Foil {det_id} (r={det.circle.r:.1f}px)',
            sigma=1,
            n_sigma=config.n_sigma,
            probe_r=config.probe_radius,
            strip_w=20,
            clip_pct=99.9,
        )
        report.add_figure(fig,
                          caption=f"Foil {det_id} processed",
                          source_paths=[str(data_path)])

    # Horizontal profiles
    fig_prof, ax = plt.subplots(figsize=(10, 6))
    for det_id in sorted(final_data.keys())[:3]:
        profile = final_data[det_id].raw.image[180:220, :].mean(axis=0)
        ax.plot(profile, label=f'Foil {det_id}')
    ax.set(xlabel='Pixel', ylabel='Intensity', title='Horizontal Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_prof.tight_layout()
    report.add_figure(fig_prof,
                      caption="Horizontal profiles (rows 180-220)",
                      source_paths=[str(data_path)])

    report.save()
    print(f"PDF report saved to: {pdf_path}")

    # Save pickle
    collection = DetectorDataCollection(path=data_path, data=final_data)
    pkl_path = output_dir_path / (config.output_pkl
                                  or f"mbo_{config.facility}_processed.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(collection, f)
    print(f"Pickle saved to: {pkl_path}")

    return {'final_data': final_data, 'collection': collection}
