"""EBT film comparison: AIC-144 vs CCB dose profiles.

Ported from notebook: 2.0_ebt_comparison.ipynb
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pub2026.config import EBTComparisonConfig, resolve_file
from src.pub2026.pdf_report import PDFReport
from src.pub2026.profile_metrics import (
    find_fwhm,
    find_distal_pct,
    calculate_profile_metrics,
)
from src.visualise.dose import create_dose_colormap


def _plot_dose_maps(dose_aic: np.ndarray, dose_ccb: np.ndarray, extent_aic,
                    extent_ccb, strip_aic, strip_ccb, vmax: float,
                    cmap_threshold: float) -> plt.Figure:
    cmap = create_dose_colormap(cmap_threshold)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(dose_aic,
                     cmap=cmap,
                     vmin=0,
                     vmax=vmax,
                     extent=extent_aic,
                     aspect='equal')
    ax1.axhline(y=strip_aic[0], color='black', ls='-', lw=1.5)
    ax1.axhline(y=strip_aic[1], color='black', ls='-', lw=1.5)
    ax1.axvline(x=0, color='black', ls='-', lw=1)
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title('AIC-144 Dose Distribution')
    plt.colorbar(im1, ax=ax1, label='Dose [Gy]', fraction=0.046)

    im2 = ax2.imshow(dose_ccb,
                     cmap=cmap,
                     vmin=0,
                     vmax=vmax,
                     extent=extent_ccb,
                     aspect='equal')
    ax2.axhline(y=strip_ccb[0], color='black', ls='-', lw=1.5)
    ax2.axhline(y=strip_ccb[1], color='black', ls='-', lw=1.5)
    ax2.axvline(x=0, color='black', ls='-', lw=1)
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_title('CCB Dose Distribution')
    plt.colorbar(im2, ax=ax2, label='Dose [Gy]', fraction=0.046)

    fig.tight_layout()
    return fig


def _plot_profiles(x_aic, prof_aic, x_ccb, prof_ccb, fwhm_aic, fwhm_ccb,
                   strip_width: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x_aic, prof_aic, color='red', label='AIC-144', lw=1)
    ax.plot(x_ccb, prof_ccb, color='green', label='CCB', lw=1)

    # FWHM segments
    la, ra, hma = find_fwhm(prof_aic, x_aic)
    lc, rc, hmc = find_fwhm(prof_ccb, x_ccb)

    if la is not None and ra is not None:
        ax.hlines(y=hma, xmin=la, xmax=ra, color='red', lw=2)
        ax.text((la + ra) / 2,
                hma + 0.3,
                f'{ra - la:.2f} mm',
                ha='center',
                va='bottom',
                color='red',
                fontsize=10,
                fontweight='bold')
    if lc is not None and rc is not None:
        ax.hlines(y=hmc, xmin=lc, xmax=rc, color='green', lw=2)
        ax.text((lc + rc) / 2,
                hmc - 0.3,
                f'{rc - lc:.2f} mm',
                ha='center',
                va='top',
                color='green',
                fontsize=10,
                fontweight='bold')

    # 90% markers
    x90a, d90a = find_distal_pct(prof_aic, x_aic, 90)
    x90c, d90c = find_distal_pct(prof_ccb, x_ccb, 90)
    if x90a is not None:
        ax.plot(x90a, d90a, 'o', color='red', ms=8)
        ax.text(x90a + 0.5,
                d90a,
                f'90%: {x90a:.2f} mm',
                ha='left',
                va='center',
                color='red',
                fontsize=9,
                fontweight='bold')
    if x90c is not None:
        ax.plot(x90c, d90c, 'o', color='green', ms=8)
        ax.text(x90c + 0.5,
                d90c,
                f'90%: {x90c:.2f} mm',
                ha='left',
                va='center',
                color='green',
                fontsize=9,
                fontweight='bold')

    ax.axvline(x=0, color='black', ls='-', lw=1)
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Dose [Gy]')
    ax.set_title(
        f'Horizontal Dose Profiles (averaged over {strip_width:.1f} mm strip)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, 30)
    ax.set_ylim(0, None)
    fig.tight_layout()
    return fig


def compare_ebt(config: EBTComparisonConfig,
                output_dir: str = ".",
                pdf_path: Optional[str] = None) -> pd.DataFrame:
    """Compare EBT profiles from AIC-144 and CCB.

    Parameters
    ----------
    config : EBTComparisonConfig
    output_dir : str
    pdf_path : str, optional

    Returns
    -------
    DataFrame with comparison metrics.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(output_dir_path / "ebt_comparison.pdf")

    source_paths = [config.aic144_npz, config.ccb_npz]
    report = PDFReport(pdf_path,
                       title="EBT Comparison: AIC-144 vs CCB",
                       config_path=str(source_paths))

    # Load — resolve with output_dir fallback for intermediate outputs
    aic_path = resolve_file(config.aic144_npz, output_dir_path)
    ccb_path = resolve_file(config.ccb_npz, output_dir_path)
    dose_aic = np.load(str(aic_path))['arr_0']
    dose_ccb = np.load(str(ccb_path))['arr_0']

    px_to_mm = 25.4 / config.dpi
    pm_aic = config.pencil_mark_x_aic144_mm
    pm_ccb = config.pencil_mark_x_ccb_mm

    # Extents (X shifted by pencil mark)
    h_a, w_a = dose_aic.shape
    extent_aic = [-pm_aic, w_a * px_to_mm - pm_aic, h_a * px_to_mm, 0]
    h_c, w_c = dose_ccb.shape
    extent_ccb = [-pm_ccb, w_c * px_to_mm - pm_ccb, h_c * px_to_mm, 0]

    strip_half = config.strip_width_mm / 2
    strip_aic = (h_a * px_to_mm / 2 - strip_half,
                 h_a * px_to_mm / 2 + strip_half)
    strip_ccb = (h_c * px_to_mm / 2 - strip_half,
                 h_c * px_to_mm / 2 + strip_half)

    vmax_a = np.percentile(dose_aic[dose_aic > 0],
                           99) if (dose_aic > 0).any() else dose_aic.max()
    vmax_c = np.percentile(dose_ccb[dose_ccb > 0],
                           99) if (dose_ccb > 0).any() else dose_ccb.max()
    vmax = max(vmax_a, vmax_c)

    # 2D dose maps
    fig_maps = _plot_dose_maps(dose_aic, dose_ccb, extent_aic, extent_ccb,
                               strip_aic, strip_ccb, vmax, 1.0)
    report.add_figure(fig_maps,
                      caption="2D dose maps with profile strips",
                      source_paths=source_paths)

    # Extract profiles
    strip_half_px = int(strip_half / px_to_mm)
    cr_a = h_a // 2
    prof_aic = dose_aic[cr_a - strip_half_px:cr_a +
                        strip_half_px, :].mean(axis=0)
    x_aic = np.arange(len(prof_aic)) * px_to_mm - pm_aic

    cr_c = h_c // 2
    prof_ccb = dose_ccb[cr_c - strip_half_px:cr_c +
                        strip_half_px, :].mean(axis=0)
    x_ccb = np.arange(len(prof_ccb)) * px_to_mm - pm_ccb

    # Profile comparison plot
    fig_prof = _plot_profiles(x_aic, prof_aic, x_ccb, prof_ccb, 0, 0,
                              config.strip_width_mm)
    report.add_figure(fig_prof,
                      caption="Horizontal dose profile comparison",
                      source_paths=source_paths)

    # Metrics
    m_aic = calculate_profile_metrics(prof_aic, x_aic)
    m_ccb = calculate_profile_metrics(prof_ccb, x_ccb)

    metrics_df = pd.DataFrame({
        'Metric': ['FWHM [mm]', '90%-10% [mm]', '80%-20% [mm]'],
        'AIC-144': [
            f'{m_aic["fwhm"]:.2f}', f'{m_aic["dist_90_10"]:.2f}',
            f'{m_aic["dist_80_20"]:.2f}'
        ],
        'CCB': [
            f'{m_ccb["fwhm"]:.2f}', f'{m_ccb["dist_90_10"]:.2f}',
            f'{m_ccb["dist_80_20"]:.2f}'
        ],
    })
    report.add_table(metrics_df,
                     title="Profile Metrics Comparison",
                     source_paths=source_paths)

    report.save()
    print(f"PDF report saved to: {pdf_path}")

    # Save CSVs
    df_aic = pd.DataFrame({'x_mm': x_aic, 'dose_Gy': prof_aic})
    df_ccb = pd.DataFrame({'x_mm': x_ccb, 'dose_Gy': prof_ccb})

    csv_aic = output_dir_path / config.output_csv_aic144
    csv_ccb = output_dir_path / config.output_csv_ccb
    df_aic.to_csv(csv_aic, index=False)
    df_ccb.to_csv(csv_ccb, index=False)
    print(f"CSVs saved to: {csv_aic}, {csv_ccb}")

    return metrics_df
