"""MC profile comparison between AIC-144 and CCB facilities.

Ported from notebook: 0.4_mc_comparison.ipynb
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pub2026.config import MCComparisonConfig, resolve_file
from src.pub2026.mc.wedge_profile import find_fwhm_boundaries, find_distal_percent_position
from src.pub2026.pdf_report import PDFReport
from src.pub2026.profile_metrics import calculate_profile_metrics


def get_dose_at_x0(df: pd.DataFrame) -> float:
    """Get dose value at depth closest to 0."""
    idx_0 = np.abs(df['depth']).argmin()
    return float(df.iloc[idx_0]['dose'])


def _plot_comparison(depth_aic, dose_aic, depth_ccb, dose_ccb,
                     ref_dose_gy: float) -> plt.Figure:
    """Plot AIC-144 vs CCB profiles with FWHM and 90% markers."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(depth_aic, dose_aic, color='red', label='AIC-144', linewidth=1)
    ax.plot(depth_ccb, dose_ccb, color='green', label='CCB', linewidth=1)

    # AIC-144 FWHM
    left_a, right_a, hm_a = find_fwhm_boundaries(depth_aic, dose_aic)
    fwhm_a = (right_a -
              left_a) if (left_a is not None and right_a is not None) else None
    if fwhm_a is not None:
        ax.hlines(y=hm_a, xmin=left_a, xmax=right_a, color='red', linewidth=2)
        ax.text((left_a + right_a) / 2,
                hm_a + 0.15,
                f'{fwhm_a:.2f} mm',
                ha='center',
                va='bottom',
                color='red',
                fontsize=10,
                fontweight='bold')

    # CCB FWHM
    left_c, right_c, hm_c = find_fwhm_boundaries(depth_ccb, dose_ccb)
    fwhm_c = (right_c -
              left_c) if (left_c is not None and right_c is not None) else None
    if fwhm_c is not None:
        ax.hlines(y=hm_c,
                  xmin=left_c,
                  xmax=right_c,
                  color='green',
                  linewidth=2)
        ax.text((left_c + right_c) / 2,
                hm_c - 0.15,
                f'{fwhm_c:.2f} mm',
                ha='center',
                va='top',
                color='green',
                fontsize=10,
                fontweight='bold')

    # 90% markers
    x90_a, d90_a = find_distal_percent_position(depth_aic, dose_aic, 90)
    x90_c, d90_c = find_distal_percent_position(depth_ccb, dose_ccb, 90)

    if x90_a is not None:
        ax.plot(x90_a, d90_a, 'o', color='red', markersize=8)
        ax.text(x90_a + 0.3,
                d90_a,
                f'90%: {x90_a:.2f} mm',
                ha='left',
                va='center',
                color='red',
                fontsize=9,
                fontweight='bold')
    if x90_c is not None:
        ax.plot(x90_c, d90_c, 'o', color='green', markersize=8)
        ax.text(x90_c + 0.3,
                d90_c - 0.3,
                f'90%: {x90_c:.2f} mm',
                ha='left',
                va='top',
                color='green',
                fontsize=9,
                fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Depth [mm]')
    ax.set_ylabel('Dose [Gy]')
    ax.set_title(
        f'MC Depth Dose Profiles (normalized to {ref_dose_gy} Gy at X=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    fig.tight_layout()
    return fig


def compare_mc_profiles(config: MCComparisonConfig,
                        output_dir: str = ".",
                        pdf_path: Optional[str] = None) -> pd.DataFrame:
    """Compare MC profiles from AIC-144 and CCB.

    Parameters
    ----------
    config : MCComparisonConfig
        Configuration with file paths and normalization settings.
    output_dir : str
        Directory for output files.
    pdf_path : str, optional
        Path for PDF report.

    Returns
    -------
    DataFrame with comparison metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(output_dir / "mc_comparison.pdf")

    # Load data — try resolved path first, then fall back to output_dir
    aic_path = resolve_file(config.aic144_csv, output_dir)
    ccb_path = resolve_file(config.ccb_csv, output_dir)

    source_paths = [str(aic_path), str(ccb_path)]
    report = PDFReport(pdf_path,
                       title="MC Profile Comparison: AIC-144 vs CCB",
                       config_path=str(source_paths))

    df_aic = pd.read_csv(aic_path)
    df_ccb = pd.read_csv(ccb_path)

    # Normalize to reference dose at X=0
    scale_aic = config.reference_dose_gy / get_dose_at_x0(df_aic)
    scale_ccb = config.reference_dose_gy / get_dose_at_x0(df_ccb)

    depth_aic = df_aic['depth'].values
    dose_aic = df_aic['dose'].values * scale_aic
    depth_ccb = df_ccb['depth'].values
    dose_ccb = df_ccb['dose'].values * scale_ccb

    # Comparison plot
    fig = _plot_comparison(depth_aic, dose_aic, depth_ccb, dose_ccb,
                           config.reference_dose_gy)
    report.add_figure(
        fig,
        caption="MC profiles comparison with FWHM and 90% markers",
        source_paths=source_paths)

    # Calculate metrics
    metrics_aic = calculate_profile_metrics(dose_aic, depth_aic)
    metrics_ccb = calculate_profile_metrics(dose_ccb, depth_ccb)

    # Build metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['FWHM [mm]', '90%-10% [mm]', '80%-20% [mm]'],
        'AIC-144': [
            f'{metrics_aic["fwhm"]:.2f}', f'{metrics_aic["dist_90_10"]:.2f}',
            f'{metrics_aic["dist_80_20"]:.2f}'
        ],
        'CCB': [
            f'{metrics_ccb["fwhm"]:.2f}', f'{metrics_ccb["dist_90_10"]:.2f}',
            f'{metrics_ccb["dist_80_20"]:.2f}'
        ],
    })

    report.add_table(metrics_df,
                     title="Profile Metrics Comparison",
                     source_paths=source_paths)

    # Detailed positions
    detail_df = pd.DataFrame({
        'Position': ['90% [mm]', '80% [mm]', '20% [mm]', '10% [mm]'],
        'AIC-144': [
            f'{metrics_aic["pos_90"]:.2f}', f'{metrics_aic["pos_80"]:.2f}',
            f'{metrics_aic["pos_20"]:.2f}', f'{metrics_aic["pos_10"]:.2f}'
        ],
        'CCB': [
            f'{metrics_ccb["pos_90"]:.2f}', f'{metrics_ccb["pos_80"]:.2f}',
            f'{metrics_ccb["pos_20"]:.2f}', f'{metrics_ccb["pos_10"]:.2f}'
        ],
    })

    report.add_table(detail_df,
                     title="Detailed Distal Positions",
                     source_paths=source_paths)

    report.save()
    print(f"PDF report saved to: {pdf_path}")

    return metrics_df
