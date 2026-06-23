"""MC wedge (lateral) profile processing for both CSV and DAT formats.

Ported from notebooks: 0.2_ccb_mc_wedge.ipynb, 0.3_aic144_mc_wedge_big.ipynb,
0.3_ccb_mc_wedge_big.ipynb
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.metrics import calculate_depth_dose_metrics
from src.pub2026.config import MCWedgeConfig
from src.pub2026.pdf_report import PDFReport
from src.pub2026.profile_metrics import find_fwhm, find_distal_pct


def load_csv_files(dose_path: str, fluence_path: str,
                   energy_path: str) -> pd.DataFrame:
    """Load MC wedge data from CSV files (small simulation format).

    Returns merged DataFrame with columns: depth, dose, fluence, kinetic_energy.
    """
    df_dose = pd.read_csv(dose_path, comment='#', skiprows=0)
    df_fluence = pd.read_csv(fluence_path, comment='#', skiprows=0)
    df_kinetic = pd.read_csv(energy_path, comment='#', skiprows=0)

    for df in (df_dose, df_fluence, df_kinetic):
        df.columns = df.columns.str.strip()

    depth_col = df_dose.iloc[:, 0]
    dose_col = df_dose.iloc[:, 1]
    fluence_col = df_fluence.iloc[:, 1]
    kinetic_col = df_kinetic.iloc[:, 1]

    return pd.DataFrame({
        'depth': depth_col * 10,  # cm to mm
        'dose': dose_col / dose_col.max(),
        'fluence': fluence_col,
        'kinetic_energy': kinetic_col,
    })


def load_dat_files(dose_path: str, fluence_path: str,
                   energy_path: str) -> pd.DataFrame:
    """Load MC wedge data from DAT files (big simulation format with uncertainties).

    Returns merged DataFrame with columns: depth, dose, dose_err, fluence, fluence_err,
    kinetic_energy, kinetic_energy_err.
    """
    df_fluence = pd.read_csv(fluence_path,
                             sep=r'\s+',
                             header=None,
                             names=['x_cm', 'fluence', 'fluence_err'])
    df_kinetic = pd.read_csv(
        energy_path,
        sep=r'\s+',
        header=None,
        names=['x_cm', 'kinetic_energy', 'kinetic_energy_err'])
    df_dose = pd.read_csv(dose_path,
                          sep=r'\s+',
                          header=None,
                          names=['x_cm', 'dose', 'dose_err'])

    dose_max = df_dose['dose'].max()

    return pd.DataFrame({
        'depth': df_dose['x_cm'] * 10,  # cm to mm
        'dose': df_dose['dose'] / dose_max,
        'dose_err': df_dose['dose_err'] / dose_max,
        'fluence': df_fluence['fluence'],
        'fluence_err': df_fluence['fluence_err'],
        'kinetic_energy': df_kinetic['kinetic_energy'],
        'kinetic_energy_err': df_kinetic['kinetic_energy_err'],
    })


def filter_reverse_shift(df: pd.DataFrame, min_mm: float, max_mm: float,
                         shift_mm: float) -> pd.DataFrame:
    """Filter depth range, reverse value columns, and shift depth.

    Parameters
    ----------
    df : DataFrame
        Must have 'depth' column plus value columns.
    min_mm, max_mm : float
        Depth range to keep (before shift).
    shift_mm : float
        Value to add to depth after filtering.

    Returns
    -------
    Transformed DataFrame.
    """
    df_out = df[(df['depth'] >= min_mm) & (df['depth'] <= max_mm)].copy()

    value_cols = [c for c in df_out.columns if c != 'depth']
    for col in value_cols:
        df_out[col] = df_out[col].iloc[::-1].values

    df_out['depth'] = df_out['depth'] + shift_mm
    return df_out


def plot_triple_axis(df: pd.DataFrame,
                     title: str,
                     show_errors: bool = True,
                     error_alpha: float = 0.3) -> plt.Figure:
    """Plot dose, fluence, and kinetic energy on three y-axes."""
    fig, ax = plt.subplots(figsize=(12, 7))

    color1 = 'tab:blue'
    ax.set_xlabel('Depth (mm)', fontsize=12)
    ax.set_ylabel('Normalized Dose', color=color1, fontsize=12)
    line1 = ax.plot(df['depth'],
                    df['dose'],
                    marker='.',
                    linestyle='-',
                    linewidth=1,
                    markersize=1,
                    color=color1,
                    label='Dose (normalized)')
    if show_errors and 'dose_err' in df.columns:
        ax.fill_between(df['depth'],
                        df['dose'] - df['dose_err'],
                        df['dose'] + df['dose_err'],
                        color=color1,
                        alpha=error_alpha)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Fluence', color=color2, fontsize=12)
    line2 = ax2.plot(df['depth'],
                     df['fluence'],
                     marker='.',
                     linestyle='-',
                     linewidth=1,
                     markersize=1,
                     color=color2,
                     label='Fluence')
    if show_errors and 'fluence_err' in df.columns:
        ax2.fill_between(df['depth'],
                         df['fluence'] - df['fluence_err'],
                         df['fluence'] + df['fluence_err'],
                         color=color2,
                         alpha=error_alpha)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)

    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = 'tab:green'
    ax3.set_ylabel('Kinetic Energy', color=color3, fontsize=12)
    line3 = ax3.plot(df['depth'],
                     df['kinetic_energy'],
                     marker='.',
                     linestyle='-',
                     linewidth=1,
                     markersize=1,
                     color=color3,
                     label='Kinetic Energy')
    if show_errors and 'kinetic_energy_err' in df.columns:
        ax3.fill_between(df['depth'],
                         df['kinetic_energy'] - df['kinetic_energy_err'],
                         df['kinetic_energy'] + df['kinetic_energy_err'],
                         color=color3,
                         alpha=error_alpha)
    ax3.tick_params(axis='y', labelcolor=color3)
    ax3.set_ylim(bottom=0)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best', fontsize=11)
    plt.title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_individual_quantities(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot dose, fluence, and kinetic energy as separate subplots with error bars."""
    has_errors = 'dose_err' in df.columns
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    if has_errors:
        ax.errorbar(df['depth'],
                    df['dose'],
                    yerr=df['dose_err'],
                    fmt='o-',
                    capsize=2,
                    markersize=2,
                    linewidth=1,
                    color='tab:blue')
    else:
        ax.plot(df['depth'],
                df['dose'],
                'o-',
                markersize=2,
                linewidth=1,
                color='tab:blue')
    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel('Normalized Dose')
    ax.set_title('Dose vs Depth')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if has_errors:
        ax.errorbar(df['depth'],
                    df['fluence'],
                    yerr=df['fluence_err'],
                    fmt='o-',
                    capsize=2,
                    markersize=2,
                    linewidth=1,
                    color='tab:orange')
    else:
        ax.plot(df['depth'],
                df['fluence'],
                'o-',
                markersize=2,
                linewidth=1,
                color='tab:orange')
    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel('Fluence')
    ax.set_title('Fluence vs Depth')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if has_errors:
        ax.errorbar(df['depth'],
                    df['kinetic_energy'],
                    yerr=df['kinetic_energy_err'],
                    fmt='o-',
                    capsize=2,
                    markersize=2,
                    linewidth=1,
                    color='tab:green')
    else:
        ax.plot(df['depth'],
                df['kinetic_energy'],
                'o-',
                markersize=2,
                linewidth=1,
                color='tab:green')
    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel('Kinetic Energy (MeV)')
    ax.set_title('Kinetic Energy vs Depth')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def find_fwhm_boundaries(depth, dose):
    """Find left and right FWHM boundaries by linear interpolation."""
    depth = np.asarray(depth, dtype=float)
    dose = np.asarray(dose, dtype=float)
    left, right, half_max = find_fwhm(dose, depth)
    return left, right, half_max


def find_distal_percent_position(depth, dose, target_percent):
    """Find distal position where dose drops to target_percent of max."""
    depth = np.asarray(depth, dtype=float)
    dose = np.asarray(dose, dtype=float)
    return find_distal_pct(dose, depth, target_percent)


def plot_dose_with_markers(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot normalized dose with FWHM segment and 90% distal marker."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['depth'],
            df['dose'],
            color='tab:blue',
            linewidth=1.5,
            label='Normalized Dose')

    left_fwhm, right_fwhm, half_max = find_fwhm_boundaries(
        df['depth'], df['dose'])
    fwhm = (right_fwhm - left_fwhm) if (left_fwhm is not None
                                        and right_fwhm is not None) else None

    x_90, dose_90 = find_distal_percent_position(df['depth'], df['dose'], 90)

    if left_fwhm is not None and right_fwhm is not None:
        ax.hlines(y=half_max,
                  xmin=left_fwhm,
                  xmax=right_fwhm,
                  color='red',
                  linewidth=2,
                  linestyle='-')
        ax.text((left_fwhm + right_fwhm) / 2,
                half_max + 0.03,
                f'FWHM: {fwhm:.2f} mm',
                ha='center',
                va='bottom',
                color='red',
                fontsize=10,
                fontweight='bold')

    if x_90 is not None:
        ax.plot(x_90, dose_90, 'o', color='green', markersize=8)
        ax.text(x_90 + 0.5,
                dose_90,
                f'90%: {x_90:.2f} mm',
                ha='left',
                va='center',
                color='green',
                fontsize=10,
                fontweight='bold')

    ax.set_xlabel('Depth (mm)', fontsize=12)
    ax.set_ylabel('Normalized Dose', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 35)
    fig.tight_layout()
    return fig


def process_wedge_profile(config: MCWedgeConfig,
                          output_dir: str = ".",
                          pdf_path: Optional[str] = None) -> pd.DataFrame:
    """Run full wedge profile processing pipeline.

    Parameters
    ----------
    config : MCWedgeConfig
        Configuration with file paths and processing parameters.
    output_dir : str
        Directory for output files.
    pdf_path : str, optional
        Path for PDF report.

    Returns
    -------
    Processed DataFrame with transformed profiles.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(output_dir / f"mc_wedge_{config.facility}.pdf")

    source_paths = [config.dose_file, config.fluence_file, config.energy_file]
    report = PDFReport(pdf_path,
                       title=f"MC Wedge Profile ({config.facility.upper()})",
                       config_path=source_paths[0])

    # Load data
    if config.file_format == "csv":
        df_merged = load_csv_files(config.dose_file, config.fluence_file,
                                   config.energy_file)
    else:
        df_merged = load_dat_files(config.dose_file, config.fluence_file,
                                   config.energy_file)

    has_errors = 'dose_err' in df_merged.columns

    # Plot raw data
    fig = plot_triple_axis(
        df_merged,
        f'MC Data ({config.facility}): Dose, Fluence, and Kinetic Energy',
        show_errors=has_errors)
    report.add_figure(fig,
                      caption="Raw MC data (unfiltered)",
                      source_paths=source_paths)

    # Filter, reverse, shift
    df_mc = filter_reverse_shift(df_merged, config.depth_filter_min_mm,
                                 config.depth_filter_max_mm,
                                 config.depth_shift_mm)

    # Plot transformed data
    fig = plot_triple_axis(
        df_mc,
        f'MC Data ({config.facility}): Filtered, Reversed, Shifted',
        show_errors=has_errors)
    report.add_figure(
        fig,
        caption="Transformed MC data (filtered, reversed, shifted)",
        source_paths=source_paths)

    # Individual quantity plots (for big format with errors)
    if has_errors:
        fig = plot_individual_quantities(
            df_mc,
            f'MC Big Data ({config.facility}): Individual Quantities with Uncertainties'
        )
        report.add_figure(fig,
                          caption="Individual quantities with error bars",
                          source_paths=source_paths)

    # Dose with FWHM and 90% markers
    fig = plot_dose_with_markers(
        df_mc, f'MC Data ({config.facility}): Dose with FWHM and 90% Markers')
    report.add_figure(
        fig,
        caption="Normalized dose with FWHM and 90% distal markers",
        source_paths=source_paths)

    # Calculate metrics
    metrics = calculate_depth_dose_metrics(df_mc,
                                           f"MC {config.facility}",
                                           verbose=False)
    if metrics:
        metrics_text = (
            f"=== MC {config.facility} Metrics ===\n"
            f"Range (dose 90%): {metrics['dose_09']:.2f} mm\n"
            f"Falloff 90%-10%: {metrics['dose_drop_09_to_01']:.2f} mm\n"
            f"FWHM: {metrics['fwhm']:.2f} mm\n"
            f"Max/Plateau ratio: {metrics['first_point_inverse']:.4f}")
        report.add_text(metrics_text,
                        title=f"Dose Metrics ({config.facility})",
                        source_paths=source_paths)

    # Save output CSV
    csv_path = output_dir / config.output_csv
    df_mc.to_csv(str(csv_path), index=False)

    report.save()
    print(f"PDF report saved to: {pdf_path}")
    print(f"CSV data saved to: {csv_path}")

    return df_mc
