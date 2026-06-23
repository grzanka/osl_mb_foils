"""MC depth-dose validation: compare experimental Bragg peak with MC simulation.

Ported from notebook: 0.1_ccb_mc_depth_validation.ipynb
"""

from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.data.metrics import calculate_depth_dose_metrics
from src.pub2026.config import MCDepthValidationConfig
from src.pub2026.pdf_report import PDFReport

# Experimental Bragg peak data (ion chamber measurement at CCB)
CCB_EXPERIMENTAL_DATA = """1.99	0.2570
3.99	0.2631
5.99	0.2681
7.99	0.2737
9.99	0.2809
11.99	0.2895
13.99	0.2990
15.99	0.3121
17.99	0.3273
19.99	0.3464
21.99	0.3726
23.99	0.4080
24.99	0.4320
25.89	0.4593
26.29	0.4722
26.69	0.4882
27.09	0.5057
27.49	0.5285
27.89	0.5488
28.34	0.5786
28.79	0.6183
29.04	0.6440
29.29	0.6732
29.54	0.7147
29.79	0.7527
30.04	0.8060
30.29	0.8609
30.39	0.8867
30.49	0.9170
30.59	0.9394
30.69	0.9611
30.79	0.9775
30.89	0.9907
30.99	1.0000
31.09	0.9969
31.19	0.9918
31.29	0.9767
31.39	0.9506
31.49	0.9161
31.59	0.8811
31.69	0.8314
31.89	0.7251
32.09	0.6001
32.29	0.4737
32.49	0.3549
32.69	0.2512
32.89	0.1638
33.09	0.0979
33.29	0.0540
33.49	0.0274
33.69	0.0133
33.89	0.0046
34.09	0.0016
34.29	0.0006
34.69	0.0000
35.09	0.0000
35.49	0.0000
"""


def load_experimental_data() -> pd.DataFrame:
    """Load the embedded experimental Bragg peak data."""
    return pd.read_csv(StringIO(CCB_EXPERIMENTAL_DATA),
                       sep='\t',
                       header=None,
                       names=['x', 'y'])


def load_mc_depth_dose(csv_path: str) -> pd.DataFrame:
    """Load MC depth-dose CSV and normalize.

    Parameters
    ----------
    csv_path : str
        Path to the MC CSV file (graph_z_profile_DOSE.csv).

    Returns
    -------
    DataFrame with columns 'depth' (mm) and 'dose' (normalized to max=1).
    """
    df = pd.read_csv(csv_path, comment='#', skiprows=0)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        col: 'depth' if 'Position' in col else 'dose'
        for col in df.columns
    })

    if 'depth' in df.columns:
        df['depth'] = df['depth'] * 10  # cm to mm
    if 'dose' in df.columns:
        df['dose'] = df['dose'] / df['dose'].max()

    return df


def calculate_ratio(df_exp: pd.DataFrame,
                    df_mc: pd.DataFrame,
                    threshold: float = 0.1) -> Tuple[pd.Series, Dict]:
    """Calculate exp/MC ratio at experimental data points.

    Returns
    -------
    ratio : Series
        Ratio values where exp dose > threshold.
    stats : dict
        Mean, min, max, std, n_points of the ratio.
    """
    mc_interp = interp1d(df_mc['depth'],
                         df_mc['dose'],
                         kind='linear',
                         bounds_error=False,
                         fill_value='extrapolate')

    mc_at_exp = mc_interp(df_exp['x'])
    valid_mask = df_exp['y'] > threshold

    ratio = pd.Series(index=df_exp.index, dtype=float)
    ratio[valid_mask] = df_exp.loc[valid_mask, 'y'] / mc_at_exp[valid_mask]

    stats = {
        'n_points': int(valid_mask.sum()),
        'mean': float(ratio[valid_mask].mean()),
        'min': float(ratio[valid_mask].min()),
        'max': float(ratio[valid_mask].max()),
        'std': float(ratio[valid_mask].std()),
    }
    return ratio, stats


def _plot_experimental(df_exp: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_exp['x'],
            df_exp['y'],
            marker='o',
            linestyle='-',
            linewidth=1.5,
            markersize=4)
    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel('Normalized Dose')
    ax.set_title('Experimental Bragg Peak (CCB)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_mc(df_mc: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_mc['depth'],
            df_mc['dose'],
            marker='s',
            linestyle='-',
            linewidth=1.5,
            markersize=5,
            label='MC Data')
    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel('Normalized Dose')
    ax.set_title('MC Depth-Dose Profile (CCB)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_comparison(df_exp: pd.DataFrame, df_mc: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(df_exp['x'],
            df_exp['y'],
            marker='o',
            linestyle='-',
            linewidth=1.5,
            markersize=4,
            label='Experimental Data')
    ax.plot(df_mc['depth'],
            df_mc['dose'],
            marker='s',
            linestyle='-',
            linewidth=1.5,
            markersize=5,
            label='MC Data')
    ax.set_xlabel('Depth (mm)')
    ax.set_ylabel('Normalized Dose')
    ax.set_title('Experimental vs MC Depth-Dose Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_ratio(df_exp: pd.DataFrame, df_mc: pd.DataFrame, ratio: pd.Series,
                stats: Dict, threshold: float) -> plt.Figure:
    valid_mask = df_exp['y'] > threshold
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(df_exp['x'],
             df_exp['y'],
             marker='o',
             linestyle='-',
             linewidth=1.5,
             markersize=4,
             label='Experimental Data')
    ax1.plot(df_mc['depth'],
             df_mc['dose'],
             marker='s',
             linestyle='-',
             linewidth=1.5,
             markersize=5,
             label='MC Data')
    ax1.axhline(y=threshold,
                color='r',
                linestyle='--',
                alpha=0.5,
                label=f'Threshold ({threshold})')
    ax1.set_xlabel('Depth (mm)')
    ax1.set_ylabel('Normalized Dose')
    ax1.set_title('Comparison of Experimental vs MC Data')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ratio_percent = (ratio[valid_mask] - 1.0) * 100
    mean_percent = (stats['mean'] - 1.0) * 100

    ax2.plot(df_exp.loc[valid_mask, 'x'],
             ratio_percent,
             marker='o',
             linestyle='-',
             linewidth=1.5,
             markersize=4,
             color='green',
             label='Exp/MC Ratio (dose > threshold)')
    ax2.axhline(y=0.0,
                color='k',
                linestyle='--',
                alpha=0.5,
                label='No difference')
    ax2.axhline(y=mean_percent,
                color='r',
                linestyle='--',
                alpha=0.5,
                label=f'Mean: {mean_percent:.2f}%')
    ax2.set_xlabel('Depth (mm)')
    ax2.set_ylabel('Difference from ratio 1 (%)')
    ax2.set_title(f'Relative Difference (dose > {threshold})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    return fig


def _build_metrics_table(metrics_exp: Dict, metrics_mc: Dict) -> pd.DataFrame:
    features = [
        ("Range (dose 90%)", "mm", 'dose_09'),
        ("Falloff 90% to 10%", "mm", 'dose_drop_09_to_01'),
        ("FWHM", "mm", 'fwhm'),
        ("Max to Plateau Ratio", "-", 'first_point_inverse'),
    ]

    rows = []
    for name, unit, key in features:
        exp_val = metrics_exp.get(key)
        mc_val = metrics_mc.get(key)
        if exp_val is not None and mc_val is not None:
            abs_diff = mc_val - exp_val
            rel_diff = (abs_diff /
                        exp_val) * 100 if exp_val != 0 else float('nan')
            rows.append({
                'Feature': name,
                'Unit': unit,
                'Experimental': f'{exp_val:.4f}',
                'MC': f'{mc_val:.4f}',
                'Abs Diff': f'{abs_diff:.4f}',
                'Rel Diff (%)': f'{rel_diff:.2f}',
            })
        else:
            rows.append({
                'Feature': name,
                'Unit': unit,
                'Experimental': 'N/A',
                'MC': 'N/A',
                'Abs Diff': 'N/A',
                'Rel Diff (%)': 'N/A',
            })

    return pd.DataFrame(rows)


def validate_depth_dose(config: MCDepthValidationConfig,
                        output_dir: str = ".",
                        pdf_path: Optional[str] = None) -> pd.DataFrame:
    """Run full depth-dose validation pipeline.

    Parameters
    ----------
    config : MCDepthValidationConfig
        Configuration with data paths and threshold.
    output_dir : str
        Directory for output files (H5, CSV).
    pdf_path : str, optional
        Path for PDF report. If None, uses output_dir/mc_depth_validation.pdf.

    Returns
    -------
    DataFrame with comparison metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(output_dir / "mc_depth_validation.pdf")

    source_paths = [config.mc_csv_path]
    report = PDFReport(pdf_path,
                       title="MC Depth-Dose Validation (CCB)",
                       config_path=config.mc_csv_path)

    # Load data
    df_exp = load_experimental_data()
    df_mc = load_mc_depth_dose(config.mc_csv_path)

    # Plot experimental data
    fig = _plot_experimental(df_exp)
    report.add_figure(fig,
                      caption="Experimental Bragg peak (ion chamber, CCB)",
                      source_paths=["Embedded experimental data"])

    # Plot MC data
    fig = _plot_mc(df_mc)
    report.add_figure(fig,
                      caption="MC depth-dose profile",
                      source_paths=source_paths)

    # Comparison plot
    fig = _plot_comparison(df_exp, df_mc)
    report.add_figure(fig,
                      caption="Experimental vs MC comparison",
                      source_paths=source_paths)

    # Calculate metrics
    metrics_exp = calculate_depth_dose_metrics(df_exp,
                                               "Experimental",
                                               verbose=False)
    mc_interp = interp1d(df_mc['depth'],
                         df_mc['dose'],
                         kind='linear',
                         bounds_error=False,
                         fill_value='extrapolate')
    metrics_mc = calculate_depth_dose_metrics(df_mc,
                                              "MC",
                                              ref_x_coord=df_exp['x'].iloc[0],
                                              interp_func=mc_interp,
                                              verbose=False)

    # Ratio analysis
    ratio, ratio_stats = calculate_ratio(df_exp, df_mc, config.threshold)
    fig = _plot_ratio(df_exp, df_mc, ratio, ratio_stats, config.threshold)
    report.add_figure(fig,
                      caption="Exp/MC ratio analysis",
                      source_paths=source_paths)

    # Metrics table
    metrics_df = _build_metrics_table(metrics_exp, metrics_mc)
    report.add_table(metrics_df,
                     title="Metrics Comparison: Experimental vs MC",
                     source_paths=source_paths)

    # Ratio statistics text
    ratio_text = (
        f"Ratio Statistics (exp/MC where dose > {config.threshold}):\n"
        f"  Points: {ratio_stats['n_points']}\n"
        f"  Mean:   {ratio_stats['mean']:.4f}\n"
        f"  Min:    {ratio_stats['min']:.4f}\n"
        f"  Max:    {ratio_stats['max']:.4f}\n"
        f"  Std:    {ratio_stats['std']:.4f}")
    report.add_text(ratio_text,
                    title="Ratio Statistics",
                    source_paths=source_paths)

    # Save HDF5 output
    h5_path = output_dir / config.output_h5
    df_mc.to_hdf(str(h5_path), key='mc_depth_dose', mode='w')
    df_exp.to_hdf(str(h5_path), key='exp_depth_dose', mode='a')

    report.save()
    print(f"PDF report saved to: {pdf_path}")
    print(f"HDF5 data saved to: {h5_path}")

    return metrics_df
