"""Cross-facility summary comparison: CCB vs AIC-144 across all modalities.

Ported from notebook: 2.0-comparisons.ipynb
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.pub2026.config import ComparisonSummaryConfig, resolve_file
from src.pub2026.pdf_report import PDFReport
from src.pub2026.profile_metrics import (
    all_metrics_dense,
    align_mbo_to_reference,
)


def compare_summary(config: ComparisonSummaryConfig,
                    output_dir: str = ".",
                    pdf_path: Optional[str] = None) -> Dict:
    """Create cross-facility summary comparison report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(out / "comparison_summary.pdf")

    report = PDFReport(pdf_path,
                       title="Cross-facility Summary Comparison",
                       config_path=config.mc_ccb_csv)

    # Load all data (resolve with output_dir fallback)
    mc_ccb = pd.read_csv(resolve_file(config.mc_ccb_csv, out))
    mc_aic = pd.read_csv(resolve_file(config.mc_aic144_csv, out))
    ebt_ccb = pd.read_csv(resolve_file(config.ebt_ccb_csv, out))
    ebt_aic = pd.read_csv(resolve_file(config.ebt_aic144_csv, out))
    mbo_ccb_d = np.load(str(resolve_file(config.mbo_ccb_npz, out)))
    mbo_aic_d = np.load(str(resolve_file(config.mbo_aic144_npz, out)))

    mbo_ccb_x, mbo_ccb_y = mbo_ccb_d['x_mm'], mbo_ccb_d['profile_opt_smooth']
    mbo_aic_x, mbo_aic_y = mbo_aic_d['x_mm'], mbo_aic_d['profile_opt_smooth']

    # Align MBO to EBT
    ebt_ccb_interp = interp1d(ebt_ccb['x_mm'],
                              ebt_ccb['dose_Gy'],
                              fill_value='extrapolate')
    ebt_aic_interp = interp1d(ebt_aic['x_mm'],
                              ebt_aic['dose_Gy'],
                              fill_value='extrapolate')
    ebt_ccb_max_x = float(ebt_ccb.loc[ebt_ccb['dose_Gy'].idxmax(), 'x_mm'])
    ebt_aic_max_x = float(ebt_aic.loc[ebt_aic['dose_Gy'].idxmax(), 'x_mm'])

    mbo_ccb_xs, mbo_ccb_sc = align_mbo_to_reference(mbo_ccb_x, mbo_ccb_y,
                                                    ebt_ccb_max_x,
                                                    ebt_ccb_interp)
    mbo_aic_xs, mbo_aic_sc = align_mbo_to_reference(mbo_aic_x, mbo_aic_y,
                                                    ebt_aic_max_x,
                                                    ebt_aic_interp)

    # Scale MC to match EBT at X=0
    mc_ccb_interp = interp1d(mc_ccb['depth'],
                             mc_ccb['dose'],
                             fill_value='extrapolate')
    mc_aic_interp = interp1d(mc_aic['depth'],
                             mc_aic['dose'],
                             fill_value='extrapolate')
    s_ccb = float(ebt_ccb_interp(0)) / float(mc_ccb_interp(0))
    s_aic = float(ebt_aic_interp(0)) / float(mc_aic_interp(0))
    mc_ccb_sc = mc_ccb['dose'].values * s_ccb
    mc_aic_sc = mc_aic['dose'].values * s_aic

    # -- Per-modality cross-facility plots (EBT, MBO, MC) --
    for title, pairs in [
        ('EBT3', [
            ('CCB', ebt_ccb['x_mm'].values, ebt_ccb['dose_Gy'].values, 'b'),
            ('AIC144', ebt_aic['x_mm'].values, ebt_aic['dose_Gy'].values, 'r'),
        ]),
        ('MBO', [
            ('CCB', mbo_ccb_xs, mbo_ccb_sc, 'b'),
            ('AIC144', mbo_aic_xs, mbo_aic_sc, 'r'),
        ]),
        ('MC', [
            ('CCB', mc_ccb['depth'].values, mc_ccb_sc, 'b'),
            ('AIC144', mc_aic['depth'].values, mc_aic_sc, 'r'),
        ]),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        mtext = []
        for lbl, xp, yp, col in pairs:
            ax.plot(xp, yp, f'{col}-', lw=1.5, label=lbl)
            m = all_metrics_dense(xp, yp)
            mtext.append(f"{lbl}:\n  FWHM:     {m['fwhm']:.2f} mm\n"
                         f"  Range 90%: {m['x90']:.2f} mm\n"
                         f"  Falloff:  {m['falloff']:.2f} mm" if m['fwhm'] and
                         m['x90'] and m['falloff'] else f"{lbl}: metrics N/A")
        ax.set(xlabel='X [mm]',
               ylabel='Dose [Gy]',
               title=f'{title}: CCB vs AIC144',
               xlim=(-10, 30))
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02,
                0.98,
                '\n\n'.join(mtext),
                transform=ax.transAxes,
                fontsize=9,
                va='top',
                ha='left',
                bbox=props,
                family='monospace')
        fig.tight_layout()
        report.add_figure(fig, caption=f'{title}: CCB vs AIC144')

    # -- Per-facility paired plots (MC vs EBT, EBT vs MBO, MC vs MBO) --
    for pair_title, data_sets in [
        ('MC vs EBT3', [
            ('CCB', [('EBT3', ebt_ccb['x_mm'].values,
                      ebt_ccb['dose_Gy'].values, 'k'),
                     ('MC', mc_ccb['depth'].values, mc_ccb_sc, 'r')]),
            ('AIC144', [('EBT3', ebt_aic['x_mm'].values,
                         ebt_aic['dose_Gy'].values, 'k'),
                        ('MC', mc_aic['depth'].values, mc_aic_sc, 'r')]),
        ]),
        ('EBT3 vs MBO', [
            ('CCB', [('EBT3', ebt_ccb['x_mm'].values,
                      ebt_ccb['dose_Gy'].values, 'k'),
                     ('MBO', mbo_ccb_xs, mbo_ccb_sc, 'b')]),
            ('AIC144', [('EBT3', ebt_aic['x_mm'].values,
                         ebt_aic['dose_Gy'].values, 'k'),
                        ('MBO', mbo_aic_xs, mbo_aic_sc, 'b')]),
        ]),
        ('MC vs MBO', [
            ('CCB', [('MC', mc_ccb['depth'].values, mc_ccb_sc, 'r'),
                     ('MBO', mbo_ccb_xs, mbo_ccb_sc, 'b')]),
            ('AIC144', [('MC', mc_aic['depth'].values, mc_aic_sc, 'r'),
                        ('MBO', mbo_aic_xs, mbo_aic_sc, 'b')]),
        ]),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (fac, curves) in zip(axes, data_sets):
            mtext = []
            for lbl, xp, yp, col in curves:
                ax.plot(xp, yp, f'{col}-', lw=1.5, label=lbl)
                m = all_metrics_dense(xp, yp)
                fw_s = f"{m['fwhm']:.2f}" if m['fwhm'] else 'N/A'
                x9_s = f"{m['x90']:.2f}" if m['x90'] else 'N/A'
                fl_s = f"{m['falloff']:.2f}" if m['falloff'] else 'N/A'
                mtext.append(
                    f"{lbl}:\n  FWHM:     {fw_s} mm\n  Range 90%: {x9_s} mm\n  Falloff:  {fl_s} mm"
                )
            ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
            ax.set(xlabel='X [mm]',
                   ylabel='Dose [Gy]',
                   title=f'{fac}: {pair_title}',
                   xlim=(-10, 30))
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.02,
                    0.98,
                    '\n\n'.join(mtext),
                    transform=ax.transAxes,
                    fontsize=8,
                    va='top',
                    ha='left',
                    bbox=props,
                    family='monospace')
        fig.tight_layout()
        report.add_figure(fig, caption=pair_title)

    # -- Combined normalised plot --
    fig, ax = plt.subplots(figsize=(14, 8))
    for col, fac, pairs in [
        ('b', 'CCB', [
            (mbo_ccb_xs, mbo_ccb_sc, '-', 'CCB - MBO'),
            (ebt_ccb['x_mm'].values, ebt_ccb['dose_Gy'].values, '--',
             'CCB - EBT3'),
            (mc_ccb['depth'].values, mc_ccb_sc, ':', 'CCB - MC'),
        ]),
        ('r', 'AIC144', [
            (mbo_aic_xs, mbo_aic_sc, '-', 'AIC144 - MBO'),
            (ebt_aic['x_mm'].values, ebt_aic['dose_Gy'].values, '--',
             'AIC144 - EBT3'),
            (mc_aic['depth'].values, mc_aic_sc, ':', 'AIC144 - MC'),
        ]),
    ]:
        for xp, yp, ls, label in pairs:
            yn = yp / yp.max()
            ax.plot(xp, yn, color=col, ls=ls, lw=2, label=label)
    ax.axvline(0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.3)
    ax.axhline(0.9, color='gray', ls=':', lw=1, alpha=0.3)
    ax.set(
        xlabel='Distance from center [mm]',
        ylabel='Relative Dose',
        title='Summary: All Profiles\nMBO (solid), EBT3 (dashed), MC (dotted)',
        xlim=(-10, 50),
        ylim=(-0.05, 1.15))
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    report.add_figure(fig, caption='Combined normalised profiles')

    # -- Metrics summary table --
    rows = []
    all_data = [
        ('CCB', 'EBT3', ebt_ccb['x_mm'].values, ebt_ccb['dose_Gy'].values),
        ('AIC144', 'EBT3', ebt_aic['x_mm'].values, ebt_aic['dose_Gy'].values),
        ('CCB', 'MC', mc_ccb['depth'].values, mc_ccb_sc),
        ('AIC144', 'MC', mc_aic['depth'].values, mc_aic_sc),
        ('CCB', 'MBO', mbo_ccb_xs, mbo_ccb_sc),
        ('AIC144', 'MBO', mbo_aic_xs, mbo_aic_sc),
    ]
    for fac, dtype, xp, yp in all_data:
        m = all_metrics_dense(xp, yp)
        rows.append({
            'Facility': fac,
            'Data Type': dtype,
            'FWHM [mm]': m['fwhm'],
            'Range 90% [mm]': m['x90'],
            'Falloff 80-20 [mm]': m['falloff']
        })

    df_all = pd.DataFrame(rows)
    report.add_table(df_all, title='Summary Metrics')

    # Differences text
    diff_lines = []
    for cmp_label, d1, d2 in [('MC vs EBT3', 'MC', 'EBT3'),
                              ('MBO vs EBT3', 'MBO', 'EBT3'),
                              ('MBO vs MC', 'MBO', 'MC')]:
        diff_lines.append(f'\n--- {cmp_label} ---')
        for fac in ['CCB', 'AIC144']:
            r1 = df_all[(df_all['Facility'] == fac)
                        & (df_all['Data Type'] == d1)]
            r2 = df_all[(df_all['Facility'] == fac)
                        & (df_all['Data Type'] == d2)]
            if r1.empty or r2.empty:
                continue
            for col in ['FWHM [mm]', 'Range 90% [mm]', 'Falloff 80-20 [mm]']:
                v1, v2 = r1[col].values[0], r2[col].values[0]
                if pd.notna(v1) and pd.notna(v2) and v2 != 0:
                    diff_lines.append(
                        f'  {fac} {col}: {v1 - v2:+.2f} mm ({(v1/v2 - 1)*100:+.1f}%)'
                    )

    report.add_text('\n'.join(diff_lines), title='Metric Differences')
    report.save()
    print(f"PDF saved to: {pdf_path}")

    # Save metrics
    df_all.to_csv(out / 'comparison_summary_metrics.csv', index=False)
    return {'metrics': df_all}
