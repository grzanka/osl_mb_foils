"""MBO foil profile comparison: AIC-144 vs CCB.

Ported from notebook: 0.3-comparison.ipynb
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pub2026.config import MBOComparisonConfig, resolve_file
from src.pub2026.pdf_report import PDFReport
from src.pub2026.profile_metrics import (
    find_fwhm,
    find_distal_pct,
    calculate_profile_metrics,
    get_value_at_x,
)


def compare_mbo(config: MBOComparisonConfig,
                output_dir: str = ".",
                pdf_path: Optional[str] = None) -> Dict:
    """Compare MBO profiles from two facilities."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if pdf_path is None:
        pdf_path = str(out / "mbo_comparison.pdf")

    report = PDFReport(pdf_path,
                       title="MBO Profile Comparison",
                       config_path=str(config.aic144_profile_path))

    # Load profiles (resolve with output_dir fallback)
    aic_path = resolve_file(config.aic144_profile_path, out)
    ccb_path = resolve_file(config.ccb_profile_path, out)
    aic = np.load(str(aic_path))
    ccb = np.load(str(ccb_path))
    x_a, p_a = aic['x_mm'], aic['profile_opt_smooth']
    x_c, p_c = ccb['x_mm'], ccb['profile_opt_smooth']

    # Normalise to reference dose at reference X
    ref = config.reference_dose_gy
    rx = config.reference_x_mm
    dose_a = p_a * (ref / get_value_at_x(x_a, p_a, rx))
    dose_c = p_c * (ref / get_value_at_x(x_c, p_c, rx))

    # Metrics
    m_a = calculate_profile_metrics(dose_a, x_a)
    m_c = calculate_profile_metrics(dose_c, x_c)

    la, ra, hm_a = find_fwhm(dose_a, x_a)
    lc, rc, hm_c = find_fwhm(dose_c, x_c)
    x90a, d90a = find_distal_pct(dose_a, x_a, 90)
    x90c, d90c = find_distal_pct(dose_c, x_c, 90)
    x80a, d80a = find_distal_pct(dose_a, x_a, 80)
    x80c, d80c = find_distal_pct(dose_c, x_c, 80)
    x20a, _ = find_distal_pct(dose_a, x_a, 20)
    x20c, _ = find_distal_pct(dose_c, x_c, 20)

    # --- Main comparison plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_a, dose_a, 'r-', lw=1.5, label='AIC-144')
    ax.plot(x_c, dose_c, 'g-', lw=1.5, label='CCB')

    if la is not None and ra is not None:
        ax.hlines(hm_a, la, ra, colors='red', lw=2)
        ax.text((la + ra) / 2,
                hm_a + 0.15,
                f'FWHM: {m_a["fwhm"]:.2f} mm',
                ha='center',
                color='red',
                fontsize=9,
                fontweight='bold')
    if lc is not None and rc is not None:
        ax.hlines(hm_c, lc, rc, colors='green', lw=2)
        ax.text((lc + rc) / 2,
                hm_c - 0.15,
                f'FWHM: {m_c["fwhm"]:.2f} mm',
                ha='center',
                va='top',
                color='green',
                fontsize=9,
                fontweight='bold')

    if x90a is not None:
        ax.plot(x90a, d90a, 'ro', ms=8)
        ax.text(x90a + 0.3,
                d90a + 0.1,
                f'R90: {x90a:.2f}',
                color='red',
                fontsize=9,
                fontweight='bold')
    if x90c is not None:
        ax.plot(x90c, d90c, 'go', ms=8)
        ax.text(x90c + 0.3,
                d90c - 0.1,
                f'R90: {x90c:.2f}',
                va='top',
                color='green',
                fontsize=9,
                fontweight='bold')

    for xa80, xa20, col in [(x80a, x20a, 'red'), (x80c, x20c, 'green')]:
        if xa80 is not None and xa20 is not None:
            ax.axvspan(xa80, xa20, alpha=0.1, color=col)

    ax.axvline(rx, color='k', ls='--', lw=1, alpha=0.7)
    ax.plot(rx, ref, 'k*', ms=12)
    ax.set(xlabel='Depth [mm]',
           ylabel='Dose [Gy]',
           title=f'MBO Profile Comparison (norm {ref} Gy @ X={rx} mm)',
           xlim=(0, 40),
           ylim=(0, None))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    report.add_figure(
        fig,
        caption='Dose profiles with FWHM and range markers',
        source_paths=[config.aic144_profile_path, config.ccb_profile_path])

    # --- Normalised falloff plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x_a, dose_a / np.max(dose_a) * 100, 'r-', lw=1.5, label='AIC-144')
    ax2.plot(x_c, dose_c / np.max(dose_c) * 100, 'g-', lw=1.5, label='CCB')
    for lvl, st in [(90, '--'), (80, ':'), (50, '-'), (20, ':'), (10, '--')]:
        ax2.axhline(lvl, color='gray', ls=st, lw=0.8, alpha=0.5)
        ax2.text(25.5, lvl, f'{lvl}%', va='center', fontsize=8, color='gray')
    ax2.set(xlabel='Depth [mm]',
            ylabel='Relative Dose [%]',
            title='Distal Falloff Comparison',
            xlim=(20, 40),
            ylim=(0, 110))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    report.add_figure(fig2, caption='Normalised distal falloff (80%–20%)')

    # Metrics summary
    metrics_df = pd.DataFrame([
        {
            'Metric': 'FWHM [mm]',
            'AIC-144': f'{m_a["fwhm"]:.2f}',
            'CCB': f'{m_c["fwhm"]:.2f}'
        },
        {
            'Metric': 'Range 90% [mm]',
            'AIC-144':
            f'{m_a["pos_90"]:.2f}' if m_a["pos_90"] is not None else 'N/A',
            'CCB':
            f'{m_c["pos_90"]:.2f}' if m_c["pos_90"] is not None else 'N/A'
        },
        {
            'Metric': 'Falloff 80-20% [mm]',
            'AIC-144': f'{m_a["dist_80_20"]:.2f}',
            'CCB': f'{m_c["dist_80_20"]:.2f}'
        },
    ])
    report.add_table(metrics_df, title='Profile Metrics')

    report.save()
    print(f"PDF saved to: {pdf_path}")

    # Save CSVs
    for label, x, dose, m in [('aic144', x_a, dose_a, m_a),
                              ('ccb', x_c, dose_c, m_c)]:
        df = pd.DataFrame({'x_mm': x, 'dose_Gy': dose})
        df.to_csv(out / f'mbo_profile_{label}.csv', index=False)

    metrics_df = pd.DataFrame({
        'Metric': [
            'FWHM [mm]', 'Range 90% [mm]', 'Distal 90%-10% [mm]',
            'Distal 80%-20% [mm]'
        ],
        'AIC-144':
        [m_a['fwhm'], m_a['pos_90'], m_a['dist_90_10'], m_a['dist_80_20']],
        'CCB':
        [m_c['fwhm'], m_c['pos_90'], m_c['dist_90_10'], m_c['dist_80_20']],
    })
    metrics_df.to_csv(out / 'mbo_profile_metrics.csv', index=False)

    return {
        'metrics_aic144': m_a,
        'metrics_ccb': m_c,
        'dose_aic144': dose_a,
        'dose_ccb': dose_c
    }
