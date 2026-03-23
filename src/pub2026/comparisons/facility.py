"""Per-facility cross-modality comparison: MC vs EBT3 vs MBO.

Ported from notebooks: 1.0-aic144-comparisons.ipynb, 1.0-ccb-comparisons.ipynb
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.pub2026.config import ComparisonFacilityConfig, resolve_file
from src.pub2026.pdf_report import PDFReport
from src.pub2026.profile_metrics import (
    get_dense,
    find_fwhm_dense,
    find_distal_pct_dense,
    falloff_80_20,
    all_metrics_dense,
    align_mbo_to_reference,
)

# -- Main pipeline --


def compare_facility(config: ComparisonFacilityConfig,
                     output_dir: str = ".",
                     pdf_path: Optional[str] = None) -> Dict:
    """Run cross-modality comparison for one facility."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    facility = config.facility

    if pdf_path is None:
        pdf_path = str(out / f"comparison_{facility}.pdf")

    report = PDFReport(pdf_path,
                       title=f"Cross-modality Comparison: {facility.upper()}",
                       config_path=config.mc_csv)

    # Load data (resolve with output_dir fallback for intermediate outputs)
    df_mc = pd.read_csv(resolve_file(config.mc_csv, out))
    df_ebt = pd.read_csv(resolve_file(config.ebt_csv, out))
    mbo = np.load(str(resolve_file(config.mbo_npz, out)))
    mbo_x, mbo_y = mbo['x_mm'], mbo['profile_opt_smooth']

    # EBT interpolator
    ebt_interp = interp1d(df_ebt['x_mm'],
                          df_ebt['dose_Gy'],
                          kind='linear',
                          fill_value='extrapolate')
    ebt_at_0 = float(ebt_interp(config.normalize_at_x))
    ebt_max_x = float(df_ebt.loc[df_ebt['dose_Gy'].idxmax(), 'x_mm'])

    # Scale MC
    mc_interp = interp1d(df_mc['depth'],
                         df_mc['dose'],
                         kind='linear',
                         fill_value='extrapolate')
    mc_scale = ebt_at_0 / float(mc_interp(config.normalize_at_x))
    mc_dose_scaled = df_mc['dose'].values * mc_scale

    # Shift & scale MBO (two-foil)
    mbo_x_sh, mbo_scaled = align_mbo_to_reference(
        mbo_x, mbo_y, ebt_max_x, ebt_interp, match_x=config.normalize_at_x)

    # Optional single-foil MBO
    mbo_single_x_sh = mbo_single_scaled = None
    if config.mbo_single_foil_npz:
        mbo_s = np.load(str(resolve_file(config.mbo_single_foil_npz, out)))
        sx = mbo_s.get('profile_x_mm', mbo_s.get('x_mm'))
        sy = mbo_s.get('profile_values', mbo_s.get('profile_opt_smooth'))
        mbo_single_x_sh, mbo_single_scaled = align_mbo_to_reference(
            sx, sy, ebt_max_x, ebt_interp, match_x=config.normalize_at_x)

    # -- Plots --
    profiles: List[Tuple[str, np.ndarray, np.ndarray, str]] = [
        ('EBT3', df_ebt['x_mm'].values, df_ebt['dose_Gy'].values, 'black'),
        ('MC', df_mc['depth'].values, mc_dose_scaled, 'red'),
    ]
    if mbo_single_scaled is not None:
        profiles.append(
            ('MBO (single)', mbo_single_x_sh, mbo_single_scaled, 'blue'))
    profiles.append(
        ('MBO (two-foil)' if mbo_single_scaled is not None else 'MBO',
         mbo_x_sh, mbo_scaled,
         'green' if mbo_single_scaled is not None else 'blue'))

    for mbo_label, mbo_xp, mbo_yp, mbo_color in profiles[2:]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_ebt['x_mm'], df_ebt['dose_Gy'], 'k-', lw=1.5, label='EBT3')
        ax.plot(df_mc['depth'],
                mc_dose_scaled,
                'r-',
                lw=1.5,
                label='MC (scaled)')

        if config.mbo_gap_exclude_min_mm is not None:
            yp_gap = mbo_yp.copy()
            mask = (mbo_xp >= config.mbo_gap_exclude_min_mm) & (
                mbo_xp <= config.mbo_gap_exclude_max_mm)
            yp_gap[mask] = np.nan
            ax.plot(mbo_xp,
                    yp_gap,
                    color=mbo_color,
                    lw=1.5,
                    label=f'{mbo_label} (scaled)')
        else:
            ax.plot(mbo_xp,
                    mbo_yp,
                    color=mbo_color,
                    lw=1.5,
                    label=f'{mbo_label} (scaled)')

        ax.axvline(x=config.normalize_at_x,
                   color='gray',
                   ls='--',
                   lw=1,
                   alpha=0.5)

        # FWHM lines + metrics box
        metrics_text = []
        for name, xp, yp, col in [
            ('EBT3', df_ebt['x_mm'].values, df_ebt['dose_Gy'].values, 'black'),
            ('MC', df_mc['depth'].values, mc_dose_scaled, 'red'),
            (mbo_label, mbo_xp, mbo_yp, mbo_color)
        ]:
            xd, yd = get_dense(xp, yp)
            fw, xl, xr, hm = find_fwhm_dense(xd, yd)
            if fw is not None and xl is not None:
                ax.hlines(hm, xl, xr, colors=col, lw=2)
                ax.vlines([xl, xr], 0, hm, colors=col, ls=':', lw=1, alpha=0.7)
            x90 = find_distal_pct_dense(xd, yd, 0.9)
            fw_s = f'{fw:.2f}' if fw is not None else 'N/A'
            x90_s = f'{x90:.2f}' if x90 is not None else 'N/A'
            metrics_text.append(f'{name}: FWHM={fw_s} mm, R90={x90_s} mm')

        ax.text(0.03,
                0.5,
                '\n'.join(metrics_text),
                transform=ax.transAxes,
                fontsize=9,
                va='center',
                ha='left',
                bbox=dict(boxstyle='round',
                          facecolor='white',
                          alpha=0.9,
                          edgecolor='gray'),
                family='monospace')

        ax.set(xlabel='X [mm]',
               ylabel='Dose [Gy]',
               title=f'{facility.upper()}: MC vs EBT3 vs {mbo_label}',
               xlim=(-10, 30),
               ylim=(0, None))
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        report.add_figure(
            fig,
            caption=f'Dose profiles with {mbo_label}',
            source_paths=[config.mc_csv, config.ebt_csv, config.mbo_npz])

    # Summary metrics table
    rows = []
    for name, xp, yp in [('EBT3', df_ebt['x_mm'].values,
                          df_ebt['dose_Gy'].values),
                         ('MC', df_mc['depth'].values, mc_dose_scaled),
                         ('MBO', mbo_x_sh, mbo_scaled)]:
        m = all_metrics_dense(xp, yp)
        xd, yd = get_dense(xp, yp)
        dat0 = float(
            interp1d(xp, yp, fill_value='extrapolate')(config.normalize_at_x))
        rows.append({
            'Profile': name,
            'FWHM [mm]': m['fwhm'],
            'Range (90%) [mm]': m['x90'],
            'Max/Dose@X=0': float(yd.max()) / dat0 if dat0 != 0 else None,
            'Penumbra 80-20% [mm]': m['penumbra_80_20'],
        })
    if mbo_single_scaled is not None:
        m = all_metrics_dense(mbo_single_x_sh, mbo_single_scaled)
        xd, yd = get_dense(mbo_single_x_sh, mbo_single_scaled)
        dat0 = float(
            interp1d(mbo_single_x_sh,
                     mbo_single_scaled,
                     fill_value='extrapolate')(config.normalize_at_x))
        rows.append({
            'Profile': 'MBO (single)',
            'FWHM [mm]': m['fwhm'],
            'Range (90%) [mm]': m['x90'],
            'Max/Dose@X=0': float(yd.max()) / dat0 if dat0 != 0 else None,
            'Penumbra 80-20% [mm]': m['penumbra_80_20'],
        })

    df_metrics = pd.DataFrame(rows)
    report.add_table(df_metrics, title=f'{facility.upper()} Profile Metrics')
    report.save()
    print(f"PDF saved to: {pdf_path}")

    # Save CSV
    df_metrics.to_csv(out / f'comparison_metrics_{facility}.csv', index=False)

    return {
        'metrics': df_metrics,
        'mc_dose_scaled': mc_dose_scaled,
        'mbo_x_shifted': mbo_x_sh,
        'mbo_scaled': mbo_scaled
    }
