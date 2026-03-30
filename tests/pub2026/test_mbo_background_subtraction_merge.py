"""Tests for the post-alignment MBO background-subtraction merge pipeline."""

import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import numpy as np

from src.data.analysis import Circle
from src.pub2026.config import MBOBackgroundSubtractionMergeConfig
from src.pub2026.mbo.background_subtraction_merge import merge_background_subtraction


def _gaussian(shape, circle, amplitude, sigma, background=0.0):
    """Create a 2D Gaussian centred on circle."""
    yy, xx = np.indices(shape)
    rr2 = (xx - circle.x)**2 + (yy - circle.y)**2
    return background + amplitude * np.exp(-rr2 / (2.0 * sigma**2))


def _write_aligned_npz(path: Path, circles, images):
    """Write an aligned NPZ in the same format as align_mbo output."""
    save_dict = {}
    for foil_id, circle in circles.items():
        save_dict[f'foil_{foil_id}_image'] = images[foil_id]
        save_dict[f'foil_{foil_id}_circle_x'] = circle.x
        save_dict[f'foil_{foil_id}_circle_y'] = circle.y
        save_dict[f'foil_{foil_id}_circle_r'] = circle.r
        save_dict[f'foil_{foil_id}_rotation_deg'] = 0.0
        save_dict[f'foil_{foil_id}_edge_source'] = 'raw'
    np.savez_compressed(path, **save_dict)


def _write_bg_subtraction_npz(path: Path, foil_ids, shape, circles):
    """Write a bg-subtraction NPZ with synthetic scenario A/B data."""
    save_dict = {'background_level': 5.0}
    for fid in foil_ids:
        c = circles[fid]
        # Scenario A: difference image — Gaussian signal on left, lower on right
        diff = _gaussian(shape, c, amplitude=200.0, sigma=40.0)
        save_dict[f'scenario_a_foil_{fid}_reference_centered'] = np.full(shape, 5.0)
        save_dict[f'scenario_a_foil_{fid}_target_centered'] = diff + 5.0
        save_dict[f'scenario_a_foil_{fid}_difference'] = diff

        # Scenario B: ratio image — close to 1.0 with a bump
        ratio = 1.0 + 0.1 * _gaussian(shape, c, amplitude=1.0, sigma=40.0) / 200.0
        save_dict[f'scenario_b_foil_{fid}_ratio'] = ratio
        save_dict[f'scenario_b_foil_{fid}_circle_x'] = c.x
        save_dict[f'scenario_b_foil_{fid}_circle_y'] = c.y
        save_dict[f'scenario_b_foil_{fid}_circle_r'] = c.r
        save_dict[f'scenario_b_foil_{fid}_normalization_mean'] = 1.0
    np.savez_compressed(path, **save_dict)


def test_merge_background_subtraction_runs_and_saves(tmp_path):
    """End-to-end: synthetic data with rotation-based merge → PDF + NPZ."""
    shape = (128, 128)
    circle_1 = Circle(x=64.0, y=64.0, r=40.0)
    circle_2 = Circle(x=64.0, y=64.0, r=40.0)

    # Build aligned images with signal and a bottom edge (signal drops below y=100)
    img1 = _gaussian(shape, circle_1, amplitude=500.0, sigma=30.0, background=10.0)
    img1[100:, :] = 0.0  # bottom edge at y=100
    img2 = _gaussian(shape, circle_2, amplitude=500.0, sigma=30.0, background=10.0)
    img2[100:, :] = 0.0  # bottom edge at y=100

    circles = {1: circle_1, 2: circle_2}
    images = {1: img1, 2: img2}

    tgt_npz = tmp_path / 'mbo_align_target.npz'
    _write_aligned_npz(tgt_npz, circles, images)

    bg_sub_npz = tmp_path / 'mbo_bg_sub.npz'
    _write_bg_subtraction_npz(bg_sub_npz, [1, 2], shape, circles)

    config = MBOBackgroundSubtractionMergeConfig(
        facility='synthetic',
        bg_subtraction_npz=str(bg_sub_npz),
        target_npz=str(tgt_npz),
        pixel_size_mm=0.1,
        foil_ids=[1, 2],
        edge_threshold=50.0,
        merge_margin=0,
        vmax_scenario_a=300.0,
        vmax_scenario_b=2.0,
        output_height_mm=10.0,
        output_width_mm=15.0,
        profile_y_offsets_mm=[-2.0, 0.0, 2.0],
        profile_strip_width_mm=1.0,
        profile_y_min_mm=3.0,
        profile_y_max_mm=7.0,
        section1_x_min=1.0,
        section1_x_max=5.0,
        section2_x_min=10.0,
        section2_x_max=14.0,
        smoothing_sigma_px=2,
        output_npz='test_merge.npz',
        output_profile_npz='test_profile.npz',
    )

    result = merge_background_subtraction(
        config,
        output_dir=str(tmp_path),
        pdf_path=str(tmp_path / 'test_merge.pdf'),
    )

    # Verify outputs exist
    assert Path(result['pdf_path']).exists()
    assert Path(result['output_npz']).exists()

    # Verify NPZ contents
    saved = np.load(result['output_npz'])
    assert 'scenario_a_merged' in saved
    assert 'scenario_a_merged_opt' in saved
    assert 'scenario_b_merged' in saved
    assert 'scenario_b_merged_opt' in saved

    # Merged images should be 2D
    assert saved['scenario_a_merged'].ndim == 2
    assert saved['scenario_b_merged'].ndim == 2

    # Optimal shift should be finite
    assert np.isfinite(saved['scenario_a_optimal_shift_mm'])
    assert np.isfinite(saved['scenario_b_optimal_shift_mm'])

    # Verify profile NPZ (scenario B optimised, match-step compatible)
    assert Path(result['output_profile_npz']).exists()
    prof = np.load(result['output_profile_npz'])
    assert 'x_mm' in prof
    assert 'profile_opt_smooth' in prof
    assert prof['x_mm'].ndim == 1
    assert prof['profile_opt_smooth'].ndim == 1
    assert len(prof['x_mm']) == len(prof['profile_opt_smooth'])
