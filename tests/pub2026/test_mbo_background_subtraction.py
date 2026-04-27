"""Tests for the post-alignment MBO background-subtraction pipeline."""

from pathlib import Path

import numpy as np

from src.data.analysis import Circle
from src.pub2026.config import MBOBackgroundSubtractionConfig
from src.pub2026.mbo.background_subtraction import explore_background_subtraction


def _gaussian(shape, circle, amplitude, sigma, background):
    yy, xx = np.indices(shape)
    rr2 = (xx - circle.x)**2 + (yy - circle.y)**2
    return background + amplitude * np.exp(-rr2 / (2.0 * sigma**2))


def _write_aligned_npz(path: Path, circles, images):
    save_dict = {}
    for foil_id, circle in circles.items():
        save_dict[f'foil_{foil_id}_image'] = images[foil_id]
        save_dict[f'foil_{foil_id}_circle_x'] = circle.x
        save_dict[f'foil_{foil_id}_circle_y'] = circle.y
        save_dict[f'foil_{foil_id}_circle_r'] = circle.r
        save_dict[f'foil_{foil_id}_rotation_deg'] = 0.0
        save_dict[f'foil_{foil_id}_edge_source'] = 'raw'
    np.savez_compressed(path, **save_dict)


def test_background_subtraction_pipeline_centers_and_saves(tmp_path):
    shape = (64, 64)
    circle_ref = Circle(x=31.0, y=30.0, r=18.0)
    circle_tgt = Circle(x=34.0, y=32.0, r=18.0)
    circles_ref = {1: circle_ref, 2: circle_ref, 3: circle_ref, 4: circle_ref}
    circles_tgt = {1: circle_tgt, 2: circle_tgt, 3: circle_tgt, 4: circle_tgt}

    images_ref = {
        1: _gaussian(shape,
                     circle_ref,
                     amplitude=20.0,
                     sigma=6.0,
                     background=5.0),
        2: _gaussian(shape,
                     circle_ref,
                     amplitude=16.0,
                     sigma=7.0,
                     background=5.0),
        3: np.full(shape, 5.0),
        4: np.full(shape, 5.0),
    }
    images_tgt = {
        1: _gaussian(shape,
                     circle_tgt,
                     amplitude=20.0,
                     sigma=6.0,
                     background=5.0),
        2: _gaussian(shape,
                     circle_tgt,
                     amplitude=16.0,
                     sigma=7.0,
                     background=5.0),
        3: np.full(shape, 5.0),
        4: np.full(shape, 5.0),
    }

    reference_npz = tmp_path / 'mbo_align_ref.npz'
    target_npz = tmp_path / 'mbo_align_target.npz'
    _write_aligned_npz(reference_npz, circles_ref, images_ref)
    _write_aligned_npz(target_npz, circles_tgt, images_tgt)

    config = MBOBackgroundSubtractionConfig(
        facility='synthetic',
        reference_npz=str(reference_npz),
        target_npz=str(target_npz),
        reference_label='03-17',
        target_label='03-19',
        output_npz='synthetic_bg.npz',
        smoothing_sigma_px=2.0,
    )

    result = explore_background_subtraction(config,
                                            output_dir=str(tmp_path),
                                            pdf_path=str(tmp_path /
                                                         'synthetic.pdf'))

    assert Path(result['pdf_path']).exists()
    assert Path(result['output_npz']).exists()

    saved = np.load(result['output_npz'])
    assert np.isclose(saved['background_level'], 5.0)
    assert np.nanmax(np.abs(saved['scenario_a_foil_1_difference'])) < 1.0
    assert np.nanmax(np.abs(saved['scenario_a_foil_2_difference'])) < 1.0

    center_y = int(round(float(saved['scenario_b_foil_1_circle_y'])))
    center_x = int(round(float(saved['scenario_b_foil_1_circle_x'])))
    assert np.isfinite(saved['scenario_b_foil_1_ratio'][center_y, center_x])


def test_background_subtraction_sets_zero_when_dividing_by_zero(tmp_path):
    shape = (32, 32)
    circle = Circle(x=16.0, y=16.0, r=10.0)
    circles = {1: circle, 2: circle, 3: circle, 4: circle}

    images_ref = {
        1: np.full(shape, 5.0),
        2: _gaussian(shape, circle, amplitude=10.0, sigma=4.0, background=5.0),
        3: np.full(shape, 5.0),
        4: np.full(shape, 5.0),
    }
    images_tgt = {
        1: np.full(shape, 8.0),
        2: _gaussian(shape, circle, amplitude=10.0, sigma=4.0, background=5.0),
        3: np.full(shape, 5.0),
        4: np.full(shape, 5.0),
    }

    reference_npz = tmp_path / 'mbo_align_ref_zero.npz'
    target_npz = tmp_path / 'mbo_align_target_zero.npz'
    _write_aligned_npz(reference_npz, circles, images_ref)
    _write_aligned_npz(target_npz, circles, images_tgt)

    config = MBOBackgroundSubtractionConfig(
        facility='synthetic-zero-division',
        reference_npz=str(reference_npz),
        target_npz=str(target_npz),
        reference_label='03-17',
        target_label='03-19',
        output_npz='synthetic_zero_bg.npz',
        smoothing_sigma_px=1.0,
    )

    result = explore_background_subtraction(config,
                                            output_dir=str(tmp_path),
                                            pdf_path=str(tmp_path /
                                                         'synthetic_zero.pdf'))

    saved = np.load(result['output_npz'])
    ratio = saved['scenario_b_foil_1_ratio']
    mask = np.isfinite(ratio)
    assert np.all(ratio[mask] == 0.0)


def test_background_subtraction_scenario_b_uses_soft_interior_renormalization(
        tmp_path):
    shape = (64, 64)
    circle = Circle(x=32.0, y=32.0, r=20.0)
    circles = {1: circle, 2: circle, 3: circle, 4: circle}

    # Reference has pronounced center; after stage-1 normalization, only the
    # interior should receive the full renormalization correction.
    images_ref = {
        1: _gaussian(shape, circle, amplitude=25.0, sigma=8.0, background=5.0),
        2: _gaussian(shape, circle, amplitude=20.0, sigma=9.0, background=5.0),
        3: np.full(shape, 5.0),
        4: np.full(shape, 5.0),
    }
    images_tgt = {
        1: _gaussian(shape, circle, amplitude=30.0, sigma=8.0, background=5.0),
        2: _gaussian(shape, circle, amplitude=24.0, sigma=9.0, background=5.0),
        3: np.full(shape, 5.0),
        4: np.full(shape, 5.0),
    }

    reference_npz = tmp_path / 'mbo_align_ref_soft.npz'
    target_npz = tmp_path / 'mbo_align_target_soft.npz'
    _write_aligned_npz(reference_npz, circles, images_ref)
    _write_aligned_npz(target_npz, circles, images_tgt)

    config = MBOBackgroundSubtractionConfig(
        facility='synthetic-soft-renorm',
        reference_npz=str(reference_npz),
        target_npz=str(target_npz),
        reference_label='03-17',
        target_label='03-19',
        output_npz='synthetic_soft_bg.npz',
        smoothing_sigma_px=2.0,
        renormalization_transition_sigma_px=2.0,
    )

    result = explore_background_subtraction(config,
                                            output_dir=str(tmp_path),
                                            pdf_path=str(tmp_path /
                                                         'synthetic_soft.pdf'))

    saved = np.load(result['output_npz'])
    weight = saved['scenario_b_foil_1_renormalization_weight']
    factor = saved['scenario_b_foil_1_renormalization_factor']

    cy = int(round(float(saved['scenario_b_foil_1_circle_y'])))
    cx = int(round(float(saved['scenario_b_foil_1_circle_x'])))

    center_weight = float(weight[cy, cx])
    edge_weight = float(weight[2, 2])
    center_factor = float(factor[cy, cx])
    edge_factor = float(factor[2, 2])

    assert 0.0 <= edge_weight < center_weight <= 1.0
    assert center_factor > edge_factor
