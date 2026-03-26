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
        1: _gaussian(shape, circle_ref, amplitude=20.0, sigma=6.0, background=5.0),
        2: _gaussian(shape, circle_ref, amplitude=16.0, sigma=7.0, background=5.0),
        3: np.full(shape, 5.0),
        4: np.full(shape, 5.0),
    }
    images_tgt = {
        1: _gaussian(shape, circle_tgt, amplitude=20.0, sigma=6.0, background=5.0),
        2: _gaussian(shape, circle_tgt, amplitude=16.0, sigma=7.0, background=5.0),
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
                                            pdf_path=str(tmp_path / 'synthetic.pdf'))

    assert Path(result['pdf_path']).exists()
    assert Path(result['output_npz']).exists()

    saved = np.load(result['output_npz'])
    assert np.isclose(saved['background_level'], 5.0)
    assert np.nanmax(np.abs(saved['scenario_a_foil_1_difference'])) < 1.0
    assert np.nanmax(np.abs(saved['scenario_a_foil_2_difference'])) < 1.0
