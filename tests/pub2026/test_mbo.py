"""Tests for MBO module helper functions using synthetic data."""

import matplotlib

matplotlib.use('Agg')
import numpy as np
import pytest

from src.data.analysis import Circle
from src.pub2026.mbo.match import (
    filter_edge_points_near_circle,
    fit_circle_to_points,
    find_gradient_edge_points,
)
from src.pub2026.profile_metrics import (
    find_fwhm as _find_fwhm,
    find_distal_pct as _find_distal,
    calculate_profile_metrics as _calc_metrics,
    get_value_at_x as _get_value_at_x,
)


class TestMBOMatchHelpers:

    def test_find_gradient_edge_points_returns_list(self):
        rng = np.random.default_rng(42)
        img = rng.random((100, 100)).astype(np.float32)
        # Create a sharp horizontal edge
        img[50:, :] = 5.0
        points = find_gradient_edge_points(img,
                                           high_thresh=2.0,
                                           low_thresh=0.5)
        assert isinstance(points, list)
        # Should find edge points near row 50
        if len(points) > 0:
            ys = [p[0] for p in points]
            assert min(ys) >= 45
            assert max(ys) <= 55

    def test_filter_edge_points_near_circle(self):
        circle = Circle(x=50, y=50, r=30)
        # Points on the circle
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        on_circle = [(int(50 + 30 * np.sin(a)), int(50 + 30 * np.cos(a)))
                     for a in angles]
        # Points far away
        far = [(0, 0), (99, 99), (10, 10)]
        all_pts = on_circle + far
        filtered = filter_edge_points_near_circle(all_pts,
                                                  circle,
                                                  max_distance_px=5)
        assert len(
            filtered) >= len(on_circle) - 2  # most on-circle points kept
        assert len(filtered) <= len(on_circle) + 1  # far points excluded

    def test_fit_circle_to_points(self):
        # Generate noisy points on a circle in (x, y) format
        rng = np.random.default_rng(123)
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        true_cx, true_cy, true_r = 50.0, 60.0, 25.0
        pts = [(true_cx + true_r * np.cos(a) + rng.normal(0, 0.5),
                true_cy + true_r * np.sin(a) + rng.normal(0, 0.5))
               for a in angles]
        initial = Circle(x=48, y=58, r=24)
        result = fit_circle_to_points(pts, initial, radius_tolerance=0.2)
        assert isinstance(result, Circle)
        assert abs(result.x - true_cx) < 3
        assert abs(result.y - true_cy) < 3
        assert abs(result.r - true_r) < 3


class TestMBOComparisonHelpers:

    def test_get_value_at_x_interpolation(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        val = _get_value_at_x(x, y, 1.5)
        assert abs(val - 3.0) < 0.1

    def test_get_value_at_x_exact(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        val = _get_value_at_x(x, y, 1.0)
        assert abs(val - 20.0) < 1.0

    def test_find_fwhm_symmetric(self):
        x = np.linspace(-10, 10, 1000)
        sigma = 2.0
        y = np.exp(-0.5 * (x / sigma)**2)
        left, right, hm = _find_fwhm(y, x)
        assert left is not None
        assert right is not None
        fwhm = right - left
        expected = 2 * sigma * np.sqrt(2 * np.log(2))
        assert abs(fwhm - expected) < 0.1
        assert abs(hm - 0.5) < 0.01

    def test_find_fwhm_no_crossing(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 1.0, 1.0])  # flat, no crossing
        left, right, hm = _find_fwhm(y, x)
        # Expect None for boundaries if no crossing found
        assert left is None or right is None

    def test_find_distal(self):
        x = np.linspace(0, 30, 300)
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        pos, val = _find_distal(y, x, 90)
        assert pos is not None
        assert pos > 15  # distal is beyond the peak

    def test_find_distal_returns_none_when_not_found(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 1.0, 1.0])  # flat, never drops
        pos, val = _find_distal(y, x, 90)
        assert pos is None

    def test_calc_metrics_keys(self):
        x = np.linspace(-10, 30, 400)
        y = np.exp(-0.5 * ((x - 10) / 3)**2)
        m = _calc_metrics(y, x)
        assert "fwhm" in m
        assert "pos_90" in m
        assert "dist_80_20" in m
        assert "dist_90_10" in m

    def test_calc_metrics_values_reasonable(self):
        x = np.linspace(-10, 30, 400)
        sigma = 3.0
        y = np.exp(-0.5 * ((x - 10) / sigma)**2)
        m = _calc_metrics(y, x)
        expected_fwhm = 2 * sigma * np.sqrt(2 * np.log(2))
        assert abs(m["fwhm"] - expected_fwhm) < 0.5
