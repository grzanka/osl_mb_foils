"""Tests for comparisons module helper functions."""

import matplotlib

matplotlib.use('Agg')
import numpy as np
import pytest

from src.pub2026.profile_metrics import (
    all_metrics_dense as fac_all_metrics,
    find_fwhm_dense as fac_fwhm,
    find_distal_pct_dense as fac_distal,
    get_dense as fac_get_dense,
    falloff_80_20 as _penumbra_80_20,
    align_mbo_to_reference as _process_mbo,
    all_metrics_dense as sum_all_metrics,
    find_fwhm_dense as sum_fwhm,
    find_distal_pct_dense as sum_distal,
    falloff_80_20 as _falloff_80_20,
    align_mbo_to_reference as _align_mbo,
)
from scipy.interpolate import interp1d


class TestFacilityHelpers:

    def test_get_dense(self):
        x = np.array([0.0, 5.0, 10.0])
        y = np.array([0.0, 5.0, 10.0])
        xd, yd = fac_get_dense(x, y, n=100)
        assert len(xd) == 100
        assert len(yd) == 100
        assert abs(yd[50] - 5.0) < 0.5

    def test_fwhm_gaussian(self):
        x = np.linspace(-10, 10, 1000)
        sigma = 2.0
        y = np.exp(-0.5 * (x / sigma)**2)
        fw, xl, xr, hm = fac_fwhm(x, y)
        expected = 2 * sigma * np.sqrt(2 * np.log(2))
        assert fw is not None
        assert abs(fw - expected) < 0.2

    def test_fwhm_returns_none_for_flat(self):
        x = np.linspace(0, 10, 100)
        y = np.ones_like(x)
        fw, xl, xr, hm = fac_fwhm(x, y)
        # flat profile never crosses half max from below
        assert fw is None

    def test_distal_pct(self):
        x = np.linspace(0, 30, 1000)
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        pos = fac_distal(x, y, 0.9)
        assert pos is not None
        assert pos > 15

    def test_distal_pct_returns_none(self):
        x = np.linspace(0, 10, 50)
        y = np.ones_like(x)  # never drops below
        pos = fac_distal(x, y, 0.5)
        assert pos is None

    def test_penumbra_80_20(self):
        x = np.linspace(0, 30, 1000)
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        p = _penumbra_80_20(x, y)
        assert p is not None
        assert p > 0

    def test_all_metrics_keys(self):
        x = np.linspace(0, 30, 300)
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        m = fac_all_metrics(x, y)
        assert "fwhm" in m
        assert "x90" in m
        assert "penumbra_80_20" in m

    def test_process_mbo_alignment(self):
        # Simulated EBT with peak at x=5
        ebt_x = np.linspace(-5, 25, 100)
        ebt_y = np.exp(-0.5 * ((ebt_x - 5) / 3)**2) * 10
        ebt_interp = interp1d(ebt_x, ebt_y, fill_value='extrapolate')

        # MBO with peak at x=20
        mbo_x = np.linspace(0, 40, 100)
        mbo_y = np.exp(-0.5 * ((mbo_x - 20) / 3)**2) * 8

        xs, ys = _process_mbo(mbo_x,
                              mbo_y,
                              ref_max_x=5.0,
                              ref_interp=ebt_interp,
                              match_x=0.0)
        # After shifting, MBO peak should be near EBT peak (x=5)
        peak_pos = xs[np.argmax(ys)]
        assert abs(peak_pos - 5.0) < 1.0


class TestSummaryHelpers:

    def test_fwhm_gaussian(self):
        x = np.linspace(-10, 10, 1000)
        sigma = 2.5
        y = np.exp(-0.5 * (x / sigma)**2)
        fw, _, _, _ = sum_fwhm(x, y)
        expected = 2 * sigma * np.sqrt(2 * np.log(2))
        assert fw is not None
        assert abs(fw - expected) < 0.3

    def test_distal_pct(self):
        x = np.linspace(0, 30, 1000)
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        pos = sum_distal(x, y, 0.9)
        assert pos is not None
        assert pos > 15

    def test_falloff_80_20(self):
        x = np.linspace(0, 30, 1000)
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        f = _falloff_80_20(x, y)
        assert f is not None
        assert f > 0

    def test_align_mbo(self):
        ebt_x = np.linspace(-5, 25, 100)
        ebt_y = np.exp(-0.5 * ((ebt_x - 5) / 3)**2) * 10
        ebt_interp = interp1d(ebt_x, ebt_y, fill_value='extrapolate')

        mbo_x = np.linspace(0, 40, 100)
        mbo_y = np.exp(-0.5 * ((mbo_x - 20) / 3)**2) * 8

        xs, ys = _align_mbo(mbo_x,
                            mbo_y,
                            ref_max_x=5.0,
                            ref_interp=ebt_interp,
                            match_x=0.0)
        peak_pos = xs[np.argmax(ys)]
        assert abs(peak_pos - 5.0) < 1.0

    def test_all_metrics_keys(self):
        x = np.linspace(0, 30, 300)
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        m = sum_all_metrics(x, y)
        assert "fwhm" in m
        assert "x90" in m
        assert "falloff" in m
