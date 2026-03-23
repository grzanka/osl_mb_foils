"""Tests for MC analysis modules using synthetic data."""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest

from src.pub2026.mc.depth_validation import (
    calculate_ratio,
    load_experimental_data,
    load_mc_depth_dose,
)
from src.pub2026.mc.wedge_profile import (
    filter_reverse_shift,
    find_distal_percent_position,
    find_fwhm_boundaries,
    load_csv_files,
)
from src.pub2026.mc.comparison import calculate_profile_metrics, get_dose_at_x0


class TestDepthValidation:

    def test_load_experimental_data(self):
        df = load_experimental_data()
        assert isinstance(df, pd.DataFrame)
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) > 0

    def test_load_mc_depth_dose(self, tmp_path):
        csv = tmp_path / "mc.csv"
        csv.write_text(
            "z,dose,error\n0.0,1.0,0.01\n1.0,2.0,0.02\n2.0,1.5,0.01\n")
        df = load_mc_depth_dose(str(csv))
        assert len(df) == 3

    def test_calculate_ratio(self):
        df_exp = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 1.5]})
        df_mc = pd.DataFrame({
            "depth": [0.0, 1.0, 2.0],
            "dose": [0.9, 1.8, 1.4]
        })
        ratio, stats = calculate_ratio(df_exp, df_mc, threshold=0.05)
        assert len(ratio) > 0
        assert "mean" in stats


class TestWedgeProfile:

    def test_filter_reverse_shift(self):
        df = pd.DataFrame({
            "depth": [-20.0, -10.0, 0.0, 10.0, 20.0],
            "dose": [1, 2, 3, 4, 5],
            "fluence": [1, 2, 3, 4, 5],
            "energy": [1, 2, 3, 4, 5]
        })
        result = filter_reverse_shift(df,
                                      min_mm=-15.0,
                                      max_mm=17.0,
                                      shift_mm=15.0)
        assert "depth" in result.columns
        # Original reversed + shifted, check within expected range
        assert result["depth"].min() >= 0
        assert result["depth"].max() <= 32

    def test_find_fwhm_boundaries(self):
        x = np.linspace(0, 30, 300)
        # Gaussian-like profile
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        left, right, hm = find_fwhm_boundaries(x, y)
        assert left is not None
        assert right is not None
        assert 5 < (right - left) < 9  # ~7.06 for sigma=3
        assert abs(hm - 0.5) < 0.01

    def test_find_distal_percent_position(self):
        x = np.linspace(0, 30, 300)
        y = np.exp(-0.5 * ((x - 15) / 3)**2)
        pos, val = find_distal_percent_position(x, y, 90.0)
        assert pos is not None
        assert pos > 15  # distal side

    def test_load_csv_files(self, tmp_path):
        for name in ("dose.csv", "fluence.csv", "energy.csv"):
            (tmp_path / name).write_text("depth,value\n0.0,1.0\n5.0,2.0\n")
        df = load_csv_files(
            str(tmp_path / "dose.csv"),
            str(tmp_path / "fluence.csv"),
            str(tmp_path / "energy.csv"),
        )
        assert "depth" in df.columns
        assert "dose" in df.columns


class TestMCComparison:

    def test_get_dose_at_x0(self):
        df = pd.DataFrame({"depth": [-1.0, 0.0, 1.0], "dose": [0.5, 1.0, 0.8]})
        val = get_dose_at_x0(df)
        assert abs(val - 1.0) < 0.01

    def test_calculate_profile_metrics(self):
        x = np.linspace(-10, 30, 400)
        # Bragg-peak-like profile
        profile = np.exp(-0.5 * ((x - 10) / 4)**2)
        metrics = calculate_profile_metrics(profile, x)
        assert "fwhm" in metrics or "FWHM" in metrics or "fwhm_mm" in str(
            metrics).lower()
