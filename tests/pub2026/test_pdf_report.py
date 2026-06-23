"""Tests for PDF report generator."""

import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.pub2026.pdf_report import PDFReport


class TestPDFReport:
    """Test PDFReport creation, pages, and save."""

    def test_creates_output_directory(self, tmp_path):
        path = tmp_path / "sub" / "report.pdf"
        report = PDFReport(str(path), title="Test")
        report.save()
        assert path.exists()

    def test_title_page_created(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Test Title")
        report.save()
        assert report._page_count >= 1

    def test_add_figure(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Test")
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])
        report.add_figure(fig,
                          caption="Test caption",
                          source_paths=["/data/test.csv"])
        report.save()
        assert report._page_count == 2  # title + 1 figure

    def test_add_table(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Test")
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        report.add_table(df, title="Test Table")
        report.save()
        assert report._page_count == 2

    def test_add_text(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Test")
        report.add_text("Hello world", title="Greeting")
        report.save()
        assert report._page_count == 2

    def test_add_metrics(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Test")
        metrics = {
            "CCB": {
                "FWHM": 5.2,
                "Range": 12.3
            },
            "AIC144": {
                "FWHM": 5.4,
                "Range": 12.5
            },
        }
        report.add_metrics(metrics, title="Metrics")
        report.save()
        assert report._page_count == 2

    def test_multiple_pages(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Multi")
        for i in range(5):
            fig, ax = plt.subplots()
            ax.plot([i], [i])
            report.add_figure(fig, caption=f"Figure {i}")
        report.save()
        assert report._page_count == 6  # title + 5 figures

    def test_save_returns_path(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Test")
        result = report.save()
        assert result == str(path)

    def test_config_path_in_report(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path),
                           title="Test",
                           config_path="/cfg/test.yaml")
        assert report.config_path == "/cfg/test.yaml"

    def test_timestamp_format(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Test")
        ts = report._format_timestamp()
        # Should be in YYYY-MM-DD HH:MM:SS format
        assert len(ts) == 19
        assert ts[4] == "-"
        assert ts[10] == " "

    def test_output_file_is_valid_pdf(self, tmp_path):
        path = tmp_path / "report.pdf"
        report = PDFReport(str(path), title="Test")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        report.add_figure(fig, caption="X")
        report.save()
        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"
