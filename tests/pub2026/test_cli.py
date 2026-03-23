"""Tests for CLI argument parsing and dispatch logic."""

import sys
from unittest.mock import patch, MagicMock

import pytest
import yaml

from src.pub2026.cli import main, _run_pipeline


class TestCLIParsing:

    def test_run_command_parses(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(yaml.dump({"type": "mc_depth_validation"}))
        with patch("src.pub2026.cli._run_pipeline") as mock:
            with patch("sys.argv", ["cli", "run", str(cfg)]):
                main()
            mock.assert_called_once()
            args = mock.call_args[0]
            assert str(cfg) in args[0]

    def test_run_all_command(self):
        with patch("src.pub2026.cli._run_all") as mock:
            with patch("sys.argv", ["cli", "run-all"]):
                main()
            mock.assert_called_once()

    def test_custom_output_dir(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(yaml.dump({"type": "mc_depth_validation"}))
        with patch("src.pub2026.cli._run_pipeline") as mock:
            with patch(
                    "sys.argv",
                ["cli", "run", str(cfg), "-o", "/custom/out"]):
                main()
            args = mock.call_args[0]
            assert args[1] == "/custom/out"

    def test_no_command_exits(self):
        with patch("sys.argv", ["cli"]):
            with pytest.raises(SystemExit):
                main()


class TestCLIDispatch:

    def test_unknown_type_exits(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(yaml.dump({"type": "nonexistent_type"}))
        with pytest.raises(SystemExit):
            _run_pipeline(str(cfg), str(tmp_path / "out"))

    def test_dispatch_calls_correct_pipeline(self, tmp_path):
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            yaml.dump({
                "type": "mc_depth_validation",
                "data_root": str(tmp_path),
            }))
        with patch(
                "src.pub2026.mc.depth_validation.validate_depth_dose") as mock:
            _run_pipeline(str(cfg), str(tmp_path / "out"))
            mock.assert_called_once()
