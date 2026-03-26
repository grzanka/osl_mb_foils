"""Command-line interface for pub2026 analysis pipelines.

Usage:
    python -m src.pub2026.cli run <config.yaml> [--output-dir DIR]
    python -m src.pub2026.cli run-all [--output-dir DIR]

Output directory structure::

    <output-dir>/
        mc/
            data/       ← CSVs, H5 files
            reports/    ← PDF reports
        ebt/
            data/
            reports/
        mbo/
            data/
            reports/
        comparisons/
            data/
            reports/
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

from src.pub2026.config import load_config, CONFIG_CLASSES

# Map config types to (module_subdir, data_or_reports_are_mixed)
_MODULE_MAP = {
    'mc_depth_validation': 'mc',
    'mc_wedge': 'mc',
    'mc_comparison': 'mc',
    'ebt_analysis': 'ebt',
    'ebt_comparison': 'ebt',
    'mbo_explore': 'mbo',
    'mbo_match': 'mbo',
    'mbo_comparison': 'mbo',
    'mbo_raw_survey': 'mbo',
    'mbo_align': 'mbo',
    'comparison_facility': 'comparisons',
    'comparison_summary': 'comparisons',
}


def _resolve_output_dirs(base_output_dir: str, config_type: str):
    """Return (data_dir, reports_dir) for a given config type."""
    module = _MODULE_MAP.get(config_type, '')
    base = Path(base_output_dir)
    data_dir = base / module / 'data'
    reports_dir = base / module / 'reports'
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir), str(reports_dir)


# Module-level flags set by CLI arguments
_TIMING_ENABLED = False
_PARALLEL_ENABLED = False


def _run_pipeline(config_path: str, output_dir: str):
    """Load a YAML config, resolve data paths, and dispatch to the correct pipeline."""
    t_start = time.perf_counter()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config_type = raw.get('type', '')
    data_dir, reports_dir = _resolve_output_dirs(output_dir, config_type)

    if config_type == 'mc_depth_validation':
        cfg = load_config(config_path, 'mc_depth_validation')
        from src.pub2026.mc.depth_validation import validate_depth_dose
        validate_depth_dose(
            cfg,
            output_dir=data_dir,
            pdf_path=str(Path(reports_dir) / "1_mc_depth_validation.pdf"))

    elif config_type == 'mc_wedge':
        cfg = load_config(config_path, 'mc_wedge')
        from src.pub2026.mc.wedge_profile import process_wedge_profile
        process_wedge_profile(
            cfg,
            output_dir=data_dir,
            pdf_path=str(Path(reports_dir) / f"2_mc_wedge_{cfg.facility}.pdf"))

    elif config_type == 'mc_comparison':
        cfg = load_config(config_path, 'mc_comparison')
        from src.pub2026.mc.comparison import compare_mc_profiles
        compare_mc_profiles(cfg,
                            output_dir=data_dir,
                            pdf_path=str(
                                Path(reports_dir) / "3_mc_comparison.pdf"))

    elif config_type == 'ebt_analysis':
        cfg = load_config(config_path, 'ebt_analysis')
        from src.pub2026.ebt.analysis import analyze_ebt
        analyze_ebt(cfg,
                    output_dir=data_dir,
                    pdf_path=str(
                        Path(reports_dir) /
                        f"1_ebt_analysis_{cfg.facility}.pdf"))

    elif config_type == 'ebt_comparison':
        cfg = load_config(config_path, 'ebt_comparison')
        from src.pub2026.ebt.comparison import compare_ebt
        compare_ebt(cfg,
                    output_dir=data_dir,
                    pdf_path=str(Path(reports_dir) / "2_ebt_comparison.pdf"))

    elif config_type == 'mbo_explore':
        cfg = load_config(config_path, 'mbo_explore')
        from src.pub2026.mbo.explore import explore_mbo
        explore_mbo(cfg,
                    output_dir=data_dir,
                    pdf_path=str(
                        Path(reports_dir) /
                        f"1_mbo_explore_{cfg.facility}.pdf"))

    elif config_type == 'mbo_match':
        cfg = load_config(config_path, 'mbo_match')
        from src.pub2026.mbo.match import match_mbo
        match_mbo(
            cfg,
            output_dir=data_dir,
            pdf_path=str(
                Path(reports_dir) /
                f"2_mbo_match_{cfg.facility}_{cfg.left_foil_id}_{cfg.right_foil_id}.pdf"
            ))

    elif config_type == 'mbo_comparison':
        cfg = load_config(config_path, 'mbo_comparison')
        from src.pub2026.mbo.comparison import compare_mbo
        compare_mbo(cfg,
                    output_dir=data_dir,
                    pdf_path=str(Path(reports_dir) / "3_mbo_comparison.pdf"))

    elif config_type == 'mbo_raw_survey':
        cfg = load_config(config_path, 'mbo_raw_survey')
        from src.pub2026.mbo.survey_raw import survey_raw_mbo
        survey_raw_mbo(cfg,
                       output_dir=data_dir,
                       pdf_path=str(
                           Path(reports_dir) /
                           f"4_mbo_raw_survey_{cfg.facility}.pdf"))

    elif config_type == 'mbo_align':
        cfg = load_config(config_path, 'mbo_align')
        from src.pub2026.mbo.align import align_mbo
        align_mbo(cfg,
                  output_dir=data_dir,
                  pdf_path=str(
                      Path(reports_dir) / f"5_mbo_align_{cfg.facility}.pdf"),
                  timing=_TIMING_ENABLED,
                  parallel=_PARALLEL_ENABLED)

    elif config_type == 'comparison_facility':
        cfg = load_config(config_path, 'comparison_facility')
        from src.pub2026.comparisons.facility import compare_facility
        compare_facility(cfg,
                         output_dir=data_dir,
                         pdf_path=str(
                             Path(reports_dir) /
                             f"1_comparison_{cfg.facility}.pdf"))

    elif config_type == 'comparison_summary':
        cfg = load_config(config_path, 'comparison_summary')
        from src.pub2026.comparisons.summary import compare_summary
        compare_summary(cfg,
                        output_dir=data_dir,
                        pdf_path=str(
                            Path(reports_dir) / "2_comparison_summary.pdf"))
    else:
        print(f"Unknown config type: {config_type!r}")
        print(f"Supported types: {', '.join(CONFIG_CLASSES.keys())}")
        sys.exit(1)

    if _TIMING_ENABLED:
        elapsed = time.perf_counter() - t_start
        print(f"\n[TIMING] Total pipeline time: {elapsed:.2f}s")


def _run_all(output_dir: str):
    """Execute all config files in standard order."""
    config_dirs = [
        Path(__file__).parent / 'mc' / 'config',
        Path(__file__).parent / 'ebt' / 'config',
        Path(__file__).parent / 'mbo' / 'config',
        Path(__file__).parent / 'comparisons' / 'config',
    ]

    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        for yaml_file in sorted(config_dir.glob('*.yaml')):
            print(f"\n{'='*60}")
            print(f"Running: {yaml_file.name}")
            print(f"{'='*60}")
            try:
                _run_pipeline(str(yaml_file), output_dir)
            except Exception as e:
                print(f"ERROR in {yaml_file.name}: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='pub2026 analysis pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # run single config
    run_p = sub.add_parser('run', help='Run a single config file')
    run_p.add_argument('config', help='Path to YAML config file')
    run_p.add_argument('--output-dir',
                       '-o',
                       default='output/pub2026',
                       help='Output directory (default: output/pub2026)')
    run_p.add_argument('--timing',
                       '-t',
                       action='store_true',
                       help='Print per-step timing information')
    run_p.add_argument('--parallel',
                       '-j',
                       action='store_true',
                       help='Process foils in parallel (multiprocessing)')

    # run all
    all_p = sub.add_parser('run-all', help='Run all configs in standard order')
    all_p.add_argument('--output-dir',
                       '-o',
                       default='output/pub2026',
                       help='Output directory (default: output/pub2026)')
    all_p.add_argument('--timing',
                       '-t',
                       action='store_true',
                       help='Print per-step timing information')
    all_p.add_argument('--parallel',
                       '-j',
                       action='store_true',
                       help='Process foils in parallel (multiprocessing)')

    args = parser.parse_args()

    global _TIMING_ENABLED, _PARALLEL_ENABLED
    _TIMING_ENABLED = getattr(args, 'timing', False)
    _PARALLEL_ENABLED = getattr(args, 'parallel', False)

    if args.command == 'run':
        _run_pipeline(args.config, args.output_dir)
    elif args.command == 'run-all':
        _run_all(args.output_dir)


if __name__ == '__main__':
    main()
