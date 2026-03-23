# pub2026 Analysis Pipeline

Automated analysis pipeline for the pub2026 publication, replacing 21 Jupyter notebooks with structured Python scripts that produce annotated PDF reports.

## Architecture

```
src/pub2026/
├── config.py          # YAML config loading + 10 typed dataclasses
├── pdf_report.py      # Multi-page PDF generator with timestamps
├── cli.py             # Command-line interface
├── mc/                # Monte Carlo simulation analysis
│   ├── depth_validation.py   # MC depth-dose vs experimental Bragg peak
│   ├── wedge_profile.py      # MC wedge dose/fluence/energy profiles
│   ├── comparison.py         # AIC-144 vs CCB MC profile comparison
│   └── config/               # YAML config files
├── ebt/               # EBT3 film dosimetry
│   ├── analysis.py           # Film pair processing, calibration, profiles
│   ├── comparison.py         # AIC-144 vs CCB EBT comparison
│   └── config/
├── mbo/               # MBO foil detector analysis
│   ├── explore.py            # Background subtraction, noise analysis
│   ├── match.py              # Two-foil edge matching and merging
│   ├── comparison.py         # AIC-144 vs CCB MBO comparison
│   └── config/
└── comparisons/       # Cross-modality comparisons
    ├── facility.py           # MC vs EBT3 vs MBO per facility
    ├── summary.py            # Full cross-facility, cross-modality summary
    └── config/
```

## Quick Start

```bash
# Install dependencies
poetry install

# Show available commands
poetry run python -m src.pub2026.cli --help

# Run a single analysis
poetry run python -m src.pub2026.cli run src/pub2026/mc/config/depth_validation_ccb.yaml

# Run all analyses in standard order
poetry run python -m src.pub2026.cli run-all

# Specify output directory
poetry run python -m src.pub2026.cli run-all --output-dir results/pub2026
```

## Configuration

All parameters are stored in YAML files under each module's `config/` directory. Every config must have a `type` field that identifies the pipeline to use.

### Config Types

| Type | Module | Description |
|------|--------|-------------|
| `mc_depth_validation` | mc/depth_validation.py | Bragg peak MC vs experiment |
| `mc_wedge` | mc/wedge_profile.py | Wedge dose/fluence/energy profiles |
| `mc_comparison` | mc/comparison.py | MC profile comparison across facilities |
| `ebt_analysis` | ebt/analysis.py | EBT3 film processing and calibration |
| `ebt_comparison` | ebt/comparison.py | EBT cross-facility comparison |
| `mbo_explore` | mbo/explore.py | MBO detector exploration |
| `mbo_match` | mbo/match.py | Two-foil matching and merging |
| `mbo_comparison` | mbo/comparison.py | MBO cross-facility comparison |
| `comparison_facility` | comparisons/facility.py | MC vs EBT vs MBO per facility |
| `comparison_summary` | comparisons/summary.py | Full cross-facility summary |

### Data Root

By default, data files are resolved relative to `data/raw/pub2026/` in the project root. Override with:

- `data_root` field in the YAML config, or
- `PUB2026_DATA_ROOT` environment variable

### Example Config

```yaml
type: mc_wedge
facility: ccb
file_format: csv
dose_file: mc_ccb/small/graph_x_profile_DOSE.csv
fluence_file: mc_ccb/small/graph_x_profile_FLUENCE.csv
energy_file: mc_ccb/small/graph_x_profile_ENERGY.csv
depth_filter_min_mm: -15.0
depth_filter_max_mm: 17.0
depth_shift_mm: 15.0
output_csv: mc_wedge_ccb_small.csv
```

## PDF Reports

Every pipeline generates an annotated PDF report including:

- **Title page** with report name, generation timestamp, output path
- **Figures** with captions and source data annotations
- **Tables** with formatted metrics
- **Text pages** with analysis summaries
- **Footer** on every page: timestamp, config path, source data paths, page number

## Running Tests

```bash
# Unit tests (no real data needed)
python -m pytest tests/pub2026/ -v

# Integration tests (require data files)
python -m pytest tests/pub2026/ -v -m integration
```

## Pipeline Execution Order

When using `run-all`, configs are executed in this order:

1. **MC** — depth validation, wedge profiles (CCB small, CCB big, AIC-144 big), comparison
2. **EBT** — film analysis (AIC-144, CCB), comparison
3. **MBO** — exploration (AIC-144, CCB), matching (AIC-144, CCB), comparison
4. **Comparisons** — per-facility (AIC-144, CCB), summary

## Notebook Mapping

| Original Notebook | Pipeline Script | Config(s) |
|---|---|---|
| mc/0.1-ccb-depth-validation | mc/depth_validation.py | depth_validation_ccb.yaml |
| mc/0.2-ccb-wedge-small | mc/wedge_profile.py | wedge_ccb_small.yaml |
| mc/0.3-ccb-wedge-big | mc/wedge_profile.py | wedge_ccb_big.yaml |
| mc/0.3-aic144-wedge-big | mc/wedge_profile.py | wedge_aic144_big.yaml |
| mc/0.4-comparison | mc/comparison.py | comparison.yaml |
| ebt_foils/1.0-aic144 | ebt/analysis.py | analysis_aic144.yaml |
| ebt_foils/1.0-ccb | ebt/analysis.py | analysis_ccb.yaml |
| ebt_foils/2.0-comparison | ebt/comparison.py | comparison.yaml |
| mbo_foils/0.1-explore-aic144 | mbo/explore.py | explore_aic144.yaml |
| mbo_foils/0.1-explore-ccb | mbo/explore.py | explore_ccb.yaml |
| mbo_foils/0.2-match-aic144 | mbo/match.py | match_aic144.yaml |
| mbo_foils/0.2-match-ccb | mbo/match.py | match_ccb.yaml |
| mbo_foils/0.3-comparison | mbo/comparison.py | comparison.yaml |
| comparisons/1.0-aic144 | comparisons/facility.py | facility_aic144.yaml |
| comparisons/1.0-ccb | comparisons/facility.py | facility_ccb.yaml |
| comparisons/2.0-comparisons | comparisons/summary.py | summary.yaml |
