"""YAML configuration loading and typed dataclasses for pub2026 pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from src.data.paths import project_dir

DEFAULT_DATA_ROOT = project_dir / "data" / "raw" / "pub2026"


def _resolve_data_root() -> Path:
    env_root = os.environ.get("PUB2026_DATA_ROOT")
    if env_root:
        return Path(env_root)
    return DEFAULT_DATA_ROOT


def resolve_file(path: str, output_dir: Path) -> Path:
    """Resolve a file path, falling back to output_dir if the resolved path doesn't exist.

    This handles intermediate outputs (e.g. CSVs/NPZs produced by upstream pipelines)
    that live in the output directory rather than the data root.
    Also searches sibling module data directories for the structured output layout.
    """
    p = Path(path)
    if p.exists():
        return p
    # Direct fallback: output_dir / filename
    fallback = output_dir / p.name
    if fallback.exists():
        return fallback
    # Search sibling module data directories (for structured output layout)
    # e.g. output_dir = .../comparisons/data, look in .../mc/data, .../ebt/data etc.
    if output_dir.name in ('data', 'reports'):
        base = output_dir.parent.parent
        for module in ('mc', 'ebt', 'mbo', 'comparisons'):
            candidate = base / module / 'data' / p.name
            if candidate.exists():
                return candidate
    return p  # Return original for better error messages


@dataclass
class MCDepthValidationConfig:
    """Config for MC depth-dose validation (notebook 0.1)."""
    mc_csv_path: str = "mc_ccb/small/graph_z_profile_DOSE.csv"
    threshold: float = 0.1
    output_h5: str = "ccb.h5"
    data_root: str = ""

    def resolve_paths(self) -> "MCDepthValidationConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        self.mc_csv_path = str(root / self.mc_csv_path)
        return self


@dataclass
class MCWedgeConfig:
    """Config for MC wedge profile processing (notebooks 0.2, 0.3)."""
    facility: str = "ccb"
    file_format: str = "csv"  # "csv" for small, "dat" for big
    dose_file: str = ""
    fluence_file: str = ""
    energy_file: str = ""
    depth_filter_min_mm: float = -15.0
    depth_filter_max_mm: float = 17.0
    depth_shift_mm: float = 15.0
    output_csv: str = ""
    data_root: str = ""

    def resolve_paths(self) -> "MCWedgeConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        self.dose_file = str(root / self.dose_file)
        self.fluence_file = str(root / self.fluence_file)
        self.energy_file = str(root / self.energy_file)
        return self


@dataclass
class MCComparisonConfig:
    """Config for MC profile comparison (notebook 0.4)."""
    aic144_csv: str = ""
    ccb_csv: str = ""
    reference_dose_gy: float = 5.0
    data_root: str = ""

    def resolve_paths(self) -> "MCComparisonConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        if self.aic144_csv and not Path(self.aic144_csv).is_absolute():
            self.aic144_csv = str(root / self.aic144_csv)
        if self.ccb_csv and not Path(self.ccb_csv).is_absolute():
            self.ccb_csv = str(root / self.ccb_csv)
        return self


@dataclass
class EBTAnalysisConfig:
    """Config for EBT film analysis (notebooks 1.0)."""
    facility: str = "aic144"
    file_pairs: List[dict] = field(default_factory=list)
    crop_dose_threshold_gy: float = 0.5
    crop_smooth_sigma_mm: float = 0.1
    crop_min_diameter_mm: float = 5.0
    crop_margin_mm: float = 9.0
    cmap_white_threshold_percent: float = 1.0
    pencil_threshold_gy: float = 13.0
    pencil_mark1_y_range_mm: List[float] = field(
        default_factory=lambda: [40.0, 55.0])
    pencil_mark1_x_range_mm: List[float] = field(
        default_factory=lambda: [12.0, 18.0])
    pencil_mark2_y_range_mm: List[float] = field(
        default_factory=lambda: [0.0, 10.0])
    pencil_mark2_x_range_mm: List[float] = field(
        default_factory=lambda: [12.0, 18.0])
    line_fit_dose_threshold_gy: float = 1.0
    circle_radius_mm: float = 10.0
    profile_half_width_px: int = 10
    dpi: float = 150.0
    rotation_k: int = 1
    output_npz: str = ""
    output_pair_key: str = ""
    data_root: str = ""

    def resolve_paths(self) -> "EBTAnalysisConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        for pair in self.file_pairs:
            for key in ("signal", "background"):
                if key in pair and not Path(pair[key]).is_absolute():
                    pair[key] = str(root / pair[key])
        return self


@dataclass
class EBTComparisonConfig:
    """Config for EBT comparison (notebook 2.0)."""
    aic144_npz: str = ""
    ccb_npz: str = ""
    pencil_mark_x_aic144_mm: float = 15.46
    pencil_mark_x_ccb_mm: float = 12.14
    strip_width_mm: float = 5.0
    dpi: float = 150.0
    output_csv_aic144: str = "ebt_profile_aic144.csv"
    output_csv_ccb: str = "ebt_profile_ccb.csv"
    data_root: str = ""

    def resolve_paths(self) -> "EBTComparisonConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        if self.aic144_npz and not Path(self.aic144_npz).is_absolute():
            self.aic144_npz = str(root / self.aic144_npz)
        if self.ccb_npz and not Path(self.ccb_npz).is_absolute():
            self.ccb_npz = str(root / self.ccb_npz)
        return self


@dataclass
class MBOExploreConfig:
    """Config for MBO foil exploration (notebooks 0.1-explore)."""
    facility: str = "aic144"
    data_dir: str = ""
    cut_px: int = 300
    gaussian_kernel: int = 3
    median_filter_size: int = 5
    n_sigma: float = 7.0
    probe_radius: int = 30
    pixel_size_mm: float = 0.074
    output_pkl: str = ""
    output_npz: str = ""
    data_root: str = ""

    def resolve_paths(self) -> "MBOExploreConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        if self.data_dir and not Path(self.data_dir).is_absolute():
            self.data_dir = str(root / self.data_dir)
        return self


@dataclass
class MBOMatchConfig:
    """Config for MBO foil matching (notebooks 0.2-match)."""
    facility: str = "aic144"
    data_dir: str = ""
    mode: str = "merge"  # "merge" or "single_read"
    input_pkl: str = ""
    left_foil_id: int = 1
    right_foil_id: int = 2
    edge_threshold: float = 400.0
    edge_x_positions_mm: List[float] = field(
        default_factory=lambda: [10.4, 11.8, 13.3, 14.8, 16.3, 17.8, 19.2])
    edge_stripe_width_mm: float = 0.74
    crop_size: int = 300
    merge_margin: int = 0
    pixel_size_mm: float = 0.074
    vmax: float = 3000.0
    white_threshold: float = 0.05
    circle_gradient_high: float = 400.0
    circle_gradient_low: float = 50.0
    circle_edge_distance_mm: float = 0.7
    circle_radius_tolerance: float = 0.05
    profile_y_min_mm: float = 10.0
    profile_y_max_mm: float = 16.0
    section1_x_min: float = 7.0
    section1_x_max: float = 17.0
    section2_x_min: float = 21.2
    section2_x_max: float = 27.0
    smoothing_sigma_px: int = 3
    output_merged_npz: str = ""
    output_profile_npz: str = ""
    data_root: str = ""

    def resolve_paths(self) -> "MBOMatchConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        if self.data_dir and not Path(self.data_dir).is_absolute():
            self.data_dir = str(root / self.data_dir)
        if self.input_pkl and not Path(self.input_pkl).is_absolute():
            self.input_pkl = str(root / self.input_pkl)
        return self


@dataclass
class MBOComparisonConfig:
    """Config for MBO profile comparison (notebooks 0.3-comparison, 0.4)."""
    aic144_profile_path: str = ""
    ccb_profile_path: str = ""
    reference_dose_gy: float = 5.0
    reference_x_mm: float = 5.0
    output_csv_aic144: str = "mbo_profile_aic144.csv"
    output_csv_ccb: str = "mbo_profile_ccb.csv"
    output_metrics_csv: str = "mbo_profile_metrics.csv"
    data_root: str = ""

    def resolve_paths(self) -> "MBOComparisonConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        if self.aic144_profile_path and not Path(
                self.aic144_profile_path).is_absolute():
            self.aic144_profile_path = str(root / self.aic144_profile_path)
        if self.ccb_profile_path and not Path(
                self.ccb_profile_path).is_absolute():
            self.ccb_profile_path = str(root / self.ccb_profile_path)
        return self


@dataclass
class ComparisonFacilityConfig:
    """Config for cross-modality comparison per facility (notebooks 1.0-comparisons)."""
    facility: str = "aic144"
    mc_csv: str = ""
    ebt_csv: str = ""
    mbo_npz: str = ""
    mbo_single_foil_npz: Optional[str] = None
    normalize_at_x: float = 0.0
    mbo_gap_exclude_min_mm: Optional[float] = None
    mbo_gap_exclude_max_mm: Optional[float] = None
    data_root: str = ""

    def resolve_paths(self) -> "ComparisonFacilityConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        for attr in ("mc_csv", "ebt_csv", "mbo_npz"):
            val = getattr(self, attr)
            if val and not Path(val).is_absolute():
                setattr(self, attr, str(root / val))
        if self.mbo_single_foil_npz and not Path(
                self.mbo_single_foil_npz).is_absolute():
            self.mbo_single_foil_npz = str(root / self.mbo_single_foil_npz)
        return self


@dataclass
class ComparisonSummaryConfig:
    """Config for cross-facility summary comparison (notebook 2.0-comparisons)."""
    mc_aic144_csv: str = ""
    mc_ccb_csv: str = ""
    ebt_aic144_csv: str = ""
    ebt_ccb_csv: str = ""
    mbo_aic144_npz: str = ""
    mbo_ccb_npz: str = ""
    reference_dose_gy: float = 5.0
    data_root: str = ""

    def resolve_paths(self) -> "ComparisonSummaryConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        for attr in ("mc_aic144_csv", "mc_ccb_csv", "ebt_aic144_csv",
                     "ebt_ccb_csv", "mbo_aic144_npz", "mbo_ccb_npz"):
            val = getattr(self, attr)
            if val and not Path(val).is_absolute():
                setattr(self, attr, str(root / val))
        return self


@dataclass
class MBORawSurveyConfig:
    """Config for raw MBO foil survey — no background subtraction."""
    facility: str = "ccb"
    data_dir: str = ""
    cut_px: int = 200
    pixel_size_mm: float = 0.074
    smooth_method: str = "none"  # "none", "gaussian", or "median"
    smooth_kernel: int = 5
    clip_percentile_low: float = 1.0
    clip_percentile_high: float = 99.0
    contour_levels: List[float] = field(
        default_factory=lambda: [400, 600, 800, 1000, 1200, 1400, 1600])
    output_npz: str = ""
    data_root: str = ""

    def resolve_paths(self) -> "MBORawSurveyConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        if self.data_dir and not Path(self.data_dir).is_absolute():
            self.data_dir = str(root / self.data_dir)
        return self


@dataclass
class MBOAlignConfig:
    """Config for MBO foil alignment: circle detection, edge finding, rotation."""
    facility: str = "ccb"
    input_npz: str = ""
    pixel_size_mm: float = 0.074
    n_sigma: float = 7.0
    probe_radius: int = 30
    edge_threshold: float = 400.0
    edge_x_positions_mm: List[float] = field(
        default_factory=lambda: [10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    edge_stripe_width_mm: float = 0.74
    circle_gradient_high: float = 400.0
    circle_gradient_low: float = 50.0
    circle_edge_distance_mm: float = 0.7
    circle_radius_tolerance: float = 0.05
    crop_size_mm: float = 24.0
    clip_percentile_low: float = 1.0
    clip_percentile_high: float = 99.0
    contour_levels: List[float] = field(
        default_factory=lambda: [600, 800, 1000, 1200, 1400, 1600])
    output_npz: str = ""
    data_root: str = ""

    def resolve_paths(self) -> "MBOAlignConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        if self.input_npz and not Path(self.input_npz).is_absolute():
            self.input_npz = str(root / self.input_npz)
        return self


@dataclass
class MBOBackgroundSubtractionConfig:
    """Config for post-alignment MBO background-subtraction scenarios."""
    facility: str = "ccb_2026-03-17_2026-03-19"
    reference_npz: str = ""
    target_npz: str = ""
    reference_label: str = "03-17"
    target_label: str = "03-19"
    pixel_size_mm: float = 0.074
    reference_foil_ids: List[int] = field(default_factory=lambda: [1, 2])
    background_foil_ids: List[int] = field(default_factory=lambda: [3, 4])
    background_radius_fraction: float = 0.8
    normalization_radius_fraction: float = 0.9
    smoothing_sigma_px: float = 6.0
    profile_strip_half_width_px: int = 5
    minimum_divisor: float = 0.5
    ratio_mask_radius_fraction: float = 1.0
    target_bg_y_threshold_mm: float = 7.0
    scenario_a_contour_levels: List[float] = field(
        default_factory=lambda: [-300.0, -200.0, -100.0, -50.0, -25.0,
                                 25.0, 50.0, 100.0, 200.0, 300.0])
    normalized_contour_levels: List[float] = field(
        default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    ratio_contour_levels: List[float] = field(
        default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    output_npz: str = ""
    data_root: str = ""

    def resolve_paths(self) -> "MBOBackgroundSubtractionConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        for attr in ("reference_npz", "target_npz"):
            val = getattr(self, attr)
            if val and not Path(val).is_absolute():
                setattr(self, attr, str(root / val))
        return self


@dataclass
class MBOBackgroundSubtractionMergeConfig:
    """Config for merging foils after background subtraction (scenarios A & B)."""
    facility: str = "ccb_2026-03-17_2026-03-19"
    bg_subtraction_npz: str = ""
    reference_npz: str = ""
    target_npz: str = ""
    pixel_size_mm: float = 0.074
    foil_ids: List[int] = field(default_factory=lambda: [1, 2])
    edge_threshold: float = 400.0
    edge_x_positions_mm: List[float] = field(
        default_factory=lambda: [10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    edge_stripe_width_mm: float = 0.74
    crop_size: int = 300
    merge_margin: int = 0
    vmax_scenario_a: float = 300.0
    vmax_scenario_b: float = 2.0
    white_threshold: float = 0.05
    circle_gradient_high: float = 400.0
    circle_gradient_low: float = 50.0
    circle_edge_distance_mm: float = 0.7
    circle_radius_tolerance: float = 0.05
    profile_y_offsets_mm: List[float] = field(
        default_factory=lambda: [-4.0, -2.0, 0.0, 2.0, 4.0])
    profile_strip_width_mm: float = 1.0
    profile_y_min_mm: float = 10.0
    profile_y_max_mm: float = 16.0
    section1_x_min: float = 3.0
    section1_x_max: float = 15.0
    section2_x_min: float = 25.0
    section2_x_max: float = 37.0
    smoothing_sigma_px: int = 3
    scenario_a_contour_levels: List[float] = field(
        default_factory=lambda: [25.0, 50.0, 100.0, 200.0, 300.0])
    scenario_b_contour_levels: List[float] = field(
        default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    output_height_mm: float = 25.0
    output_width_mm: float = 40.0
    output_npz: str = ""
    output_profile_npz: str = ""
    data_root: str = ""

    def resolve_paths(self) -> "MBOBackgroundSubtractionMergeConfig":
        root = Path(self.data_root) if self.data_root else _resolve_data_root()
        for attr in ("bg_subtraction_npz", "reference_npz", "target_npz"):
            val = getattr(self, attr)
            if val and not Path(val).is_absolute():
                setattr(self, attr, str(root / val))
        return self


# Map of config type names to classes
CONFIG_CLASSES = {
    "mc_depth_validation": MCDepthValidationConfig,
    "mc_wedge": MCWedgeConfig,
    "mc_comparison": MCComparisonConfig,
    "ebt_analysis": EBTAnalysisConfig,
    "ebt_comparison": EBTComparisonConfig,
    "mbo_explore": MBOExploreConfig,
    "mbo_match": MBOMatchConfig,
    "mbo_comparison": MBOComparisonConfig,
    "mbo_raw_survey": MBORawSurveyConfig,
    "mbo_align": MBOAlignConfig,
    "mbo_background_subtraction": MBOBackgroundSubtractionConfig,
    "mbo_background_subtraction_merge": MBOBackgroundSubtractionMergeConfig,
    "comparison_facility": ComparisonFacilityConfig,
    "comparison_summary": ComparisonSummaryConfig,
}


def load_config(yaml_path: str, config_type: str = None):
    """Load a YAML config file and return a typed dataclass.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.
    config_type : str, optional
        Key into CONFIG_CLASSES. If None, inferred from YAML 'type' field.

    Returns
    -------
    dataclass instance with resolved paths.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    if config_type is None:
        config_type = raw.pop("type", None)
    else:
        raw.pop("type", None)

    if config_type is None:
        raise ValueError(
            f"No config type specified in {yaml_path} and none provided as argument"
        )

    cls = CONFIG_CLASSES.get(config_type)
    if cls is None:
        raise ValueError(
            f"Unknown config type '{config_type}'. Available: {list(CONFIG_CLASSES.keys())}"
        )

    # Filter to only fields the dataclass knows about
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    config = cls(**filtered)
    config.resolve_paths()
    return config
