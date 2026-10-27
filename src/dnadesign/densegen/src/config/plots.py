"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/plots.py

DenseGen plotting configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal

from ..viz.plot_registry import PLOT_SPECS

CURRENT_PUBLIC_PLOT_IDS = frozenset(PLOT_SPECS.keys())
LEGACY_PUBLIC_PLOT_IDS = frozenset(
    {
        "accepted_arrays_by_plan",
        "dataset_metadata_heatmap",
        "dataset_source_inventory",
        "dense_array_video_showcase",
        "plan_by_regulator_heatmap",
        "placement_map",
        "retained_vs_deployed_length_shift",
        "retained_vs_deployed_tier_mix",
        "run_health",
        "run_health/compression_ratio_distribution",
        "run_health/outcomes_over_time",
        "run_health/run_health",
        "run_health/summary_table.pdf",
        "run_health/tfbs_length_by_regulator",
        "stage_a_summary",
        "stage_a_summary/background_logo",
        "stage_a_summary/diversity",
        "stage_a_summary/pool_tiers",
        "stage_a_summary/sampling_vs_length_ridgeline",
        "stage_a_summary/yield_bias",
        "tfbs_usage",
        "upstream_evidence_quality_summary",
        "used_unique_vs_retained",
    }
)
SUPPORTED_PLOT_OPTION_BLOCKS = frozenset({"placement_occupancy_map", "tfbs_concentration_profile"})


class PlotVideoSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stride: int = 5
    max_source_rows: int = 20_000
    max_snapshots: int = 140
    plan_snapshots: Dict[str, int] = Field(default_factory=dict)

    @field_validator("stride")
    @classmethod
    def _stride_ok(cls, value: int) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("plots.video.sampling.stride must be >= 1")
        return int(value)

    @field_validator("max_source_rows")
    @classmethod
    def _max_source_rows_ok(cls, value: int) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("plots.video.sampling.max_source_rows must be >= 1")
        return int(value)

    @field_validator("max_snapshots")
    @classmethod
    def _max_snapshots_ok(cls, value: int) -> int:
        if not isinstance(value, int) or value < 1:
            raise ValueError("plots.video.sampling.max_snapshots must be >= 1")
        return int(value)

    @field_validator("plan_snapshots")
    @classmethod
    def _plan_snapshots_ok(cls, value: Dict[str, int]) -> Dict[str, int]:
        normalized: dict[str, int] = {}
        for raw_name, raw_count in dict(value or {}).items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("plots.video.sampling.plan_snapshots keys must be non-empty plan names")
            if not isinstance(raw_count, int) or raw_count < 1:
                raise ValueError("plots.video.sampling.plan_snapshots values must be positive integers")
            normalized[name] = int(raw_count)
        return normalized

    @model_validator(mode="after")
    def _plan_snapshot_budget_ok(self) -> "PlotVideoSamplingConfig":
        requested = sum(int(value) for value in self.plan_snapshots.values())
        if requested > int(self.max_snapshots):
            raise ValueError("plots.video.sampling.plan_snapshots total must be <= plots.video.sampling.max_snapshots")
        return self


class PlotVideoPlaybackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_duration_sec: float = 14.0
    fps: int = 12

    @field_validator("target_duration_sec")
    @classmethod
    def _target_duration_ok(cls, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError("plots.video.playback.target_duration_sec must be numeric")
        duration = float(value)
        if duration < 3.0 or duration > 20.0:
            raise ValueError("plots.video.playback.target_duration_sec must be between 3 and 20")
        return duration

    @field_validator("fps")
    @classmethod
    def _fps_ok(cls, value: int) -> int:
        if not isinstance(value, int) or value < 8 or value > 20:
            raise ValueError("plots.video.playback.fps must be between 8 and 20")
        return int(value)


class PlotVideoLimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_total_frames: int = 180
    max_estimated_render_sec: float = 30.0

    @field_validator("max_total_frames")
    @classmethod
    def _max_total_frames_ok(cls, value: int) -> int:
        if not isinstance(value, int) or value < 2:
            raise ValueError("plots.video.limits.max_total_frames must be >= 2")
        return int(value)

    @field_validator("max_estimated_render_sec")
    @classmethod
    def _max_estimated_render_sec_ok(cls, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError("plots.video.limits.max_estimated_render_sec must be numeric")
        seconds = float(value)
        if seconds <= 0.0:
            raise ValueError("plots.video.limits.max_estimated_render_sec must be > 0")
        return seconds


class PlotVideoPresentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    palette: Dict[str, str] = Field(default_factory=dict)
    legend_font_size: Optional[int] = None
    legend_height_px: Optional[float] = None

    @field_validator("palette")
    @classmethod
    def _palette_ok(cls, value: Dict[str, str]) -> Dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_tag, raw_color in dict(value or {}).items():
            tag = str(raw_tag).strip()
            color = str(raw_color).strip()
            if not tag:
                raise ValueError("plots.video.presentation.palette keys must be non-empty tags")
            if not color:
                raise ValueError("plots.video.presentation.palette values must be non-empty color strings")
            normalized[tag] = color
        return normalized

    @field_validator("legend_font_size")
    @classmethod
    def _legend_font_size_ok(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if not isinstance(value, int) or int(value) < 8 or int(value) > 48:
            raise ValueError("plots.video.presentation.legend_font_size must be between 8 and 48")
        return int(value)

    @field_validator("legend_height_px")
    @classmethod
    def _legend_height_px_ok(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            raise ValueError("plots.video.presentation.legend_height_px must be numeric")
        height = float(value)
        if height < 0.0 or height > 400.0:
            raise ValueError("plots.video.presentation.legend_height_px must be between 0 and 400")
        return height


class PlotVideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    show_title: bool = True
    show_subtitle: bool = True
    mode: Literal["all_plans_round_robin_single_video", "single_plan_single_video"] = (
        "all_plans_round_robin_single_video"
    )
    single_plan_name: Optional[str] = None
    output_name: str = "showcase.mp4"
    sampling: PlotVideoSamplingConfig = Field(default_factory=PlotVideoSamplingConfig)
    playback: PlotVideoPlaybackConfig = Field(default_factory=PlotVideoPlaybackConfig)
    limits: PlotVideoLimitsConfig = Field(default_factory=PlotVideoLimitsConfig)
    presentation: PlotVideoPresentationConfig = Field(default_factory=PlotVideoPresentationConfig)

    @field_validator("output_name")
    @classmethod
    def _output_name_ok(cls, value: str) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("plots.video.output_name must be a non-empty filename")
        if "/" in name or "\\" in name:
            raise ValueError("plots.video.output_name must be a flat filename")
        if not name.lower().endswith(".mp4"):
            raise ValueError("plots.video.output_name must end with '.mp4'")
        return name

    @model_validator(mode="after")
    def _single_plan_rules(self) -> "PlotVideoConfig":
        if self.mode == "single_plan_single_video":
            if self.single_plan_name is None or not str(self.single_plan_name).strip():
                raise ValueError(
                    "plots.video.single_plan_name is required when plots.video.mode='single_plan_single_video'"
                )
            self.single_plan_name = str(self.single_plan_name).strip()
        return self

    @model_validator(mode="after")
    def _frame_budget_guardrail(self) -> "PlotVideoConfig":
        frame_budget = int(round(float(self.playback.target_duration_sec) * float(self.playback.fps)))
        if frame_budget > int(self.limits.max_total_frames):
            raise ValueError(
                "plots.video.playback target frames exceed plots.video.limits.max_total_frames; "
                "reduce target_duration_sec/fps or raise max_total_frames"
            )
        return self


class StageASummaryPlotOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StageBScopePlotOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["auto", "per_plan", "per_group"] = "auto"
    max_plans: int = 12
    drilldown_plans: int = 0

    @field_validator("max_plans")
    @classmethod
    def _max_plans_ok(cls, value: int) -> int:
        if not isinstance(value, int) or int(value) <= 0:
            raise ValueError("plots.options.<plot>.max_plans must be > 0")
        return int(value)

    @field_validator("drilldown_plans")
    @classmethod
    def _drilldown_plans_ok(cls, value: int) -> int:
        if not isinstance(value, int) or int(value) < 0:
            raise ValueError("plots.options.<plot>.drilldown_plans must be >= 0")
        return int(value)


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    out_dir: str = "outputs/plots"
    format: Literal["png", "pdf", "svg"] = "pdf"
    source: Optional[Literal["usr", "parquet"]] = None
    default: List[str] = Field(default_factory=list)
    options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    sample_rows: Optional[int] = None
    allow_truncated: bool = False
    video: PlotVideoConfig = Field(default_factory=PlotVideoConfig)

    @field_validator("sample_rows")
    @classmethod
    def _sample_rows_ok(cls, v: Optional[int]):
        if v is None:
            return v
        if int(v) <= 0:
            raise ValueError("plots.sample_rows must be > 0")
        return int(v)

    @field_validator("default")
    @classmethod
    def _validate_default_plot_ids(cls, value: List[str]) -> List[str]:
        normalized: list[str] = []
        for raw_name in list(value or []):
            name = str(raw_name).strip()
            if not name:
                raise ValueError("plots.default must not contain empty plot ids")
            if name in LEGACY_PUBLIC_PLOT_IDS:
                raise ValueError(f"plots.default.{name} is no longer supported; use concrete plot ids instead")
            if name not in CURRENT_PUBLIC_PLOT_IDS:
                raise ValueError(f"plots.default.{name} is not a supported DenseGen plot id")
            normalized.append(name)
        return normalized

    @field_validator("options")
    @classmethod
    def _validate_known_plot_options(cls, value: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        options = {str(name): dict(payload or {}) for name, payload in dict(value or {}).items()}
        for name in list(options):
            if name in LEGACY_PUBLIC_PLOT_IDS:
                raise ValueError(f"plots.options.{name} is no longer supported; use concrete plot ids instead")
            if name not in SUPPORTED_PLOT_OPTION_BLOCKS:
                raise ValueError(
                    "plots.options."
                    + name
                    + " is not supported; supported plot option blocks: "
                    + ", ".join(sorted(SUPPORTED_PLOT_OPTION_BLOCKS))
                )
        if "placement_occupancy_map" in options:
            options["placement_occupancy_map"] = StageBScopePlotOptions.model_validate(
                options["placement_occupancy_map"]
            ).model_dump(exclude_none=False)
        if "tfbs_concentration_profile" in options:
            options["tfbs_concentration_profile"] = StageBScopePlotOptions.model_validate(
                options["tfbs_concentration_profile"]
            ).model_dump(exclude_none=False)
        return options
