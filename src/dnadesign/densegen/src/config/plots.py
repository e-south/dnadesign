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


class PlotVideoSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stride: int = 5
    max_source_rows: int = 20_000
    max_snapshots: int = 140

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


class PlotVideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    mode: Literal["all_plans_round_robin_single_video", "single_plan_single_video"] = (
        "all_plans_round_robin_single_video"
    )
    single_plan_name: Optional[str] = None
    output_name: str = "showcase.mp4"
    sampling: PlotVideoSamplingConfig = Field(default_factory=PlotVideoSamplingConfig)
    playback: PlotVideoPlaybackConfig = Field(default_factory=PlotVideoPlaybackConfig)
    limits: PlotVideoLimitsConfig = Field(default_factory=PlotVideoLimitsConfig)

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
