"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/spec_rendering_models.py

Rendering-facing YIU spec models and cross-field validation rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.errors import (
    YIU_CONTRACT_UNKNOWN,
    YIU_JUNCTION_INVALID,
    YIU_MISMATCH_INVALID,
    YIU_PWM_CONTEXT_INVALID,
    YIU_SCHEMA_VERSION_UNSUPPORTED,
)
from dnadesign.cruncher.yiu.spec_common import validate_workspace_relative_path
from dnadesign.cruncher.yiu.spec_input_models import InputSpec, YiuSpecRoot
from dnadesign.cruncher.yiu.spec_pwm_models import PwmOptimizationSpec


class JunctionOptimizationSpec(StrictBaseModel):
    mode: Literal["center_locked", "explicit_window", "optimize"] = "center_locked"
    start: int | None = None
    end: int | None = None
    overhang_length: Literal[4] = 4
    max_payload_body_length: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "JunctionOptimizationSpec":
        if self.overhang_length != 4:
            raise ValueError(f"{YIU_JUNCTION_INVALID}: optimization.junction.overhang_length must equal 4")
        if self.mode == "explicit_window":
            if self.start is None or self.end is None:
                raise ValueError(
                    f"{YIU_JUNCTION_INVALID}: optimization.junction.start/end are required for explicit_window"
                )
            if self.start < 0 or self.end <= self.start:
                raise ValueError(f"{YIU_JUNCTION_INVALID}: explicit junction window must be forward and non-empty")
            if self.end - self.start != 4:
                raise ValueError(f"{YIU_JUNCTION_INVALID}: explicit junction window length must equal 4")
        elif self.start is not None or self.end is not None:
            raise ValueError(
                f"{YIU_JUNCTION_INVALID}: optimization.junction.start/end are only valid for explicit_window"
            )
        return self


class MismatchesSpec(StrictBaseModel):
    count: Literal[1, 2]
    candidate_positions: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    allowed_strands: list[Literal["complement", "payload"]] = Field(default_factory=lambda: ["complement", "payload"])
    strand_mode: Literal["per_position"] = "per_position"
    default_strand_preference: Literal["complement", "payload"] = "complement"
    ligation_profile: Literal["none", "t4", "t7", "t3", "pbcv1", "hlig3"] = "none"
    ligation_awareness_mode: Literal["disabled", "secondary"] = "secondary"
    bad_pattern_heuristics: bool = False
    ligation_selection_mode: Literal["secondary", "pwm_tolerance_then_ligation", "hard_ligation_filter"] = "secondary"
    pwm_worst_loss_tolerance: float = Field(default=0.0, ge=0.0)
    pwm_total_loss_tolerance: float = Field(default=0.0, ge=0.0)
    max_worst_mismatch_class_tier: int = Field(default=0, ge=0, le=3)
    max_middle_mismatch_count: int = Field(default=1, ge=0, le=2)
    allow_double_middle: bool = False
    allow_tnna_like_overhangs: bool = False

    @field_validator("ligation_selection_mode", mode="before")
    @classmethod
    def _normalize_ligation_selection_mode(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "hard_filter":
            return "hard_ligation_filter"
        return value

    @field_validator("candidate_positions")
    @classmethod
    def _validate_positions(cls, value: list[int]) -> list[int]:
        positions = [int(item) for item in value]
        if not positions:
            raise ValueError(f"{YIU_MISMATCH_INVALID}: candidate_positions must be non-empty")
        if len(set(positions)) != len(positions):
            raise ValueError(f"{YIU_MISMATCH_INVALID}: candidate_positions must be unique")
        if any(position not in {0, 1, 2, 3} for position in positions):
            raise ValueError(f"{YIU_MISMATCH_INVALID}: candidate_positions must be a subset of 0..3")
        return sorted(positions)

    @field_validator("allowed_strands")
    @classmethod
    def _validate_allowed_strands(
        cls, value: list[Literal["complement", "payload"]]
    ) -> list[Literal["complement", "payload"]]:
        strands = [str(item).strip() for item in value]
        if not strands:
            raise ValueError(f"{YIU_MISMATCH_INVALID}: allowed_strands must be non-empty")
        if len(set(strands)) != len(strands):
            raise ValueError(f"{YIU_MISMATCH_INVALID}: allowed_strands must be unique")
        if any(strand not in {"complement", "payload"} for strand in strands):
            raise ValueError(f"{YIU_MISMATCH_INVALID}: allowed_strands must contain payload/complement only")
        return [item for item in ("complement", "payload") if item in set(strands)]

    @model_validator(mode="after")
    def _validate_count(self) -> "MismatchesSpec":
        if self.strand_mode != "per_position":
            raise ValueError(f"{YIU_MISMATCH_INVALID}: optimization.mismatches.strand_mode must be per_position")
        if self.count > len(self.candidate_positions):
            raise ValueError(f"{YIU_MISMATCH_INVALID}: mismatches.count exceeds the candidate position pool size")
        if self.bad_pattern_heuristics and self.ligation_awareness_mode != "secondary":
            raise ValueError(
                f"{YIU_MISMATCH_INVALID}: bad_pattern_heuristics requires ligation_awareness_mode=secondary"
            )
        if self.bad_pattern_heuristics and self.ligation_profile == "none":
            raise ValueError(f"{YIU_MISMATCH_INVALID}: bad_pattern_heuristics requires a ligation_profile")
        if self.ligation_selection_mode != "secondary" and self.ligation_awareness_mode != "secondary":
            raise ValueError(
                f"{YIU_MISMATCH_INVALID}: ligation_selection_mode={self.ligation_selection_mode} "
                "requires ligation_awareness_mode=secondary"
            )
        if self.ligation_selection_mode != "secondary" and self.ligation_profile == "none":
            raise ValueError(
                f"{YIU_MISMATCH_INVALID}: ligation_selection_mode={self.ligation_selection_mode} "
                "requires ligation_profile to name a ligase instead of none"
            )
        if self.ligation_selection_mode != "pwm_tolerance_then_ligation" and (
            self.pwm_worst_loss_tolerance > 0.0 or self.pwm_total_loss_tolerance > 0.0
        ):
            raise ValueError(
                f"{YIU_MISMATCH_INVALID}: pwm_*_loss_tolerance fields require "
                "ligation_selection_mode=pwm_tolerance_then_ligation"
            )
        if self.ligation_selection_mode != "hard_ligation_filter" and (
            self.max_worst_mismatch_class_tier != 0
            or self.max_middle_mismatch_count != 1
            or self.allow_double_middle
            or self.allow_tnna_like_overhangs
        ):
            raise ValueError(
                f"{YIU_MISMATCH_INVALID}: hard_ligation_filter-only fields require "
                "ligation_selection_mode=hard_ligation_filter"
            )
        return self


class OptimizationSpec(StrictBaseModel):
    junction: JunctionOptimizationSpec = Field(default_factory=JunctionOptimizationSpec)
    mismatches: MismatchesSpec
    pwm: PwmOptimizationSpec = Field(default_factory=PwmOptimizationSpec)


class OutputSpec(StrictBaseModel):
    bundle_dir: Path
    published_plot_path: Path | None = None
    emit_render_jobs_debug: bool = False

    @field_validator("bundle_dir")
    @classmethod
    def _validate_bundle_dir(cls, value: Path) -> Path:
        return validate_workspace_relative_path(value=value, field_name="output.bundle_dir")

    @field_validator("published_plot_path")
    @classmethod
    def _validate_published_plot_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        path = validate_workspace_relative_path(value=value, field_name="output.published_plot_path")
        if path.suffix.lower() != ".pdf":
            raise ValueError("output.published_plot_path must point to a .pdf artifact")
        return path


class YiuPayloadRenderingSpec(StrictBaseModel):
    yiu: YiuSpecRoot
    input: InputSpec
    optimization: OptimizationSpec
    output: OutputSpec

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_topology(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for legacy_key in ("junction", "bulge_mask", "split"):
                if legacy_key in value:
                    raise ValueError(
                        "split_yiu_payload_rendering_v4 uses top-level optimization.{junction,mismatches,pwm}; "
                        f"legacy key {legacy_key!r} is not supported."
                    )
        return value

    @model_validator(mode="after")
    def _validate_cross_field_contract(self) -> "YiuPayloadRenderingSpec":
        if self.yiu.contract != "split_yiu_payload_rendering_v4":
            raise ValueError(
                f"{YIU_CONTRACT_UNKNOWN}: yiu.contract must equal split_yiu_payload_rendering_v4 for v4 specs"
            )
        if self.yiu.schema_version != 1:
            raise ValueError(
                f"{YIU_SCHEMA_VERSION_UNSUPPORTED}: split_yiu_payload_rendering_v4 only supports schema_version=1"
            )
        if self.optimization.pwm.source.kind == "sample_context" and self.input.kind != "sample_hit":
            raise ValueError(
                f"{YIU_PWM_CONTEXT_INVALID}: optimization.pwm.source.kind=sample_context requires input.kind=sample_hit"
            )
        return self


__all__ = [
    "JunctionOptimizationSpec",
    "MismatchesSpec",
    "OptimizationSpec",
    "OutputSpec",
    "YiuPayloadRenderingSpec",
]
