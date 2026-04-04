"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/spec_models.py

Strict v4 spec and PWM-context models for payload-centric YIU workflows.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio import normalize_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.errors import (
    YIU_CONTRACT_UNKNOWN,
    YIU_INPUT_KIND_UNKNOWN,
    YIU_INPUT_MUTUALLY_EXCLUSIVE,
    YIU_JUNCTION_INVALID,
    YIU_MISMATCH_INVALID,
    YIU_PATH_INVALID,
    YIU_PWM_CONTEXT_INVALID,
    YIU_SCHEMA_VERSION_UNSUPPORTED,
    YIU_SEQUENCE_INVALID,
)

_BASES = ("A", "C", "G", "T")
_SECONDARY_OBJECTIVE_LADDER = (
    "total_loss",
    "midpoint_proximity",
    "body_length_balance",
    "terminal_position_avoidance",
    "default_strand_preference",
    "lexical_stability",
)
_PWM_SOURCE_KINDS = {"none", "sample_context", "file", "inline"}


def _normalize_sequence(value: str, *, ctx: str) -> str:
    try:
        return normalize_iupac(value)
    except Exception as exc:
        raise ValueError(f"{YIU_SEQUENCE_INVALID}: invalid {ctx} ({exc})") from exc


class YiuSpecRoot(StrictBaseModel):
    schema_version: Literal[1] = 1
    contract: Literal["split_yiu_payload_rendering_v4"] = "split_yiu_payload_rendering_v4"
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("yiu.name must be non-empty")
        return text


class UserSequenceInput(StrictBaseModel):
    sequence: str

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return _normalize_sequence(value, ctx="input.user_sequence.sequence")


class SampleHitInput(StrictBaseModel):
    hit_id: str
    sample_name: str
    payload_sequence: str | None = None
    source_artifact_path: str | None = None
    source_artifact: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("hit_id", "sample_name")
    @classmethod
    def _validate_required_text(cls, value: str, info) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{info.field_name} must be non-empty")
        return text

    @field_validator("payload_sequence")
    @classmethod
    def _validate_payload_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_sequence(value, ctx="input.sample_hit.payload_sequence")

    @field_validator("source_artifact_path", "source_artifact")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    @model_validator(mode="after")
    def _validate_resolution_paths(self) -> "SampleHitInput":
        source_workspace = self.metadata.get("source_workspace")
        has_workspace_ref = isinstance(source_workspace, str) and str(source_workspace).strip() != ""
        has_artifact_ref = self.source_artifact_path is not None or self.source_artifact is not None
        if self.payload_sequence is None and not has_artifact_ref and not has_workspace_ref:
            raise ValueError(
                "sample_hit requires payload_sequence or a resolvable source artifact reference "
                "(source_artifact_path, source_artifact, or metadata.source_workspace)."
            )
        return self


class InputSpec(StrictBaseModel):
    kind: Literal["user_sequence", "sample_hit"]
    user_sequence: UserSequenceInput | None = None
    sample_hit: SampleHitInput | None = None

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        kind = str(value).strip()
        if kind not in {"user_sequence", "sample_hit"}:
            raise ValueError(f"{YIU_INPUT_KIND_UNKNOWN}: unsupported input.kind={value!r}")
        return kind

    @model_validator(mode="after")
    def _validate_subtype(self) -> "InputSpec":
        declared = {
            "user_sequence": self.user_sequence is not None,
            "sample_hit": self.sample_hit is not None,
        }
        if sum(declared.values()) != 1:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: exactly one input subtype must be populated")
        if self.kind == "user_sequence" and self.user_sequence is None:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: input.kind=user_sequence requires input.user_sequence")
        if self.kind == "sample_hit" and self.sample_hit is None:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: input.kind=sample_hit requires input.sample_hit")
        if self.kind != "user_sequence" and self.user_sequence is not None:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: input.user_sequence is incompatible with input.kind")
        if self.kind != "sample_hit" and self.sample_hit is not None:
            raise ValueError(f"{YIU_INPUT_MUTUALLY_EXCLUSIVE}: input.sample_hit is incompatible with input.kind")
        return self


class JunctionOptimizationSpec(StrictBaseModel):
    mode: Literal["derived", "explicit_window", "optimize"] = "derived"
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
    candidate_positions: list[int] = Field(default_factory=lambda: [1, 2])
    allowed_strands: list[Literal["complement", "payload"]] = Field(default_factory=lambda: ["complement", "payload"])
    strand_mode: Literal["per_position"] = "per_position"
    default_strand_preference: Literal["complement", "payload"] = "complement"

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
        return self


class YiuPwmProbabilities(StrictBaseModel):
    alphabet: list[str]
    rows: list[list[float]]

    @field_validator("alphabet")
    @classmethod
    def _validate_alphabet(cls, value: list[str]) -> list[str]:
        alphabet = [str(item).strip().upper() for item in value]
        if alphabet != list(_BASES):
            raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: probabilities.alphabet must equal [A, C, G, T]")
        return alphabet

    @field_validator("rows")
    @classmethod
    def _validate_rows(cls, value: list[list[float]]) -> list[list[float]]:
        rows: list[list[float]] = []
        if not value:
            raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: probabilities.rows must be non-empty")
        for idx, row in enumerate(value):
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError(
                    f"{YIU_PWM_CONTEXT_INVALID}: probabilities.rows[{idx}] must contain exactly 4 values [A,C,G,T]"
                )
            parsed = [float(item) for item in row]
            if any(not math.isfinite(item) or item < 0.0 for item in parsed):
                raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: probabilities.rows[{idx}] must be finite and >= 0")
            total = sum(parsed)
            if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"{YIU_PWM_CONTEXT_INVALID}: probabilities.rows[{idx}] must be normalized to sum to 1.0"
                )
            rows.append(parsed)
        return rows


class YiuPwmProvenance(StrictBaseModel):
    source_kind: Literal["file", "sample_context", "inline"]
    source_ref: str

    @field_validator("source_ref")
    @classmethod
    def _validate_ref(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("provenance.source_ref must be non-empty")
        return text


class YiuPwmMotifInstanceV1(StrictBaseModel):
    motif_instance_id: str
    tf_name: str
    motif_name: str
    reference_strand: Literal["+", "-"]
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    probabilities: YiuPwmProbabilities
    provenance: YiuPwmProvenance

    @field_validator("motif_instance_id", "tf_name", "motif_name")
    @classmethod
    def _validate_text(cls, value: str, info) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{info.field_name} must be non-empty")
        return text

    @model_validator(mode="after")
    def _validate_length(self) -> "YiuPwmMotifInstanceV1":
        if self.end <= self.start:
            raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: motif intervals must be forward and non-empty")
        if len(self.probabilities.rows) != (self.end - self.start):
            raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: motif interval length must match PWM row count")
        return self


class YiuPwmContextV1(StrictBaseModel):
    contract: Literal["yiu_pwm_context_v1"] = "yiu_pwm_context_v1"
    schema_version: Literal[1] = 1
    name: str
    motifs: list[YiuPwmMotifInstanceV1]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("name must be non-empty")
        return text

    @model_validator(mode="after")
    def _validate_motifs(self) -> "YiuPwmContextV1":
        if not self.motifs:
            raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: yiu_pwm_context_v1.motifs must be non-empty")
        by_id: set[str] = set()
        signatures: dict[tuple[object, ...], tuple[tuple[float, ...], ...]] = {}
        for motif in self.motifs:
            if motif.motif_instance_id in by_id:
                raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: duplicate motif_instance_id {motif.motif_instance_id!r}")
            by_id.add(motif.motif_instance_id)
            signature = (
                motif.start,
                motif.end,
                motif.reference_strand,
                motif.motif_name,
                motif.tf_name,
            )
            matrix = tuple(tuple(row) for row in motif.probabilities.rows)
            if signature in signatures and signatures[signature] != matrix:
                raise ValueError(
                    f"{YIU_PWM_CONTEXT_INVALID}: duplicate motif definitions with conflicting matrices are ambiguous"
                )
            signatures[signature] = matrix
        return self


class PwmSourceSpec(StrictBaseModel):
    kind: Literal["none", "sample_context", "file", "inline"] = "none"
    path: str | None = None
    inline_context: YiuPwmContextV1 | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "PwmSourceSpec":
        if self.kind == "file":
            if self.path is None:
                raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: optimization.pwm.source.path is required for file")
            if self.inline_context is not None:
                raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: inline_context is not valid for file source")
        elif self.kind == "inline":
            if self.inline_context is None:
                raise ValueError(
                    f"{YIU_PWM_CONTEXT_INVALID}: optimization.pwm.source.inline_context is required for inline"
                )
            if self.path is not None:
                raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: path is not valid for inline source")
        else:
            if self.path is not None or self.inline_context is not None:
                raise ValueError(f"{YIU_PWM_CONTEXT_INVALID}: source extras are invalid for kind={self.kind}")
        return self


class PwmObjectiveSpec(StrictBaseModel):
    primary: Literal["maximin"] = "maximin"
    secondary: list[
        Literal[
            "total_loss",
            "midpoint_proximity",
            "body_length_balance",
            "terminal_position_avoidance",
            "default_strand_preference",
            "lexical_stability",
        ]
    ] = Field(default_factory=lambda: list(_SECONDARY_OBJECTIVE_LADDER))

    @field_validator("secondary")
    @classmethod
    def _validate_secondary(cls, value: list[str]) -> list[str]:
        secondary = [str(item).strip() for item in value]
        if secondary != list(_SECONDARY_OBJECTIVE_LADDER):
            raise ValueError(
                "optimization.pwm.objective.secondary must use the canonical Yiu v4 ladder order: "
                + ", ".join(_SECONDARY_OBJECTIVE_LADDER)
            )
        return secondary


class PwmOptimizationSpec(StrictBaseModel):
    mode: Literal["none", "use_if_available", "require"] = "none"
    source: PwmSourceSpec = Field(default_factory=PwmSourceSpec)
    objective: PwmObjectiveSpec = Field(default_factory=PwmObjectiveSpec)

    @model_validator(mode="after")
    def _validate_mode(self) -> "PwmOptimizationSpec":
        if self.mode == "none" and self.source.kind != "none":
            raise ValueError("optimization.pwm.mode=none requires optimization.pwm.source.kind=none")
        if self.mode in {"use_if_available", "require"} and self.source.kind == "none":
            raise ValueError(
                "optimization.pwm.mode requires optimization.pwm.source.kind to be sample_context, file, or inline"
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
        return cls._validate_workspace_relative_path(value=value, field_name="output.bundle_dir")

    @field_validator("published_plot_path")
    @classmethod
    def _validate_published_plot_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        path = cls._validate_workspace_relative_path(value=value, field_name="output.published_plot_path")
        if path.suffix.lower() != ".pdf":
            raise ValueError("output.published_plot_path must point to a .pdf artifact")
        return path

    @staticmethod
    def _validate_workspace_relative_path(*, value: Path, field_name: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError(f"{YIU_PATH_INVALID}: {field_name} must be relative to the workspace root")
        if any(part == ".." for part in path.parts):
            raise ValueError(f"{YIU_PATH_INVALID}: {field_name} must not traverse outside the workspace root")
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
