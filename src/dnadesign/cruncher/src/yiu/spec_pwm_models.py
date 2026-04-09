"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/spec_pwm_models.py

PWM-facing YIU spec models and validation rules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.errors import YIU_PWM_CONTEXT_INVALID
from dnadesign.cruncher.yiu.spec_common import (
    BASES,
    SECONDARY_OBJECTIVE_LADDER,
    require_non_empty_text,
)


class YiuPwmProbabilities(StrictBaseModel):
    alphabet: list[str]
    rows: list[list[float]]

    @field_validator("alphabet")
    @classmethod
    def _validate_alphabet(cls, value: list[str]) -> list[str]:
        alphabet = [str(item).strip().upper() for item in value]
        if alphabet != list(BASES):
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
        return require_non_empty_text(value, field_name="provenance.source_ref")


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
        return require_non_empty_text(value, field_name=str(info.field_name))

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
        return require_non_empty_text(value, field_name="name")

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
    secondary: list[str] = Field(default_factory=lambda: list(SECONDARY_OBJECTIVE_LADDER))

    @field_validator("secondary")
    @classmethod
    def _validate_secondary(cls, value: list[str]) -> list[str]:
        secondary = [str(item).strip() for item in value]
        if secondary != list(SECONDARY_OBJECTIVE_LADDER):
            raise ValueError(
                "optimization.pwm.objective.secondary must use the canonical Yiu v4 ladder order: "
                + ", ".join(SECONDARY_OBJECTIVE_LADDER)
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


__all__ = [
    "PwmObjectiveSpec",
    "PwmOptimizationSpec",
    "PwmSourceSpec",
    "YiuPwmContextV1",
    "YiuPwmMotifInstanceV1",
    "YiuPwmProbabilities",
    "YiuPwmProvenance",
]
