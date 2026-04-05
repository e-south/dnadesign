"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/domain_models.py

Pure-domain models for normalized YIU payloads and optimization outcomes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.spec_pwm_models import YiuPwmMotifInstanceV1


class JunctionSelection(StrictBaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    offsets: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    mode: Literal["derived", "center_locked", "explicit_window", "optimize"]
    left_body_length: int = Field(ge=1)
    right_body_length: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "JunctionSelection":
        if self.end - self.start != 4:
            raise ValueError("junction width must equal 4")
        if self.offsets != [0, 1, 2, 3]:
            raise ValueError("junction offsets must equal [0, 1, 2, 3]")
        return self


class MismatchSelection(StrictBaseModel):
    payload_index: int = Field(ge=0)
    junction_offset: int = Field(ge=0, le=3)
    mutated_strand: Literal["payload", "complement"]
    native_base: str
    mutated_base: str
    opposing_base: str

    @model_validator(mode="after")
    def _validate_bases(self) -> "MismatchSelection":
        for field_name in ("native_base", "mutated_base", "opposing_base"):
            value = getattr(self, field_name)
            if value not in {"A", "C", "G", "T"}:
                raise ValueError(f"{field_name} must be one of A/C/G/T")
        if self.native_base == self.mutated_base:
            raise ValueError("mutated_base must differ from native_base")
        return self


class NormalizedMotifContext(StrictBaseModel):
    requested_mode: Literal["none", "use_if_available", "require"]
    effective: bool
    source_kind: Literal["none", "sample_context", "file", "inline"]
    fallback_reason: str | None = None
    motifs: list[YiuPwmMotifInstanceV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_state(self) -> "NormalizedMotifContext":
        if self.effective and not self.motifs:
            raise ValueError("effective motif_context must contain motifs")
        if not self.effective and self.motifs:
            raise ValueError("ineffective motif_context must not carry resolved motifs")
        return self


class OptimizationObjective(StrictBaseModel):
    primary: Literal["maximin"] = "maximin"
    secondary: list[str]


class OptimizationWinner(StrictBaseModel):
    junction_start: int = Field(ge=0)
    junction_end: int = Field(ge=1)
    selected_positions: list[int] = Field(default_factory=list)
    mutated_strands: list[Literal["payload", "complement"]] = Field(default_factory=list)
    mutated_bases: list[str] = Field(default_factory=list)
    worst_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)
    midpoint_distance: int = Field(ge=0)
    body_length_balance: int = Field(ge=0)
    terminal_positions_used: int = Field(ge=0)
    default_strand_preference_count: int = Field(ge=0)
    lexical_key: str

    @model_validator(mode="after")
    def _validate_lengths(self) -> "OptimizationWinner":
        if self.junction_end - self.junction_start != 4:
            raise ValueError("winner junction window must be length 4")
        if len(self.selected_positions) != len(self.mutated_strands):
            raise ValueError("selected_positions and mutated_strands must align")
        if len(self.selected_positions) != len(self.mutated_bases):
            raise ValueError("selected_positions and mutated_bases must align")
        return self


class OptimizationDecision(StrictBaseModel):
    candidate_count: int = Field(ge=1)
    objective: OptimizationObjective
    winner: OptimizationWinner
    trace: list[dict[str, Any]] = Field(default_factory=list)


class NormalizedPayload(StrictBaseModel):
    contract: Literal["yiu_normalized_payload_v4"] = "yiu_normalized_payload_v4"
    schema_version: Literal[1] = 1
    name: str
    input_kind: Literal["user_sequence", "sample_hit"]
    payload_label: str | None = None
    site_label: str | None = None
    reference_payload_sequence: str
    reference_complement_sequence: str
    selected_payload_sequence: str
    selected_complement_sequence: str
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    junction: JunctionSelection
    mismatches: list[MismatchSelection] = Field(default_factory=list)
    motif_context: NormalizedMotifContext
    optimization_decision: OptimizationDecision
    published_artifacts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> "NormalizedPayload":
        payload_length = len(self.reference_payload_sequence)
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        for field_name in (
            "reference_complement_sequence",
            "selected_payload_sequence",
            "selected_complement_sequence",
        ):
            if len(getattr(self, field_name)) != payload_length:
                raise ValueError(f"{field_name} length must match reference_payload_sequence")
        if self.junction.end > payload_length:
            raise ValueError("junction end must lie within the payload")
        expected_positions = {self.junction.start + offset for offset in range(4)}
        seen_positions: set[int] = set()
        for mismatch in self.mismatches:
            if mismatch.payload_index not in expected_positions:
                raise ValueError("mismatch payload_index must lie inside the selected junction window")
            if mismatch.payload_index in seen_positions:
                raise ValueError("mismatch payload_index values must be unique")
            seen_positions.add(mismatch.payload_index)
        return self

    @property
    def payload_length(self) -> int:
        return len(self.reference_payload_sequence)
