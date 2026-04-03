"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/yiu_payload_visual_v1.py

Shared YIU payload visual contract with optional PWM motif layers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import JsonMap, PositiveLengthSpan, VisualContractModel


class YiuPayloadDisplayV1(VisualContractModel):
    title: str | None = None


class YiuPayloadJunctionV1(PositiveLengthSpan):
    offsets: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])

    @model_validator(mode="after")
    def _validate_offsets(self) -> "YiuPayloadJunctionV1":
        super()._validate_positive_length()
        if self.end - self.start != 4:
            raise ValueError("junction span length must equal 4")
        if self.offsets != [0, 1, 2, 3]:
            raise ValueError("junction offsets must equal [0, 1, 2, 3]")
        return self


class YiuPayloadMismatchV1(VisualContractModel):
    payload_index: int = Field(ge=0)
    junction_offset: int = Field(ge=0, le=3)
    mutated_strand: Literal["payload", "complement"]
    native_base: Literal["A", "C", "G", "T"]
    mutated_base: Literal["A", "C", "G", "T"]
    opposing_base: Literal["A", "C", "G", "T"]

    @model_validator(mode="after")
    def _validate_change(self) -> "YiuPayloadMismatchV1":
        if self.native_base == self.mutated_base:
            raise ValueError("mutated_base must differ from native_base")
        return self


class YiuPayloadMotifLayerV1(PositiveLengthSpan):
    motif_instance_id: str
    tf_name: str
    motif_name: str
    reference_strand: Literal["+", "-"]
    label: str
    matrix: list[list[float]]

    @model_validator(mode="after")
    def _validate_matrix(self) -> "YiuPayloadMotifLayerV1":
        super()._validate_positive_length()
        if len(self.matrix) != (self.end - self.start):
            raise ValueError("motif layer matrix length must match the payload span")
        for idx, row in enumerate(self.matrix):
            if not isinstance(row, list) or len(row) < 4:
                raise ValueError(f"motif layer matrix row {idx} must contain at least 4 values [A,C,G,T]")
        return self


class YiuPayloadVisualV1(VisualContractModel):
    contract_kind: Literal["yiu_payload_visual_v1"] = "yiu_payload_visual_v1"
    schema_version: Literal[1] = 1
    state_id: str
    alphabet: Literal["dna", "iupac_dna"] = "iupac_dna"
    reference_payload_sequence: str
    selected_payload_sequence: str
    selected_complement_sequence: str
    show_reference_payload_row: bool = False
    junction: YiuPayloadJunctionV1
    mismatches: list[YiuPayloadMismatchV1] = Field(default_factory=list)
    motif_layers: list[YiuPayloadMotifLayerV1] = Field(default_factory=list)
    display: YiuPayloadDisplayV1 = Field(default_factory=YiuPayloadDisplayV1)
    meta: JsonMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sequences(self) -> "YiuPayloadVisualV1":
        if not self.state_id.strip():
            raise ValueError("state_id must be non-empty")
        if not self.reference_payload_sequence:
            raise ValueError("reference_payload_sequence must be non-empty")
        expected_length = len(self.reference_payload_sequence)
        if len(self.selected_payload_sequence) != expected_length:
            raise ValueError("selected_payload_sequence length must match reference_payload_sequence")
        if len(self.selected_complement_sequence) != expected_length:
            raise ValueError("selected_complement_sequence length must match reference_payload_sequence")
        if self.junction.end > expected_length:
            raise ValueError("junction span must fit within the selected payload sequence")
        mismatch_indices = set()
        for mismatch in self.mismatches:
            if mismatch.payload_index < self.junction.start or mismatch.payload_index >= self.junction.end:
                raise ValueError("mismatch payload_index must fall inside the junction span")
            if mismatch.payload_index in mismatch_indices:
                raise ValueError("mismatch payload_index values must be unique")
            mismatch_indices.add(mismatch.payload_index)
        motif_ids = set()
        for motif in self.motif_layers:
            if motif.end > expected_length:
                raise ValueError("motif layer span must fit within the selected payload sequence")
            if motif.motif_instance_id in motif_ids:
                raise ValueError("motif_instance_id values must be unique")
            motif_ids.add(motif.motif_instance_id)
        return self
