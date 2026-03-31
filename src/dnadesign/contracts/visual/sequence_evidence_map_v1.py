"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/sequence_evidence_map_v1.py

Shared nucleotide-evidence visual contract.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import JsonMap, VisualContractModel


class SequenceEvidenceDisplayV1(VisualContractModel):
    title: str | None = None


class SequenceEvidenceOwnerSpanV1(VisualContractModel):
    owner_id: str
    row_id: Literal["primary", "complement"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    display_label: str
    short_label: str

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SequenceEvidenceOwnerSpanV1":
        if self.end <= self.start:
            raise ValueError("owner span end must be > start")
        return self


class SequenceEvidenceEffectSpanV1(VisualContractModel):
    tag_id: str
    tag_kind: str
    row_id: Literal["primary", "complement"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    display_label: str
    short_label: str

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SequenceEvidenceEffectSpanV1":
        if self.end <= self.start:
            raise ValueError("effect span end must be > start")
        return self


class SequenceEvidenceBoundaryV1(VisualContractModel):
    boundary_id: str
    row_id: Literal["primary", "complement"]
    boundary: int = Field(ge=0)
    boundary_kind: Literal["cut", "nick", "ligation_junction"]
    display_label: str
    short_label: str


class SequenceEvidencePairingV1(VisualContractModel):
    pairing_id: str
    primary_start: int = Field(ge=0)
    primary_end: int = Field(gt=0)
    complement_start: int = Field(ge=0)
    complement_end: int = Field(gt=0)
    display_label: str | None = None
    short_label: str | None = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SequenceEvidencePairingV1":
        if self.primary_end <= self.primary_start:
            raise ValueError("pairing primary span end must be > start")
        if self.complement_end <= self.complement_start:
            raise ValueError("pairing complement span end must be > start")
        return self


class SequenceEvidenceMapV1(VisualContractModel):
    contract_kind: Literal["sequence_evidence_map_v1"] = "sequence_evidence_map_v1"
    state_id: str
    topology_kind: Literal[
        "linear_ssdna",
        "linear_dsdna",
        "circularized_linearized",
        "hairpin_folded",
        "branched_adapter",
    ]
    alphabet: Literal["dna", "iupac_dna"] = "dna"
    primary_sequence: str
    complement_sequence: str | None = None
    owners: list[SequenceEvidenceOwnerSpanV1] = Field(default_factory=list)
    effect_tags: list[SequenceEvidenceEffectSpanV1] = Field(default_factory=list)
    boundaries: list[SequenceEvidenceBoundaryV1] = Field(default_factory=list)
    pairings: list[SequenceEvidencePairingV1] = Field(default_factory=list)
    display: SequenceEvidenceDisplayV1 = Field(default_factory=SequenceEvidenceDisplayV1)
    meta: JsonMap = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> "SequenceEvidenceMapV1":
        if not self.primary_sequence:
            raise ValueError("primary_sequence must be non-empty")
        primary_length = len(self.primary_sequence)
        complement_length = len(self.complement_sequence) if self.complement_sequence is not None else primary_length
        for owner in self.owners:
            limit = primary_length if owner.row_id == "primary" else complement_length
            if owner.end > limit:
                raise ValueError("owner span exceeds row sequence length")
        for tag in self.effect_tags:
            limit = primary_length if tag.row_id == "primary" else complement_length
            if tag.end > limit:
                raise ValueError("effect span exceeds row sequence length")
        for boundary in self.boundaries:
            limit = primary_length if boundary.row_id == "primary" else complement_length
            if boundary.boundary > limit:
                raise ValueError("boundary exceeds row sequence length")
        for pairing in self.pairings:
            if pairing.primary_end > primary_length:
                raise ValueError("pairing primary span exceeds primary sequence length")
            if pairing.complement_end > complement_length:
                raise ValueError("pairing complement span exceeds complement sequence length")
        return self
