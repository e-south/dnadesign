"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/sequence/annotated_sequence_part_v1.py

Neutral immutable annotated sequence-part handoff contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_IUPAC_DNA = frozenset("ACGTRYSWKMBDHVN")


class AnnotatedPartContractModel(BaseModel):
    """Strict immutable base for the annotated-part boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _required_text(value: str, *, label: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    return text


class AnnotatedSequenceSourceRefV1(AnnotatedPartContractModel):
    """Digest-pinned authority for the supplied part or its lineage."""

    kind: Literal["artifact", "record"]
    authority: str
    identifier: str
    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("authority", "identifier")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _required_text(value, label="source reference field")


class AnnotatedSequenceFeatureV1(AnnotatedPartContractModel):
    """One source-owned feature located on an annotated sequence part."""

    feature_id: str
    role: str
    owner: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    orientation: Literal["forward", "reverse_complement", "not_asserted"]
    sequence: str
    source_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("feature_id", "role", "owner")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _required_text(value, label="feature field")

    @field_validator("sequence", mode="before")
    @classmethod
    def normalize_sequence(cls, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("feature sequence must be a non-empty DNA string.")
        if value != value.upper():
            raise ValueError("feature sequence must use canonical uppercase IUPAC DNA.")
        sequence = value
        invalid = sorted(set(sequence) - _IUPAC_DNA)
        if invalid:
            raise ValueError(f"feature sequence contains invalid IUPAC DNA: {', '.join(invalid)}.")
        return sequence

    @model_validator(mode="after")
    def validate_span(self) -> AnnotatedSequenceFeatureV1:
        if self.end <= self.start:
            raise ValueError("feature end must be greater than start.")
        if self.end - self.start != len(self.sequence):
            raise ValueError("feature span length must match its sequence length.")
        return self


class AnnotatedSequencePartV1(AnnotatedPartContractModel):
    """One sequence object that Construct must place without re-derivation."""

    contract: Literal["annotated_sequence_part_v1"] = "annotated_sequence_part_v1"
    schema_version: Literal[1] = 1
    part_id: str
    representation: Literal["one_dimensional_sequence"] = "one_dimensional_sequence"
    molecule_type: Literal["dna"] = "dna"
    strandedness: Literal["not_asserted", "single", "double"]
    topology: Literal["not_asserted", "linear", "circular"]
    coordinate_system: Literal["zero_based_half_open"] = "zero_based_half_open"
    sequence: str
    sequence_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_refs: tuple[AnnotatedSequenceSourceRefV1, ...] = Field(min_length=1)
    features: tuple[AnnotatedSequenceFeatureV1, ...] = Field(min_length=1)

    @field_validator("part_id")
    @classmethod
    def require_part_id(cls, value: str) -> str:
        return _required_text(value, label="part_id")

    @field_validator("sequence", mode="before")
    @classmethod
    def normalize_sequence(cls, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("part sequence must be a non-empty DNA string.")
        if value != value.upper():
            raise ValueError("part sequence must use canonical uppercase IUPAC DNA.")
        sequence = value
        invalid = sorted(set(sequence) - _IUPAC_DNA)
        if invalid:
            raise ValueError(f"part sequence contains invalid IUPAC DNA: {', '.join(invalid)}.")
        return sequence

    @field_validator("source_refs", "features", mode="before")
    @classmethod
    def normalize_sequence_collections(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_part(self) -> AnnotatedSequencePartV1:
        observed_digest = f"sha256:{hashlib.sha256(self.sequence.encode()).hexdigest()}"
        if self.sequence_digest != observed_digest:
            raise ValueError("sequence_digest does not match the annotated part sequence.")
        feature_ids: set[str] = set()
        for feature in self.features:
            if feature.feature_id in feature_ids:
                raise ValueError(f"Duplicate annotated feature_id {feature.feature_id!r}.")
            feature_ids.add(feature.feature_id)
            if feature.end > len(self.sequence):
                raise ValueError(f"Annotated feature {feature.feature_id!r} exceeds the part sequence.")
            if self.sequence[feature.start : feature.end] != feature.sequence:
                raise ValueError(
                    f"Annotated feature sequence does not match part coordinates for {feature.feature_id!r}."
                )
        return self


__all__ = [
    "AnnotatedSequenceFeatureV1",
    "AnnotatedSequencePartV1",
    "AnnotatedSequenceSourceRefV1",
]
