"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/presentation/sequence_review/contract.py

Typed Junction sequence evidence for one assembly group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.contracts.visual.common import VisualContractModel
from dnadesign.contracts.visual.three_way_junction_review_v1 import (
    JUNCTION_STRING_V1_ALGORITHM,
    ReviewSource,
)

_DNA = re.compile(r"^[ACGT]+$")


def _require_dna(value: str, *, field: str) -> str:
    if not _DNA.fullmatch(value):
        raise ValueError(f"{field} must be non-empty uppercase ACGT")
    return value


class JunctionSequenceChoice(VisualContractModel):
    junction_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    toehold_sequence_5to3: str
    barcode_sequence_5to3: str

    @field_validator("toehold_sequence_5to3", "barcode_sequence_5to3")
    @classmethod
    def _validate_sequence(cls, value: str, info) -> str:
        return _require_dna(value, field=f"junction.{info.field_name}")


class JunctionSequenceDissimilarityV1(VisualContractModel):
    """Junction strings selected for one assembly-group review."""

    contract_kind: Literal["junction_sequence_dissimilarity_v1"]
    source: ReviewSource
    assembly_group_id: str = Field(min_length=1)
    junctions: tuple[JunctionSequenceChoice, ...] = Field(min_length=1)
    thermodynamic_screening: Literal["not_run"]

    @model_validator(mode="after")
    def _validate_group(self) -> "JunctionSequenceDissimilarityV1":
        if self.source.algorithm != JUNCTION_STRING_V1_ALGORITHM:
            raise ValueError(f"source.algorithm must be {JUNCTION_STRING_V1_ALGORITHM!r}")
        junction_ids = [junction.junction_id for junction in self.junctions]
        if len(junction_ids) != len(set(junction_ids)):
            raise ValueError("junction_id values must be unique within one assembly group")
        barcodes = [junction.barcode_sequence_5to3 for junction in self.junctions]
        if len(barcodes) != len(set(barcodes)):
            raise ValueError("barcode sequences must be unique within one assembly group")
        if len({len(junction.toehold_sequence_5to3) for junction in self.junctions}) != 1:
            raise ValueError("toehold sequences must use one length within one assembly group")
        if len({len(junction.barcode_sequence_5to3) for junction in self.junctions}) != 1:
            raise ValueError("barcode sequences must use one length within one assembly group")
        return self


__all__ = ["JunctionSequenceChoice", "JunctionSequenceDissimilarityV1"]
