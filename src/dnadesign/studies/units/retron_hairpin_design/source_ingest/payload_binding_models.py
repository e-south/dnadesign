"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/payload_binding_models.py

Payload binding-site catalog models.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MotifModel:
    motif_model_id: str
    source_ref: str
    matrix: tuple[tuple[float, float, float, float], ...]
    congruence_threshold_fraction: float

    @property
    def width(self) -> int:
        return len(self.matrix)

    @property
    def consensus(self) -> str:
        bases = "ACGT"
        return "".join(bases[max(range(4), key=lambda index, row=row: row[index])] for row in self.matrix)


@dataclass(frozen=True, slots=True)
class PayloadMember:
    family_id: str
    parent_payload_id: str
    member_id: str
    primary_sequence_5to3: str
    complement_sequence_5to3: str
    retained_parent_span_0: dict[str, int]
    motif_model_id: str | None
    parent_primary_sequence_5to3: str

    @property
    def is_parent(self) -> bool:
        span = self.retained_parent_span_0
        return self.member_id == self.parent_payload_id and span == {
            "start": 0,
            "end": len(self.parent_primary_sequence_5to3),
        }


@dataclass(frozen=True, slots=True)
class PayloadBindingCatalog:
    motif_models: Mapping[str, MotifModel]
    members_by_id: Mapping[str, PayloadMember]
    members_by_primary_sequence: Mapping[str, PayloadMember]
    reference_payload_ids: tuple[str, ...]
    default_motif_model_id: str | None

    def member_for_primary_sequence(self, sequence: str) -> PayloadMember | None:
        from .payload_binding_utils import normalize_dna

        return self.members_by_primary_sequence.get(normalize_dna(sequence))

    def reference_members(self) -> tuple[PayloadMember, ...]:
        return tuple(self.members_by_id[payload_id] for payload_id in self.reference_payload_ids)


__all__ = ["MotifModel", "PayloadBindingCatalog", "PayloadMember"]
