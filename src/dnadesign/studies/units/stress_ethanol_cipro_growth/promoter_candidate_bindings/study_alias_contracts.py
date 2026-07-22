"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/study_alias_contracts.py

Typed contracts for stable stress-study promoter aliases.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .contracts import PromoterCandidateBindingsError
from .values import required_text


def sequence_sha256(sequence: object) -> str:
    text = required_text(sequence, field="candidate sequence").upper()
    if re.fullmatch(r"[ACGTN]+", text) is None:
        raise PromoterCandidateBindingsError("Candidate sequence contains characters outside A/C/G/T/N.")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AliasFormat:
    prefix: str
    zero_pad_width: int

    def render(self, ordinal: int) -> str:
        return f"{self.prefix}-{int(ordinal):0{self.zero_pad_width}d}"


@dataclass(frozen=True)
class AliasFirstAssignment:
    source_authority: str
    source_id: str
    nomination_batch_index: int | None
    model_as_of_round: int | None


@dataclass(frozen=True)
class StudyPromoterAlias:
    ordinal: int
    alias: str
    candidate_id: str
    sequence_sha256: str
    first_assignment: AliasFirstAssignment
    source_aliases: tuple[str, ...]


@dataclass(frozen=True)
class PlannedStudyAlias:
    candidate_id: str
    sequence_sha256: str
    alias: str
    ordinal: int
    is_new: bool


@dataclass(frozen=True)
class StudyPromoterAliasRegistry:
    path: Path
    candidate_table_dataset_id: str
    candidate_table_records_path: Path
    alias_format: AliasFormat
    assignments: tuple[StudyPromoterAlias, ...]

    @property
    def next_ordinal(self) -> int:
        return len(self.assignments) + 1

    def alias_for(self, *, candidate_id: str, sequence: str) -> str:
        identifier = required_text(candidate_id, field="candidate ID")
        digest = sequence_sha256(sequence)
        by_id = {row.candidate_id: row for row in self.assignments}
        row = by_id.get(identifier)
        if row is None:
            raise PromoterCandidateBindingsError(
                f"Candidate {identifier!r} has no assigned study promoter alias; append it to {self.path}."
            )
        if row.sequence_sha256 != digest:
            raise PromoterCandidateBindingsError(
                f"Candidate {identifier!r} sequence does not match assigned study promoter alias {row.alias}."
            )
        return row.alias


__all__ = [
    "AliasFirstAssignment",
    "AliasFormat",
    "PlannedStudyAlias",
    "StudyPromoterAlias",
    "StudyPromoterAliasRegistry",
    "sequence_sha256",
]
