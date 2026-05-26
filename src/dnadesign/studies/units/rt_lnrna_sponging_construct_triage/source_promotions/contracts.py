"""
Study-owned source-promotion contracts for RT-lnRNA construct subjects.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping


class SourcePromotionContractError(ValueError):
    """Raised when study source promotion inputs cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class SourcePromotionIssue:
    source_collection_id: str
    source_record_id: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class SourceConstructSubjectPromotion:
    construct_subject_id: str
    lnrna_sequence: str
    rt_cds_sequence: str
    source_basis: str
    source_collection_id: str
    source_record_id: str
    source_record_count: int
    source_lnrna_design_id: str
    source_sequence_sha256: str
    lnrna_authority_kind: str
    rt_cds_authority_kind: str
    overlay_fields: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SourcePromotionReport:
    candidates: tuple[SourceConstructSubjectPromotion, ...]
    issues: tuple[SourcePromotionIssue, ...]
    source_row_counts: Mapping[str, int]

    @property
    def candidates_by_basis(self) -> Mapping[str, int]:
        return Counter(candidate.source_basis for candidate in self.candidates)

    @property
    def issues_by_reason(self) -> Mapping[str, int]:
        return Counter(issue.reason for issue in self.issues)
