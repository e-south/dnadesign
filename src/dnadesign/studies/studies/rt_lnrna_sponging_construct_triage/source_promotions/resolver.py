"""
Source-promotion orchestration for the RT-lnRNA study.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import ConstructWindowPolicy, duplicates, require_dna, require_no_internal_stop_codons
from .contracts import (
    SourceConstructSubjectPromotion,
    SourcePromotionContractError,
    SourcePromotionIssue,
    SourcePromotionReport,
)
from .crawford import resolve_crawford_promotions
from .khan import resolve_khan_promotions


def resolve_source_construct_subject_promotions(
    *,
    dnadesign_data_root: Path,
    wt_rt_cds_sequence: str,
    window_policy: ConstructWindowPolicy,
) -> SourcePromotionReport:
    """Resolve literature sources into Construct-ready RT-lnRNA subjects.

    Crawford is an Eco1-local lnRNA/MSD source, so promoted rows use fixed WT
    Eco1 RT. Khan rows are promoted only when the source table carries explicit
    ncRNA DNA and translation-exact RT CDS authority.
    """

    data_root = Path(dnadesign_data_root).resolve()
    wt_rt = require_dna(wt_rt_cds_sequence, label="wt_rt_cds_sequence")
    if len(wt_rt) % 3:
        raise SourcePromotionContractError("WT RT CDS sequence length must be divisible by 3.")
    require_no_internal_stop_codons(wt_rt, label="wt_rt_cds_sequence")

    candidates: list[SourceConstructSubjectPromotion] = []
    issues: list[SourcePromotionIssue] = []
    source_row_counts: Counter[str] = Counter()

    candidates.extend(
        resolve_crawford_promotions(
            data_root=data_root,
            wt_rt_cds_sequence=wt_rt,
            window_policy=window_policy,
            source_row_counts=source_row_counts,
            issues=issues,
        )
    )
    candidates.extend(
        resolve_khan_promotions(
            data_root=data_root,
            window_policy=window_policy,
            source_row_counts=source_row_counts,
            issues=issues,
        )
    )

    duplicate_ids = duplicates(candidate.construct_subject_id for candidate in candidates)
    if duplicate_ids:
        raise SourcePromotionContractError("Duplicate promoted construct subject id(s): " + ", ".join(duplicate_ids))
    return SourcePromotionReport(
        candidates=tuple(sorted(candidates, key=lambda candidate: candidate.construct_subject_id)),
        issues=tuple(issues),
        source_row_counts=dict(source_row_counts),
    )
