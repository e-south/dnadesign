"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/source_promotions/__init__.py

Public study-owned source-promotion facade for RT-lnRNA construct subjects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .common import ConstructWindowPolicy
from .contracts import (
    SourceConstructSubjectPromotion,
    SourcePromotionContractError,
    SourcePromotionIssue,
    SourcePromotionReport,
)
from .msd_compiler import reject_duplicate_msd_compiler_lnrna_sequences, resolve_msd_compiler_promotions
from .resolver import resolve_source_construct_subject_promotions
from .source_catalog import SourceRecordResolver

__all__ = [
    "ConstructWindowPolicy",
    "SourceRecordResolver",
    "SourceConstructSubjectPromotion",
    "SourcePromotionContractError",
    "SourcePromotionIssue",
    "SourcePromotionReport",
    "reject_duplicate_msd_compiler_lnrna_sequences",
    "resolve_msd_compiler_promotions",
    "resolve_source_construct_subject_promotions",
]
