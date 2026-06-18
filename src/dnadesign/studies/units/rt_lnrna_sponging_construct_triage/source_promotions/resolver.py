"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/source_promotions/resolver.py

Source-promotion orchestration for the RT-lnRNA study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
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
from .msd_compiler import reject_duplicate_msd_compiler_lnrna_sequences, resolve_msd_compiler_promotions
from .source_catalog import SourceRecordResolver


def resolve_source_construct_subject_promotions(
    *,
    dnadesign_data_root: Path,
    wt_rt_cds_sequence: str,
    window_policy: ConstructWindowPolicy,
    repo_root: Path | None = None,
    msd_variant_pool_spec_paths: tuple[Path, ...] = (),
    source_record_resolver: SourceRecordResolver | None = None,
) -> SourcePromotionReport:
    """Resolve literature sources into Construct-ready RT-lnRNA subjects.

    Crawford is an Eco1-local lnRNA/MSD source, so promoted rows use fixed WT
    Eco1 RT. Khan rows are promoted only when the source tables carry explicit
    ncRNA DNA, translation-exact RT CDS authority, and an affiliated RT-DNA
    abundance prior. Compiler-generated MSD pool specs are optional and explicit
    because they are study-owned sequence design references, not literature
    abundance priors.
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
            source_record_resolver=source_record_resolver,
        )
    )
    candidates.extend(
        resolve_khan_promotions(
            data_root=data_root,
            window_policy=window_policy,
            source_row_counts=source_row_counts,
            issues=issues,
            source_record_resolver=source_record_resolver,
        )
    )
    if msd_variant_pool_spec_paths and repo_root is None:
        raise SourcePromotionContractError("repo_root is required when resolving MSD compiler pool spec paths.")
    if repo_root is not None:
        root = Path(repo_root).resolve()
        for spec_path in msd_variant_pool_spec_paths:
            promotions = resolve_msd_compiler_promotions(
                repo_root=root,
                pool_spec_path=spec_path,
                wt_rt_cds_sequence=wt_rt,
                window_policy=window_policy,
            )
            source_row_counts[str(Path(spec_path))] = len(promotions)
            candidates.extend(promotions)

    duplicate_ids = duplicates(candidate.construct_subject_id for candidate in candidates)
    if duplicate_ids:
        raise SourcePromotionContractError("Duplicate promoted construct subject id(s): " + ", ".join(duplicate_ids))
    reject_duplicate_msd_compiler_lnrna_sequences(candidates)
    return SourcePromotionReport(
        candidates=tuple(sorted(candidates, key=lambda candidate: candidate.construct_subject_id)),
        issues=tuple(issues),
        source_row_counts=dict(source_row_counts),
    )
