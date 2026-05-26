"""
Crawford Eco1 ncRNA source promotion for the RT-lnRNA study.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .common import (
    ConstructWindowPolicy,
    construct_window_fit_issue,
    format_ratio,
    format_span,
    join_row_ids,
    join_values,
    read_tsv,
    require_dna,
    reverse_complement,
    row_id,
    sha256_text,
)
from .contracts import (
    SourceConstructSubjectPromotion,
    SourcePromotionContractError,
    SourcePromotionIssue,
)
from .source_catalog import (
    CRAWFORD_ABUNDANCE_SOURCE_ID,
    CRAWFORD_REFERENCE_SOURCE_ID,
    SourceRecordResolver,
    resolve_source_table_path,
)

CRAWFORD_PROMOTED_COLLECTION_ID = "crawford_eco1_lnrna_sequence_union_v1"
CRAWFORD_REFERENCE_COLLECTION_ID = "crawford_eco1_lnrna_msd_designs_v1"
CRAWFORD_ABUNDANCE_COLLECTION_ID = "crawford_eco1_lnrna_msd_abundance_v1"
CRAWFORD_SEQUENCE_QC_POLICY = "eco1_forward_kmer_similarity_v1"
CRAWFORD_CONTEXT_NOTE = "source_lnrna_sequence_projected_into_dnadesign_dual_cassette_not_native_expression_context"
CRAWFORD_A1A2_NOTE = "source_a1a2_context_not_assumed_to_match_dnadesign_a1a2_20"
CRAWFORD_KMER_SIZE = 8
CRAWFORD_MIN_FORWARD_KMER_IDENTITY = 0.65
CRAWFORD_REVERSE_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class CrawfordSequenceAudit:
    forward_kmer_identity: float
    reverse_kmer_identity: float
    msd_anchor_status: str
    msd_span_0: tuple[int, int] | None


def resolve_crawford_promotions(
    *,
    data_root: Path,
    wt_rt_cds_sequence: str,
    window_policy: ConstructWindowPolicy,
    source_row_counts: Counter[str],
    issues: list[SourcePromotionIssue],
    source_record_resolver: SourceRecordResolver | None = None,
) -> tuple[SourceConstructSubjectPromotion, ...]:
    reference_rows = read_tsv(
        resolve_source_table_path(
            source_id=CRAWFORD_REFERENCE_SOURCE_ID,
            data_root=data_root,
            source_record_resolver=source_record_resolver,
        )
    )
    abundance_rows = read_tsv(
        resolve_source_table_path(
            source_id=CRAWFORD_ABUNDANCE_SOURCE_ID,
            data_root=data_root,
            source_record_resolver=source_record_resolver,
        )
    )
    source_row_counts[CRAWFORD_REFERENCE_SOURCE_ID] = len(reference_rows)
    source_row_counts[CRAWFORD_ABUNDANCE_SOURCE_ID] = len(abundance_rows)
    template_kmers = _crawford_reference_kmers(reference_rows)

    abundance_rows_by_sequence: dict[str, list[Mapping[str, str]]] = {}
    for row in abundance_rows:
        record_id = row_id(row)
        raw_lnrna = str(row.get("lnrna_sequence") or "").strip()
        try:
            lnrna_sequence = require_dna(raw_lnrna, label=f"{record_id}.lnrna_sequence")
        except SourcePromotionContractError as exc:
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=CRAWFORD_ABUNDANCE_COLLECTION_ID,
                    source_record_id=record_id,
                    reason="invalid_abundance_lnrna_sequence",
                    detail=str(exc),
                )
            )
            continue
        abundance_rows_by_sequence.setdefault(lnrna_sequence, []).append(row)

    reference_rows_by_sequence: dict[str, list[Mapping[str, str]]] = {}
    for row in reference_rows:
        record_id = row_id(row)
        raw_lnrna = str(row.get("lnrna_sequence") or "").strip()
        try:
            lnrna_sequence = require_dna(raw_lnrna, label=f"{record_id}.lnrna_sequence")
        except SourcePromotionContractError as exc:
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=CRAWFORD_REFERENCE_COLLECTION_ID,
                    source_record_id=record_id,
                    reason="invalid_reference_lnrna_sequence",
                    detail=str(exc),
                )
            )
            continue
        reference_rows_by_sequence.setdefault(lnrna_sequence, []).append(row)

    promotions: list[SourceConstructSubjectPromotion] = []
    all_sequences = sorted(set(reference_rows_by_sequence) | set(abundance_rows_by_sequence))
    for lnrna_sequence in all_sequences:
        rows = reference_rows_by_sequence.get(lnrna_sequence, [])
        abundance_matches = abundance_rows_by_sequence.get(lnrna_sequence, [])
        source_rows = rows or abundance_matches
        if not source_rows:
            continue
        first = source_rows[0]
        try:
            sequence_audit = _audit_crawford_sequence_orientation(
                record_id=row_id(first),
                lnrna_sequence=lnrna_sequence,
                reference_rows=rows,
                template_kmers=template_kmers,
            )
        except SourcePromotionContractError as exc:
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=_crawford_source_collection_id(rows=rows, abundance_matches=abundance_matches),
                    source_record_id=row_id(first),
                    reason=_crawford_sequence_issue_reason(str(exc)),
                    detail=str(exc),
                )
            )
            continue
        source_record_count = len(rows) + len(abundance_matches)
        fit_issue = construct_window_fit_issue(
            lnrna_sequence=lnrna_sequence,
            rt_cds_sequence=wt_rt_cds_sequence,
            window_policy=window_policy,
        )
        if fit_issue:
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=_crawford_source_collection_id(rows=rows, abundance_matches=abundance_matches),
                    source_record_id=row_id(first),
                    reason="source_sequence_exceeds_construct_window",
                    detail=fit_issue,
                )
            )
            continue
        sequence_sha = sha256_text(lnrna_sequence)
        promotions.append(
            SourceConstructSubjectPromotion(
                construct_subject_id=f"rt_lnrna_pair__eco1_wt_rt__crawford_lnrna_{sequence_sha[:12]}__tetO",
                lnrna_sequence=lnrna_sequence,
                rt_cds_sequence=wt_rt_cds_sequence,
                source_basis="crawford_eco1_lnrna_fixed_wt_rt",
                source_collection_id=CRAWFORD_PROMOTED_COLLECTION_ID,
                source_record_id=row_id(first),
                source_record_count=source_record_count,
                source_lnrna_design_id=_crawford_lnrna_design_id(rows=rows, abundance_matches=abundance_matches),
                source_sequence_sha256=sequence_sha,
                lnrna_authority_kind=_crawford_lnrna_authority_kind(rows=rows, abundance_matches=abundance_matches),
                rt_cds_authority_kind="fixed_eco1_wt_rt",
                overlay_fields=_crawford_overlay_fields(
                    first=first,
                    rows=rows,
                    abundance_matches=abundance_matches,
                    source_record_count=source_record_count,
                    sequence_sha=sequence_sha,
                    sequence_audit=sequence_audit,
                ),
            )
        )
    return tuple(promotions)


def _crawford_overlay_fields(
    *,
    first: Mapping[str, str],
    rows: list[Mapping[str, str]],
    abundance_matches: list[Mapping[str, str]],
    source_record_count: int,
    sequence_sha: str,
    sequence_audit: CrawfordSequenceAudit,
) -> dict[str, object]:
    return {
        "construct_subject__role": "literature_lnrna_reference",
        "construct_subject__variant_class": "crawford_eco1_lnrna_variant",
        "construct_subject__construct_projection_status": "representable",
        "construct_subject__source_basis": "crawford_eco1_lnrna_fixed_wt_rt",
        "construct_subject__source_collection_id": CRAWFORD_PROMOTED_COLLECTION_ID,
        "construct_subject__source_record_id": row_id(first),
        "construct_subject__source_record_count": source_record_count,
        "construct_subject__source_reference_record_count": len(rows),
        "construct_subject__source_abundance_record_count": len(abundance_matches),
        "construct_subject__source_literature_id": "Crawford_et_al_2025_retron_ncRNA_ML",
        "construct_subject__source_label_kind": str(first.get("label_kind") or ""),
        "construct_subject__source_regime": str(first.get("regime") or ""),
        "construct_subject__source_lnrna_design_id": _crawford_lnrna_design_id(
            rows=rows,
            abundance_matches=abundance_matches,
        ),
        "construct_subject__source_sequence_sha256": sequence_sha,
        "construct_subject__lnrna_authority_kind": _crawford_lnrna_authority_kind(
            rows=rows,
            abundance_matches=abundance_matches,
        ),
        "construct_subject__rt_cds_authority_kind": "fixed_eco1_wt_rt",
        "construct_subject__rt_source": "Retron-Eco1",
        "construct_subject__rt_variant": "WT",
        "construct_subject__source_orientation": "forward",
        "construct_subject__crawford_sequence_qc_policy": CRAWFORD_SEQUENCE_QC_POLICY,
        "construct_subject__crawford_forward_kmer_identity": format_ratio(sequence_audit.forward_kmer_identity),
        "construct_subject__crawford_reverse_kmer_identity": format_ratio(sequence_audit.reverse_kmer_identity),
        "construct_subject__crawford_msd_anchor_status": sequence_audit.msd_anchor_status,
        "construct_subject__crawford_msd_span_0": format_span(sequence_audit.msd_span_0),
        "construct_subject__crawford_reference_record_ids": join_row_ids(rows),
        "construct_subject__crawford_abundance_observation_ids": join_row_ids(abundance_matches),
        "construct_subject__crawford_abundance_raw_values": join_values(abundance_matches, "raw_value"),
        "construct_subject__crawford_source_context_note": CRAWFORD_CONTEXT_NOTE,
        "construct_subject__crawford_a1a2_context_note": CRAWFORD_A1A2_NOTE,
    }


def _audit_crawford_sequence_orientation(
    *,
    record_id: str,
    lnrna_sequence: str,
    reference_rows: list[Mapping[str, str]],
    template_kmers: frozenset[str],
) -> CrawfordSequenceAudit:
    sequence_kmers = _kmers(lnrna_sequence)
    if not sequence_kmers:
        raise SourcePromotionContractError(f"{record_id}: Crawford lnRNA sequence is too short for k-mer QC.")
    forward_score = _kmer_identity(sequence_kmers, template_kmers)
    reverse_score = _kmer_identity(_kmers(reverse_complement(lnrna_sequence)), template_kmers)
    if reverse_score > forward_score + CRAWFORD_REVERSE_MARGIN:
        raise SourcePromotionContractError(
            f"{record_id}: Crawford lnRNA sequence appears reverse-complemented by Eco1 k-mer orientation QC."
        )
    if forward_score < CRAWFORD_MIN_FORWARD_KMER_IDENTITY:
        raise SourcePromotionContractError(
            f"{record_id}: Crawford lnRNA sequence has low Eco1 forward k-mer identity "
            f"({forward_score:.3f} < {CRAWFORD_MIN_FORWARD_KMER_IDENTITY:.3f})."
        )
    msd_status, span = _crawford_msd_anchor_status(lnrna_sequence=lnrna_sequence, reference_rows=reference_rows)
    return CrawfordSequenceAudit(
        forward_kmer_identity=forward_score,
        reverse_kmer_identity=reverse_score,
        msd_anchor_status=msd_status,
        msd_span_0=span,
    )


def _crawford_sequence_issue_reason(detail: str) -> str:
    if "reverse-complemented" in detail or "orientation" in detail:
        return "reverse_complement_lnrna_orientation"
    if "low Eco1 forward k-mer identity" in detail:
        return "low_eco1_lnrna_similarity"
    return "invalid_crawford_sequence_qc"


def _crawford_reference_kmers(reference_rows: tuple[Mapping[str, str], ...]) -> frozenset[str]:
    template_sequences = {
        require_dna(str(row.get("lnrna_sequence") or ""), label=f"{row_id(row)}.lnrna_sequence")
        for row in reference_rows
        if "_wt" in str(row.get("lnrna_design_id") or "").lower()
    }
    if not template_sequences:
        raise SourcePromotionContractError("Crawford reference table has no WT lnRNA template rows for orientation QC.")
    return frozenset(kmer for sequence in template_sequences for kmer in _kmers(sequence))


def _kmers(sequence: str) -> frozenset[str]:
    if len(sequence) < CRAWFORD_KMER_SIZE:
        return frozenset()
    return frozenset(
        sequence[index : index + CRAWFORD_KMER_SIZE] for index in range(0, len(sequence) - CRAWFORD_KMER_SIZE + 1)
    )


def _kmer_identity(sequence_kmers: frozenset[str], template_kmers: frozenset[str]) -> float:
    if not sequence_kmers:
        return 0.0
    return len(sequence_kmers & template_kmers) / len(sequence_kmers)


def _crawford_msd_anchor_status(
    *,
    lnrna_sequence: str,
    reference_rows: list[Mapping[str, str]],
) -> tuple[str, tuple[int, int] | None]:
    if not reference_rows:
        return "not_declared_for_abundance_only_sequence", None
    msd_sequences = {
        require_dna(str(row.get("msd_sequence") or ""), label=f"{row_id(row)}.msd_sequence")
        for row in reference_rows
        if str(row.get("msd_sequence") or "").strip()
    }
    if not msd_sequences:
        return "not_declared_in_reference_row", None
    spans = []
    for msd_sequence in msd_sequences:
        start = lnrna_sequence.find(msd_sequence)
        if start >= 0:
            spans.append((start, start + len(msd_sequence)))
    if spans:
        return "exact_declared_msd_substring", sorted(spans)[0]
    return "no_exact_msd_substring_expected_for_source_variant", None


def _crawford_lnrna_authority_kind(
    *,
    rows: list[Mapping[str, str]],
    abundance_matches: list[Mapping[str, str]],
) -> str:
    if rows and abundance_matches:
        return "source_design_reference_and_abundance_sequence"
    if rows:
        return "source_design_reference_sequence"
    return "source_abundance_observation_sequence"


def _crawford_lnrna_design_id(
    *,
    rows: list[Mapping[str, str]],
    abundance_matches: list[Mapping[str, str]],
) -> str:
    for row in (*rows, *abundance_matches):
        value = str(row.get("lnrna_design_id") or row.get("source_part_name") or row.get("prefix") or "").strip()
        if value:
            return value
    return ""


def _crawford_source_collection_id(
    *,
    rows: list[Mapping[str, str]],
    abundance_matches: list[Mapping[str, str]],
) -> str:
    if rows and abundance_matches:
        return CRAWFORD_PROMOTED_COLLECTION_ID
    if rows:
        return CRAWFORD_REFERENCE_COLLECTION_ID
    return CRAWFORD_ABUNDANCE_COLLECTION_ID
