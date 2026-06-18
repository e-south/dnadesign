"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/source_promotions/khan.py

Khan/Census source RT+ncRNA promotion for the RT-lnRNA study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

from .common import (
    ConstructWindowPolicy,
    construct_window_fit_issue,
    read_tsv,
    require_dna,
    require_no_internal_stop_codons,
    sha256_text,
    slug,
)
from .contracts import (
    SourceConstructSubjectPromotion,
    SourcePromotionContractError,
    SourcePromotionIssue,
)
from .source_catalog import (
    KHAN_ABUNDANCE_SOURCE_ID,
    KHAN_SEQUENCE_AUTHORITY_SOURCE_ID,
    SourceRecordResolver,
    resolve_source_table_path,
)

KHAN_COLLECTION_ID = "khan_cross_retron_rt_lnrna_reference_v1"
KHAN_PROMOTED_SOURCE_BASIS = "khan_abundance_affiliated_rt_lnrna_reference"
KHAN_REQUIRED_RT_CDS_VALIDATION_STATUS = "translation_exact_match"


def resolve_khan_promotions(
    *,
    data_root: Path,
    window_policy: ConstructWindowPolicy,
    source_row_counts: Counter[str],
    issues: list[SourcePromotionIssue],
    source_record_resolver: SourceRecordResolver | None = None,
) -> tuple[SourceConstructSubjectPromotion, ...]:
    rows = read_tsv(
        resolve_source_table_path(
            source_id=KHAN_SEQUENCE_AUTHORITY_SOURCE_ID,
            data_root=data_root,
            source_record_resolver=source_record_resolver,
        )
    )
    abundance_rows = read_tsv(
        resolve_source_table_path(
            source_id=KHAN_ABUNDANCE_SOURCE_ID,
            data_root=data_root,
            source_record_resolver=source_record_resolver,
        )
    )
    source_row_counts[KHAN_SEQUENCE_AUTHORITY_SOURCE_ID] = len(rows)
    source_row_counts[KHAN_ABUNDANCE_SOURCE_ID] = len(abundance_rows)
    abundance_rows_by_reference_id = _abundance_rows_by_reference_id(abundance_rows)
    promotions: list[SourceConstructSubjectPromotion] = []
    for row in rows:
        record_id = _khan_row_id(row)
        abundance_matches = abundance_rows_by_reference_id.get(_khan_abundance_reference_id(row), [])
        try:
            lnrna_sequence = require_dna(
                str(row.get("ncrna_sequence_dna") or row.get("lnrna_sequence") or ""),
                label=f"{record_id}.ncrna_sequence_dna",
            )
        except SourcePromotionContractError as exc:
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=KHAN_COLLECTION_ID,
                    source_record_id=record_id,
                    reason="invalid_lnrna_sequence",
                    detail=str(exc),
                )
            )
            continue

        if str(row.get("construct_projection_status") or "") != "representable":
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=KHAN_COLLECTION_ID,
                    source_record_id=record_id,
                    reason=_khan_blocked_reason(row),
                    detail=_khan_blocked_detail(row, record_id=record_id),
                )
            )
            continue
        rt_cds_sequence = str(row.get("rt_cds_sequence") or "").strip()
        try:
            rt_cds_sequence = require_dna(rt_cds_sequence, label=f"{record_id}.rt_cds_sequence")
            if len(rt_cds_sequence) % 3:
                raise SourcePromotionContractError(f"{record_id}.rt_cds_sequence length must be divisible by 3.")
            require_no_internal_stop_codons(rt_cds_sequence, label=f"{record_id}.rt_cds_sequence")
            if str(row.get("rt_cds_validation_status") or "").strip() != KHAN_REQUIRED_RT_CDS_VALIDATION_STATUS:
                raise SourcePromotionContractError(
                    f"{record_id}.rt_cds_sequence must have {KHAN_REQUIRED_RT_CDS_VALIDATION_STATUS} validation."
                )
        except SourcePromotionContractError as exc:
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=KHAN_COLLECTION_ID,
                    source_record_id=record_id,
                    reason="invalid_source_rt_cds_sequence",
                    detail=str(exc),
                )
            )
            continue

        fit_issue = construct_window_fit_issue(
            lnrna_sequence=lnrna_sequence,
            rt_cds_sequence=rt_cds_sequence,
            window_policy=window_policy,
        )
        if fit_issue:
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=KHAN_COLLECTION_ID,
                    source_record_id=record_id,
                    reason="source_sequence_exceeds_construct_window",
                    detail=fit_issue,
                )
            )
            continue
        if not abundance_matches:
            issues.append(
                SourcePromotionIssue(
                    source_collection_id=KHAN_COLLECTION_ID,
                    source_record_id=record_id,
                    reason="missing_affiliated_abundance_observation",
                    detail=(
                        f"{record_id}: Khan RT-lnRNA sequence authority row has no affiliated RT-DNA abundance prior."
                    ),
                )
            )
            continue

        sequence_sha = sha256_text(lnrna_sequence + "|" + rt_cds_sequence)
        rt_cds_authority_kind = str(row.get("rt_cds_sequence_authority") or "").strip()
        promotions.append(
            SourceConstructSubjectPromotion(
                construct_subject_id=f"rt_lnrna_pair__khan_source_rt__{slug(record_id)}__lnrna__source",
                lnrna_sequence=lnrna_sequence,
                rt_cds_sequence=rt_cds_sequence,
                source_basis=KHAN_PROMOTED_SOURCE_BASIS,
                source_collection_id=KHAN_COLLECTION_ID,
                source_record_id=record_id,
                source_record_count=1 + len(abundance_matches),
                source_lnrna_design_id=f"khan_terminal_{row.get('terminal_node') or row.get('mestre_number')}_ncrna",
                source_sequence_sha256=sequence_sha,
                lnrna_authority_kind="source_lnrna_sequence",
                rt_cds_authority_kind=rt_cds_authority_kind,
                overlay_fields=_khan_overlay_fields(
                    row=row,
                    record_id=record_id,
                    abundance_matches=abundance_matches,
                    sequence_sha=sequence_sha,
                    rt_cds_authority_kind=rt_cds_authority_kind,
                ),
            )
        )
    return tuple(promotions)


def _khan_overlay_fields(
    *,
    row: Mapping[str, str],
    record_id: str,
    abundance_matches: list[Mapping[str, str]],
    sequence_sha: str,
    rt_cds_authority_kind: str,
) -> dict[str, object]:
    source_lnrna_design_id = f"khan_terminal_{row.get('terminal_node') or row.get('mestre_number')}_ncrna"
    return {
        "construct_subject__role": "literature_rt_lnrna_reference",
        "construct_subject__variant_class": KHAN_PROMOTED_SOURCE_BASIS,
        "construct_subject__construct_projection_status": "representable",
        "construct_subject__source_basis": KHAN_PROMOTED_SOURCE_BASIS,
        "construct_subject__source_collection_id": KHAN_COLLECTION_ID,
        "construct_subject__source_record_id": record_id,
        "construct_subject__source_record_count": 1 + len(abundance_matches),
        "construct_subject__source_abundance_record_count": len(abundance_matches),
        "construct_subject__source_literature_id": "Khan_et_al_2024_retron_census",
        "construct_subject__source_label_kind": str(row.get("label_kind") or ""),
        "construct_subject__source_regime": str(row.get("regime") or ""),
        "construct_subject__source_lnrna_design_id": source_lnrna_design_id,
        "construct_subject__source_sequence_sha256": sequence_sha,
        "construct_subject__lnrna_authority_kind": "source_lnrna_sequence",
        "construct_subject__rt_cds_authority_kind": rt_cds_authority_kind,
        "construct_subject__rt_source": str(row.get("rt_source") or ""),
        "construct_subject__rt_variant": str(row.get("rt_variant") or ""),
        "construct_subject__rt_accession": str(row.get("rt_accession") or ""),
        "construct_subject__rt_cds_validation_status": str(row.get("rt_cds_validation_status") or ""),
        "construct_subject__rt_cds_locus_authority_id": str(row.get("rt_cds_locus_authority_id") or ""),
        "construct_subject__rt_cds_coordinate_authority": str(row.get("rt_cds_coordinate_authority") or ""),
        "construct_subject__rt_cds_coordinate_adjustment": str(row.get("rt_cds_coordinate_adjustment") or ""),
        "construct_subject__khan_abundance_observation_ids": join_values(abundance_matches, "abundance_prior_id"),
        "construct_subject__khan_abundance_raw_values": join_values(abundance_matches, "raw_value"),
        "construct_subject__khan_abundance_normalized_values": join_values(abundance_matches, "normalized_value"),
        "construct_subject__khan_abundance_ordinal_bins": join_values(abundance_matches, "ordinal_bin"),
    }


def _abundance_rows_by_reference_id(rows: list[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    by_reference_id: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        reference_id = str(row.get("maps_to_reference_record_id") or "").strip()
        if reference_id:
            by_reference_id.setdefault(reference_id, []).append(row)
    return by_reference_id


def _khan_abundance_reference_id(row: Mapping[str, str]) -> str:
    terminal = str(row.get("terminal_node") or row.get("mestre_number") or row.get("source_record_id") or "").strip()
    if terminal.startswith("khan_terminal_"):
        return terminal
    return f"khan_terminal_{terminal}"


def join_values(rows: list[Mapping[str, str]], field_name: str) -> str:
    return ";".join(str(row.get(field_name) or "").strip() for row in rows if str(row.get(field_name) or "").strip())


def _khan_row_id(row: Mapping[str, str]) -> str:
    return str(
        row.get("sequence_authority_id")
        or row.get("record_id")
        or row.get("source_record_id")
        or row.get("terminal_node")
        or "",
    ).strip()


def _khan_blocked_detail(row: Mapping[str, str], *, record_id: str) -> str:
    return (
        f"{record_id}: Khan row has rt_accession={row.get('rt_accession') or '<missing>'!s}, "
        f"rt_cds_sequence_status={row.get('rt_cds_sequence_status') or '<missing>'!s}, "
        f"ncrna_sequence_status={row.get('ncrna_sequence_status') or '<missing>'!s}, and "
        f"construct_projection_status={row.get('construct_projection_status') or '<missing>'!s}."
    )


def _khan_blocked_reason(row: Mapping[str, str]) -> str:
    rt_status = str(row.get("rt_cds_sequence_status") or "").strip()
    ncrna_status = str(row.get("ncrna_sequence_status") or "").strip()
    if rt_status in {"", "unresolved"}:
        return "missing_source_rt_cds_sequence"
    if rt_status != "resolved":
        return "invalid_source_rt_cds_sequence"
    if str(row.get("rt_cds_validation_status") or "").strip() != KHAN_REQUIRED_RT_CDS_VALIDATION_STATUS:
        return "invalid_source_rt_cds_sequence"
    if ncrna_status != "resolved":
        return "invalid_lnrna_sequence"
    return "khan_sequence_authority_not_construct_ready"
