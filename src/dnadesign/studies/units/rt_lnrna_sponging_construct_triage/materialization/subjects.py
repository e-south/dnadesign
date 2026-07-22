"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/materialization/subjects.py

Construct-subject row builders for RT-lnRNA materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dnadesign.permuter import CodingDnaDmsVariantMetadata, PermuterResult
from dnadesign.usr import BiopythonGenBankParser

from ..genbank_authority import GenBankAuthorityAudit
from ..source_promotions import SourceConstructSubjectPromotion
from ..variant_genbank_catalog import ExtractedSequenceAuthority, VariantGenBankCatalogRecord
from .common import _list, _mapping, _span_0
from .contracts import (
    _BASE_TEMPLATE_LNRNA_SPAN_0,
    _CONSTRUCT_SUBJECT_BIOLOGICAL_SEQUENCE_FIELDS,
    _GENBANK_CATALOG_SOURCE_BASIS,
    _GENBANK_CATALOG_SOURCE_COLLECTION_ID,
    _MATERIALIZATION_SOURCE,
    _SEQUENCE_ID_SOURCE_MAP,
    MaterializationContractError,
    _CatalogMaterializationCandidate,
)
from .views import _candidate_window_bounds


def _candidate_rows(
    *,
    manifest: dict[str, object],
    authority: GenBankAuthorityAudit,
    template_sequence: str,
    target_start: int,
    target_end: int,
    construct_subject_sequence_overrides: Mapping[str, Mapping[str, str]],
    omitted_construct_subject_fields: set[str],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    candidates = tuple(
        _mapping(candidate, label="candidates[]") for candidate in _list(manifest["candidates"], label="candidates")
    )
    rows: list[dict[str, object]] = []
    expected_sequences: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        construct_subject_id = str(candidate["construct_subject_id"])
        slot_bindings = _mapping(candidate["slot_bindings"], label=f"{construct_subject_id}.slot_bindings")
        row: dict[str, object] = {
            "id": construct_subject_id,
            # USR base row ids stay canonical sequence ids; construct-subject
            # identity travels through the study overlay and usr_label namespace.
            "sequence": "A" * (index + 1),
            "source": _MATERIALIZATION_SOURCE,
            **_construct_subject_envelope_overlay(),
        }
        for slot in slots:
            slot_id = str(slot["slot_id"])
            field_name = str(slot["sequence_field"])
            binding = _mapping(slot_bindings[slot_id], label=f"{construct_subject_id}.slot_bindings.{slot_id}")
            sequence = _sequence_for_binding(binding=binding, authority=authority)
            sequence = construct_subject_sequence_overrides.get(construct_subject_id, {}).get(field_name, sequence)
            expected_length = int(binding["sequence_length_nt"])
            if len(sequence) != expected_length:
                raise MaterializationContractError(
                    f"{construct_subject_id}: {field_name} length {len(sequence)} does not match "
                    f"declared {slot_id} length {expected_length}."
                )
            row[field_name] = None if field_name in omitted_construct_subject_fields else sequence
        rows.append(row)
        expected_sequences[construct_subject_id] = _expected_context_sequence(
            template_sequence=template_sequence,
            slots=slots,
            row=row,
            target_start=target_start,
            target_end=target_end,
        )
    return rows, expected_sequences


def _catalog_materialization_candidates(
    *,
    repo_root: Path,
    catalog_genbank_dir: Path,
    records: tuple[VariantGenBankCatalogRecord, ...],
    target_start: int,
    target_end: int,
) -> tuple[_CatalogMaterializationCandidate, ...]:
    parser = BiopythonGenBankParser()
    window_length = target_end - target_start
    candidates: list[_CatalogMaterializationCandidate] = []
    for record in records:
        if record.construct_projection_status != "representable":
            continue
        lnrna_sequence = _catalog_authority_sequence(
            repo_root=repo_root,
            genbank_dir=catalog_genbank_dir,
            parser=parser,
            authority=record.lnrna,
        )
        rt_cds_sequence = _catalog_authority_sequence(
            repo_root=repo_root,
            genbank_dir=catalog_genbank_dir,
            parser=parser,
            authority=record.rt_cds,
        )
        if len(lnrna_sequence) != record.lnrna.length_nt:
            raise MaterializationContractError(
                f"{record.variant_id}: lnRNA catalog span length does not match extracted sequence."
            )
        if len(rt_cds_sequence) != record.rt_cds.length_nt:
            raise MaterializationContractError(
                f"{record.variant_id}: RT CDS catalog span length does not match extracted sequence."
            )
        window_start = _BASE_TEMPLATE_LNRNA_SPAN_0[0] - int(record.construct_spans_0["lnrna"][0])
        lnrna_center = _BASE_TEMPLATE_LNRNA_SPAN_0[0] + (record.lnrna.length_nt // 2)
        window_offset_bp = window_start - (lnrna_center - (window_length // 2))
        candidates.append(
            _CatalogMaterializationCandidate(
                construct_subject_id=record.construct_subject_id,
                lnrna_sequence=lnrna_sequence,
                rt_cds_sequence=rt_cds_sequence,
                window_start=window_start,
                window_offset_bp=window_offset_bp,
                source_variant_id=record.variant_id,
                source_variant_class=record.variant_class,
                reader_design_id=record.reader_design_id,
                lnrna_authority_kind=record.lnrna.authority_kind,
                rt_cds_authority_kind=record.rt_cds.authority_kind,
            )
        )
    return tuple(candidates)


def _catalog_candidate_rows(
    *,
    manifest: dict[str, object],
    template_sequence: str,
    target_start: int,
    target_end: int,
    candidates: tuple[_CatalogMaterializationCandidate, ...],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    window_length = target_end - target_start
    rows: list[dict[str, object]] = []
    expected_sequences: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        row: dict[str, object] = {
            "id": candidate.construct_subject_id,
            "sequence": "A" * (index + 1),
            "source": _MATERIALIZATION_SOURCE,
            **_construct_subject_envelope_overlay(),
            "construct_subject__lnrna_sequence": candidate.lnrna_sequence,
            "construct_subject__rt_cds_sequence": candidate.rt_cds_sequence,
            "construct_subject__source_basis": _GENBANK_CATALOG_SOURCE_BASIS,
            "construct_subject__source_collection_id": _GENBANK_CATALOG_SOURCE_COLLECTION_ID,
            "construct_subject__source_variant_id": candidate.source_variant_id,
            "construct_subject__variant_class": candidate.source_variant_class,
            "construct_subject__reader_design_id": candidate.reader_design_id,
            "construct_subject__lnrna_authority_kind": candidate.lnrna_authority_kind,
            "construct_subject__rt_cds_authority_kind": candidate.rt_cds_authority_kind,
            "construct_subject__construct_projection_status": "representable",
            "construct_subject__role": "construct_subject",
        }
        rows.append(row)
        expected_sequences[candidate.construct_subject_id] = _expected_context_sequence_at_window(
            template_sequence=template_sequence,
            slots=slots,
            row=row,
            window_start=candidate.window_start,
            window_end=candidate.window_start + window_length,
        )
    return rows, expected_sequences


def _source_promotion_rows(
    *,
    manifest: dict[str, object],
    template_sequence: str,
    target_start: int,
    target_end: int,
    promotions: tuple[SourceConstructSubjectPromotion, ...],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    rows: list[dict[str, object]] = []
    expected_sequences: dict[str, str] = {}
    for index, promotion in enumerate(promotions, start=1):
        row: dict[str, object] = {
            "id": promotion.construct_subject_id,
            "sequence": "A" * index,
            "source": _MATERIALIZATION_SOURCE,
            **_construct_subject_envelope_overlay(),
            "construct_subject__lnrna_sequence": promotion.lnrna_sequence,
            "construct_subject__rt_cds_sequence": promotion.rt_cds_sequence,
            "construct_subject__source_basis": promotion.source_basis,
            "construct_subject__source_collection_id": promotion.source_collection_id,
            "construct_subject__source_record_id": promotion.source_record_id,
            "construct_subject__source_record_count": promotion.source_record_count,
            "construct_subject__source_lnrna_design_id": promotion.source_lnrna_design_id,
            "construct_subject__source_sequence_sha256": promotion.source_sequence_sha256,
            "construct_subject__lnrna_authority_kind": promotion.lnrna_authority_kind,
            "construct_subject__rt_cds_authority_kind": promotion.rt_cds_authority_kind,
            **dict(promotion.overlay_fields),
        }
        rows.append(row)
        expected_sequences[promotion.construct_subject_id] = _expected_context_sequence(
            template_sequence=template_sequence,
            slots=slots,
            row=row,
            target_start=target_start,
            target_end=target_end,
        )
    return rows, expected_sequences


def _group_source_promotion_rows_by_basis(
    rows: list[dict[str, object]],
) -> dict[str, tuple[dict[str, object], ...]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        basis = str(row.get("construct_subject__source_basis") or "")
        if not basis:
            raise MaterializationContractError(f"Promoted source row {row.get('id')} is missing source_basis.")
        grouped.setdefault(basis, []).append(row)
    return {basis: tuple(group) for basis, group in sorted(grouped.items())}


def _extend_construct_subject_rows(target: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    existing_ids = {str(row.get("id")) for row in target}
    incoming_ids = [str(row.get("id")) for row in rows]
    duplicate_existing = sorted(set(incoming_ids) & existing_ids)
    if duplicate_existing:
        raise MaterializationContractError(
            "Duplicate construct subject id already selected: " + ", ".join(duplicate_existing)
        )
    duplicate_incoming = _duplicates(incoming_ids)
    if duplicate_incoming:
        raise MaterializationContractError(
            "Duplicate construct subject id in selected source rows: " + ", ".join(duplicate_incoming)
        )
    target.extend(rows)


def _construct_subject_row_by_id(rows: list[dict[str, object]], *, construct_subject_id: str) -> dict[str, object]:
    matches = [row for row in rows if str(row.get("id")) == construct_subject_id]
    if not matches:
        raise MaterializationContractError(
            f"Base construct subject is absent from selected sequence authority: {construct_subject_id}"
        )
    if len(matches) > 1:
        raise MaterializationContractError(
            f"Base construct subject is not unique in selected sequence authority: {construct_subject_id}"
        )
    return matches[0]


def _construct_subject_row_by_id_or_control(
    rows: list[dict[str, object]],
    *,
    construct_subject_id: str,
    manifest: dict[str, object],
    authority: GenBankAuthorityAudit,
    template_sequence: str,
    target_start: int,
    target_end: int,
) -> dict[str, object]:
    if any(str(row.get("id")) == construct_subject_id for row in rows):
        return _construct_subject_row_by_id(rows, construct_subject_id=construct_subject_id)
    control_rows, _expected_control_sequences = _candidate_rows(
        manifest=manifest,
        authority=authority,
        template_sequence=template_sequence,
        target_start=target_start,
        target_end=target_end,
        construct_subject_sequence_overrides={},
        omitted_construct_subject_fields=set(),
    )
    return _construct_subject_row_by_id(control_rows, construct_subject_id=construct_subject_id)


def _required_candidate_sequence(row: Mapping[str, object], field_name: str) -> str:
    value = row.get(field_name)
    if value is None:
        raise MaterializationContractError(f"{row.get('id')}: {field_name} is required.")
    sequence = str(value)
    if not sequence:
        raise MaterializationContractError(f"{row.get('id')}: {field_name} must be non-empty.")
    return sequence


def _rt_cds_dms_construct_subject_rows(
    *,
    parent_construct_subject_id: str,
    lnrna_sequence: str,
    result: PermuterResult,
) -> list[dict[str, object]]:
    request_id = str(result.request_id)
    study_id = _required_result_metadata(result, "study_id")
    construct_contract = _required_result_metadata(result, "construct_contract")
    representation_contract = _required_result_metadata(result, "representation_contract")
    payload_program_id = _required_result_metadata(result, "payload_program_id")
    source_basis = _required_result_metadata(result, "source_basis")
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(result.records, start=1):
        permuter_meta = CodingDnaDmsVariantMetadata.from_record(record)
        aa_pos = permuter_meta.aa_pos
        aa_wt = permuter_meta.aa_wt
        aa_alt = permuter_meta.aa_alt
        construct_subject_id = f"{parent_construct_subject_id}__rt_cds_dms__{aa_wt}{aa_pos}{aa_alt}"
        if construct_subject_id in seen_ids:
            raise MaterializationContractError(f"Duplicate RT-CDS DMS construct subject id: {construct_subject_id}")
        seen_ids.add(construct_subject_id)
        rows.append(
            {
                "id": construct_subject_id,
                "sequence": "A" * index,
                "source": _MATERIALIZATION_SOURCE,
                **_construct_subject_envelope_overlay(),
                "construct_subject__lnrna_sequence": lnrna_sequence,
                "construct_subject__rt_cds_sequence": record.sequence,
                "construct_subject__study_id": study_id,
                "construct_subject__construct_contract": construct_contract,
                "construct_subject__representation_contract": representation_contract,
                "construct_subject__payload_program_id": payload_program_id,
                "construct_subject__source_basis": source_basis,
                "construct_subject__variant_derivation": "rt_cds_dms_top_codon_policy_v1",
                "construct_subject__construct_projection_status": "representable",
                "construct_subject__role": "in_silico_rt_cds_dms_variant",
                "construct_subject__parent_id": parent_construct_subject_id,
                "construct_subject__dms_slot": "rt_cds",
                "construct_subject__permuter_request_id": request_id,
                "construct_subject__permuter_variant_id": record.id,
                "construct_subject__permuter_modifications": list(record.modifications),
                "construct_subject__rt_cds_dms_aa_pos": aa_pos,
                "construct_subject__rt_cds_dms_aa_wt": aa_wt,
                "construct_subject__rt_cds_dms_aa_alt": aa_alt,
                "construct_subject__rt_cds_dms_codon_index": permuter_meta.codon_index,
                "construct_subject__rt_cds_dms_codon_wt": permuter_meta.codon_wt,
                "construct_subject__rt_cds_dms_codon_new": permuter_meta.codon_new,
                "construct_subject__rt_cds_dms_codon_policy": permuter_meta.codon_policy,
            }
        )
    if not rows:
        raise MaterializationContractError("Permuter RT-CDS DMS result contained no records.")
    return rows


def _duplicates(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return tuple(sorted(repeated))


def _construct_subject_envelope_overlay() -> dict[str, object]:
    return {
        "construct_subject__record_kind": "construct_subject_envelope",
        "construct_subject__sequence_authority": "overlay_only",
        "construct_subject__envelope_carrier_policy": "synthetic_unique_dna4_v1",
        "construct_subject__biological_sequence_fields": list(_CONSTRUCT_SUBJECT_BIOLOGICAL_SEQUENCE_FIELDS),
    }


def _required_result_metadata(result: PermuterResult, field_name: str) -> str:
    value = result.metadata.get(field_name)
    if value is None or str(value).strip() == "":
        raise MaterializationContractError(f"Permuter result metadata missing required field: {field_name}")
    return str(value)


def _catalog_authority_sequence(
    *,
    repo_root: Path,
    genbank_dir: Path,
    parser: BiopythonGenBankParser,
    authority: ExtractedSequenceAuthority,
) -> str:
    source_file = authority.sequence_id.removeprefix("genbank:").split("#", maxsplit=1)[0]
    records = parser.parse_file(repo_root / genbank_dir / source_file)
    if len(records) != 1:
        raise MaterializationContractError(f"{source_file}: expected one GenBank record, found {len(records)}")
    start, end = authority.span_0
    return records[0].sequence[start:end]


def _group_by_window_offset(
    candidates: tuple[_CatalogMaterializationCandidate, ...],
) -> dict[int, tuple[_CatalogMaterializationCandidate, ...]]:
    grouped: dict[int, list[_CatalogMaterializationCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.window_offset_bp, []).append(candidate)
    return {offset: tuple(group) for offset, group in sorted(grouped.items())}


def _sequence_for_binding(*, binding: dict[str, object], authority: GenBankAuthorityAudit) -> str:
    sequence_id = str(binding["sequence_id"])
    source_id = _source_id_for_sequence_id(sequence_id)
    start, end = _span_0(binding["source_sequence_span_0"], label=f"{sequence_id}.source_sequence_span_0")
    return authority.source(source_id).sequence[start:end]


def _source_id_for_sequence_id(sequence_id: str) -> str:
    if not sequence_id.startswith("genbank:"):
        raise MaterializationContractError(f"Unsupported sequence authority id: {sequence_id}")
    path_part = sequence_id.removeprefix("genbank:").split("#", maxsplit=1)[0]
    source_id = _SEQUENCE_ID_SOURCE_MAP.get(path_part)
    if source_id is None:
        raise MaterializationContractError(f"No GenBank source mapping for sequence id: {sequence_id}")
    return source_id


def _expected_context_sequence(
    *,
    template_sequence: str,
    slots: tuple[dict[str, object], ...],
    row: Mapping[str, object],
    target_start: int,
    target_end: int,
) -> str:
    full_construct = _full_construct_sequence(template_sequence=template_sequence, slots=slots, row=row)
    realized_spans = _realized_spans_for_row(template_sequence=template_sequence, slots=slots, row=row)
    window_start, window_end = _candidate_window_bounds(
        slots=slots,
        realized_spans=realized_spans,
        target_start=target_start,
        target_end=target_end,
    )
    return _slice_expected_context(
        full_construct=full_construct, row=row, window_start=window_start, window_end=window_end
    )


def _expected_context_sequence_at_window(
    *,
    template_sequence: str,
    slots: tuple[dict[str, object], ...],
    row: Mapping[str, object],
    window_start: int,
    window_end: int,
) -> str:
    full_construct = _full_construct_sequence(template_sequence=template_sequence, slots=slots, row=row)
    return _slice_expected_context(
        full_construct=full_construct, row=row, window_start=window_start, window_end=window_end
    )


def _full_construct_sequence(
    *,
    template_sequence: str,
    slots: tuple[dict[str, object], ...],
    row: Mapping[str, object],
) -> str:
    cursor = 0
    out: list[str] = []
    for slot in sorted(slots, key=lambda item: _span_0(item["template_span_0"], label="template_span_0")[0]):
        start, end = _span_0(slot["template_span_0"], label=f"{slot['slot_id']}.template_span_0")
        field_name = str(slot["sequence_field"])
        value = row.get(field_name)
        if value is None:
            raise MaterializationContractError(
                f"Input row '{row.get('id')}' is missing field '{field_name}' for part '{slot['slot_id']}'."
            )
        prefix = template_sequence[cursor:start]
        sequence = str(value)
        out.append(prefix)
        out.append(sequence)
        cursor = end
    out.append(template_sequence[cursor:])
    return "".join(out)


def _realized_spans_for_row(
    *,
    template_sequence: str,
    slots: tuple[dict[str, object], ...],
    row: Mapping[str, object],
) -> dict[str, tuple[int, int]]:
    cursor = 0
    out_len = 0
    realized_spans: dict[str, tuple[int, int]] = {}
    for slot in sorted(slots, key=lambda item: _span_0(item["template_span_0"], label="template_span_0")[0]):
        start, end = _span_0(slot["template_span_0"], label=f"{slot['slot_id']}.template_span_0")
        field_name = str(slot["sequence_field"])
        value = row.get(field_name)
        if value is None:
            raise MaterializationContractError(
                f"Input row '{row.get('id')}' is missing field '{field_name}' for part '{slot['slot_id']}'."
            )
        out_len += len(template_sequence[cursor:start])
        realized_start = out_len
        out_len += len(str(value))
        realized_spans[str(slot["slot_id"])] = (realized_start, out_len)
        cursor = end
    return realized_spans


def _slice_expected_context(
    *,
    full_construct: str,
    row: Mapping[str, object],
    window_start: int,
    window_end: int,
) -> str:
    if window_start < 0 or window_end > len(full_construct):
        raise MaterializationContractError(
            f"Input row '{row.get('id')}' target context [{window_start}, {window_end}) falls outside the "
            f"realized construct length {len(full_construct)}."
        )
    return full_construct[window_start:window_end]
