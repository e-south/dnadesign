"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/materialization/subject_binding_adapter.py

Adapt validated RT-lnRNA subject bindings into Construct input rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..subject_bindings import (
    ResolvedSubjectBinding,
    SubjectBindingByteBlock,
    SubjectBindingContractError,
    load_registered_subject_binding_materialization,
)
from ..variant_genbank_catalog import build_variant_genbank_catalog
from .contracts import (
    _MATERIALIZATION_SOURCE,
    MaterializationContractError,
    _CatalogMaterializationCandidate,
    _MaterializationContext,
)
from .subjects import (
    _catalog_materialization_candidates,
    _construct_subject_envelope_overlay,
    _expected_context_sequence_at_window,
)

_SOURCE_BASIS = "rt_lnrna_subject_binding"


@dataclass(frozen=True, slots=True)
class _SubjectBindingConstructRows:
    rows: list[dict[str, object]]
    expected_sequences: dict[str, str]
    subject_ids_by_window_offset: dict[int, tuple[str, ...]]
    catalog_subject_ids: tuple[str, ...]
    blocked_subjects: tuple[SubjectBindingByteBlock, ...]


def _adapt_registered_subject_bindings(
    *,
    context: _MaterializationContext,
    subject_ids: tuple[str, ...] | None = None,
) -> _SubjectBindingConstructRows:
    """Project independently available bytes and preserve explicit provider blocks."""

    try:
        resolution = load_registered_subject_binding_materialization(
            repo_root=context.root,
            subject_ids=subject_ids,
        )
    except SubjectBindingContractError as exc:
        raise MaterializationContractError(f"RT-lnRNA subject binding projection blocked: {exc}") from exc
    geometry_by_lnrna_part = _lnrna_geometry_by_part_id(context=context)
    slots = context.slots
    rows: list[dict[str, object]] = []
    expected_sequences: dict[str, str] = {}
    subject_ids_by_offset: dict[int, list[str]] = {}
    seen_subject_ids: set[str] = set()
    for resolved in resolution.resolved_subjects:
        binding = resolved.binding
        if binding.construct_projection_status != "representable":
            continue
        if binding.subject_id in seen_subject_ids:
            raise MaterializationContractError(f"Duplicate resolved subject binding: {binding.subject_id}")
        seen_subject_ids.add(binding.subject_id)
        geometry = geometry_by_lnrna_part.get(binding.lnrna_part.record_id)
        if geometry is None:
            raise MaterializationContractError(
                f"{binding.subject_id}: no catalog-backed Construct placement for lnRNA part "
                f"{binding.lnrna_part.record_id!r}"
            )
        _require_matching_component_bytes(resolved=resolved, geometry=geometry)
        row = _construct_row(resolved=resolved)
        rows.append(row)
        expected_sequences[binding.subject_id] = _expected_context_sequence_at_window(
            template_sequence=context.template_sequence,
            slots=slots,
            row=row,
            window_start=geometry.window_start,
            window_end=geometry.window_start + (context.target_end - context.target_start),
        )
        subject_ids_by_offset.setdefault(geometry.window_offset_bp, []).append(binding.subject_id)
    if not rows:
        raise MaterializationContractError("Subject binding registry contains no representable Construct subjects.")
    return _SubjectBindingConstructRows(
        rows=rows,
        expected_sequences=expected_sequences,
        subject_ids_by_window_offset={
            offset: tuple(subject_ids) for offset, subject_ids in sorted(subject_ids_by_offset.items())
        },
        catalog_subject_ids=tuple(candidate.construct_subject_id for candidate in geometry_by_lnrna_part.values()),
        blocked_subjects=resolution.blocked_subjects,
    )


def _lnrna_geometry_by_part_id(*, context: _MaterializationContext) -> dict[str, _CatalogMaterializationCandidate]:
    catalog = build_variant_genbank_catalog(repo_root=context.root)
    if not catalog.ok:
        raise MaterializationContractError("Variant GenBank catalog is invalid: " + "; ".join(catalog.errors))
    candidates = _catalog_materialization_candidates(
        repo_root=context.root,
        catalog_genbank_dir=Path(catalog.genbank_dir),
        records=catalog.records,
        target_start=context.target_start,
        target_end=context.target_end,
    )
    by_part_id: dict[str, _CatalogMaterializationCandidate] = {}
    for candidate in candidates:
        prior = by_part_id.get(candidate.source_variant_id)
        if prior is not None:
            raise MaterializationContractError(
                f"Ambiguous catalog-backed Construct placement for lnRNA part {candidate.source_variant_id!r}"
            )
        by_part_id[candidate.source_variant_id] = candidate
    return by_part_id


def _require_matching_component_bytes(
    *, resolved: ResolvedSubjectBinding, geometry: _CatalogMaterializationCandidate
) -> None:
    binding = resolved.binding
    if geometry.lnrna_sequence != resolved.lnrna_sequence:
        raise MaterializationContractError(
            f"{binding.subject_id}: resolved lnRNA bytes disagree with catalog-backed Construct placement"
        )
    if geometry.construct_subject_id == binding.subject_id and geometry.rt_cds_sequence != resolved.rt_cds_sequence:
        raise MaterializationContractError(
            f"{binding.subject_id}: resolved RT CDS bytes disagree with the matching catalog provenance record"
        )


def _construct_row(*, resolved: ResolvedSubjectBinding) -> dict[str, object]:
    binding = resolved.binding
    msd_variant_id = binding.msd_structure.variant_id if binding.msd_structure is not None else ""
    reader_design_ids = [alias.value for alias in binding.aliases if alias.namespace == "reader.design_id"]
    if len(reader_design_ids) > 1:
        raise MaterializationContractError(f"{binding.subject_id}: multiple reader.design_id aliases are ambiguous")
    return {
        "id": binding.subject_id,
        "source": _MATERIALIZATION_SOURCE,
        **_construct_subject_envelope_overlay(),
        "construct_subject__lnrna_sequence": resolved.lnrna_sequence,
        "construct_subject__rt_cds_sequence": resolved.rt_cds_sequence,
        "construct_subject__source_basis": _SOURCE_BASIS,
        "construct_subject__source_collection_id": resolved.binding_set_id,
        "construct_subject__source_record_id": binding.subject_id,
        "construct_subject__source_record_count": 1,
        "construct_subject__payload_program_id": binding.payload_program_id,
        "construct_subject__construct_projection_status": binding.construct_projection_status,
        "construct_subject__role": "construct_subject",
        "construct_subject__reader_design_id": reader_design_ids[0] if reader_design_ids else "",
        "construct_subject__lnrna_part_id": binding.lnrna_part.part_id,
        "construct_subject__lnrna_part_owner": binding.lnrna_part.owner_study_id,
        "construct_subject__lnrna_authority_kind": binding.lnrna_part.authority_kind,
        "construct_subject__lnrna_sequence_sha256": binding.lnrna_part.sequence_sha256,
        "construct_subject__rt_part_id": binding.rt_part.part_id,
        "construct_subject__rt_part_owner": binding.rt_part.owner_study_id,
        "construct_subject__rt_cds_authority_kind": binding.rt_part.authority_kind,
        "construct_subject__rt_cds_sequence_sha256": binding.rt_part.sequence_sha256,
        "construct_subject__msd_variant_id": msd_variant_id,
    }


__all__ = []
