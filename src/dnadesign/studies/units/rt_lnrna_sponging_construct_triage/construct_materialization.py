"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/construct_materialization.py

Construct materialization helpers for the RT-lnRNA sponging construct triage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dnadesign.permuter import CodingDnaDmsRequest, default_codon_table_path, generate_variants

from .materialization.contracts import (
    _DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
    _INPUT_DATASET,
    _OUTPUT_DATASET,
    _PAYLOAD_PROGRAM_ID,
    _RT_CDS_DMS_SOURCE_BASIS,
    ControlConstructMaterializationReport,
    MaterializationContractError,
    RtCdsDmsConstructMaterializationReport,
    UnifiedConstructSubjectMaterializationReport,
    _ConstructViewRunPlan,
)
from .materialization.execution import (
    _load_materialization_context,
    _materialize_construct_view_plans,
)
from .materialization.subject_binding_adapter import _adapt_registered_subject_bindings
from .materialization.subjects import (
    _candidate_rows,
    _construct_subject_row_by_id,
    _expected_context_sequence,
    _required_candidate_sequence,
    _rt_cds_dms_construct_subject_rows,
)
from .materialization.unified import _select_unified_construct_subjects
from .source_promotions import SourceRecordResolver


def materialize_control_construct_contexts(
    *,
    repo_root: Path | None = None,
    work_root: Path,
    construct_subject_sequence_overrides: Mapping[str, Mapping[str, str]] | None = None,
    omitted_construct_subject_fields: tuple[str, ...] = (),
) -> ControlConstructMaterializationReport:
    """Materialize the two checked-in control candidates into temp USR outputs."""
    context = _load_materialization_context(repo_root)
    rows, expected_sequences = _candidate_rows(
        manifest=context.manifest,
        authority=context.authority,
        template_sequence=context.template_sequence,
        target_start=context.target_start,
        target_end=context.target_end,
        construct_subject_sequence_overrides=construct_subject_sequence_overrides or {},
        omitted_construct_subject_fields=set(omitted_construct_subject_fields),
    )
    run = _materialize_construct_view_plans(
        work_root=work_root,
        context=context,
        rows=rows,
        plans=(
            _ConstructViewRunPlan(
                context_job_id="rt_lnrna_control_context_views",
                slot_anchor_job_id="rt_lnrna_slot_anchor_views",
                context_config_name="construct-context-views.yaml",
                slot_anchor_config_name="construct-slot-anchor-views.yaml",
                subject_ids=tuple(str(row["id"]) for row in rows),
            ),
        ),
    )
    return ControlConstructMaterializationReport(
        usr_root=run.usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_subject_id=run.input_ids_by_subject_id,
        config_paths=run.config_paths,
        run_results=run.run_results,
        template_sequence=context.template_sequence,
        template_context_sequence=context.template_context_sequence,
        expected_sequences=expected_sequences,
    )


def materialize_variant_construct_contexts(
    *,
    repo_root: Path | None = None,
    work_root: Path,
) -> ControlConstructMaterializationReport:
    """Audit catalog-provenance subjects after exact subject-binding resolution."""
    context = _load_materialization_context(repo_root)
    projection = _adapt_registered_subject_bindings(context=context)
    rows = [
        row
        for row in projection.rows
        if row.get("construct_subject__lnrna_authority_kind") == "rt_lnrna_variant_genbank_catalog"
        and row.get("construct_subject__rt_cds_authority_kind") == "rt_lnrna_variant_genbank_catalog"
    ]
    resolved_ids = {str(row["id"]) for row in rows}
    expected_catalog_ids = set(projection.catalog_subject_ids)
    if resolved_ids != expected_catalog_ids:
        missing = sorted(expected_catalog_ids - resolved_ids)
        unexpected = sorted(resolved_ids - expected_catalog_ids)
        raise MaterializationContractError(
            "Catalog provenance audit requires exact subject-binding coverage; "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_sequences = {subject_id: projection.expected_sequences[subject_id] for subject_id in resolved_ids}

    plans: list[_ConstructViewRunPlan] = []
    for group_index, (window_offset_bp, subject_ids) in enumerate(
        projection.subject_ids_by_window_offset.items(), start=1
    ):
        selected_subject_ids = tuple(subject_id for subject_id in subject_ids if subject_id in resolved_ids)
        if not selected_subject_ids:
            continue
        plans.append(
            _ConstructViewRunPlan(
                context_job_id=f"rt_lnrna_variant_context_views_offset_{group_index}",
                slot_anchor_job_id=f"rt_lnrna_variant_slot_anchor_views_offset_{group_index}",
                context_config_name=f"construct-context-views-{group_index:02d}.yaml",
                slot_anchor_config_name=f"construct-slot-anchor-views-{group_index:02d}.yaml",
                subject_ids=selected_subject_ids,
                window_offset_bp=window_offset_bp,
            )
        )
    run = _materialize_construct_view_plans(
        work_root=work_root,
        context=context,
        rows=rows,
        plans=tuple(plans),
    )
    return ControlConstructMaterializationReport(
        usr_root=run.usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_subject_id=run.input_ids_by_subject_id,
        config_paths=run.config_paths,
        run_results=run.run_results,
        template_sequence=context.template_sequence,
        template_context_sequence=context.template_context_sequence,
        expected_sequences=expected_sequences,
    )


def materialize_rt_cds_dms_construct_contexts(
    *,
    repo_root: Path | None = None,
    work_root: Path,
    base_construct_subject_id: str,
    rt_cds_positions: tuple[int, ...] = (),
    max_variants: int | None = None,
) -> RtCdsDmsConstructMaterializationReport:
    """Materialize RT-CDS in silico DMS variants via the public Permuter API."""
    context = _load_materialization_context(repo_root)
    parent_rows, _expected_control_sequences = _candidate_rows(
        manifest=context.manifest,
        authority=context.authority,
        template_sequence=context.template_sequence,
        target_start=context.target_start,
        target_end=context.target_end,
        construct_subject_sequence_overrides={},
        omitted_construct_subject_fields=set(),
    )
    parent = _construct_subject_row_by_id(parent_rows, construct_subject_id=base_construct_subject_id)
    request = CodingDnaDmsRequest(
        ref_name=f"{base_construct_subject_id}__rt_cds",
        sequence=_required_candidate_sequence(parent, "construct_subject__rt_cds_sequence"),
        codon_table=default_codon_table_path("ecoli"),
        positions=rt_cds_positions,
        max_variants=max_variants,
        metadata={
            "study_id": str(context.manifest["study_id"]),
            "construct_contract": str(context.manifest["construct_contract"]),
            "representation_contract": str(context.manifest["representation_contract"]),
            "payload_program_id": _PAYLOAD_PROGRAM_ID,
            "source_basis": _RT_CDS_DMS_SOURCE_BASIS,
            "parent_construct_subject_id": base_construct_subject_id,
            "slot_id": "rt_cds",
        },
    )
    result = generate_variants(request)
    if max_variants is not None and len(result.records) > max_variants:
        raise MaterializationContractError(
            f"RT-CDS DMS request produced {len(result.records)} variants, above max_variants={max_variants}."
        )
    rows = _rt_cds_dms_construct_subject_rows(
        parent_construct_subject_id=base_construct_subject_id,
        lnrna_sequence=_required_candidate_sequence(parent, "construct_subject__lnrna_sequence"),
        result=result,
    )
    expected_sequences = {
        str(row["id"]): _expected_context_sequence(
            template_sequence=context.template_sequence,
            slots=context.slots,
            row=row,
            target_start=context.target_start,
            target_end=context.target_end,
        )
        for row in rows
    }
    run = _materialize_construct_view_plans(
        work_root=work_root,
        context=context,
        rows=rows,
        plans=(
            _ConstructViewRunPlan(
                context_job_id="rt_lnrna_rt_cds_dms_context_views",
                slot_anchor_job_id="rt_lnrna_rt_cds_dms_slot_anchor_views",
                context_config_name="construct-rt-cds-dms-context-views.yaml",
                slot_anchor_config_name="construct-rt-cds-dms-slot-anchor-views.yaml",
                subject_ids=tuple(str(row["id"]) for row in rows),
            ),
        ),
    )
    return RtCdsDmsConstructMaterializationReport(
        usr_root=run.usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_subject_id=run.input_ids_by_subject_id,
        config_paths=run.config_paths,
        run_results=run.run_results,
        template_sequence=context.template_sequence,
        template_context_sequence=context.template_context_sequence,
        expected_sequences=expected_sequences,
        base_construct_subject_id=base_construct_subject_id,
        permuter_request_id=result.request_id,
    )


def materialize_unified_construct_subject_contexts(
    *,
    repo_root: Path | None = None,
    work_root: Path,
    include_source_promotions: bool = False,
    include_msd_compiler_promotions: bool = False,
    include_rt_cds_dms: bool = False,
    dnadesign_data_root: Path | None = None,
    source_record_resolver: SourceRecordResolver | None = None,
    msd_variant_pool_spec_paths: tuple[Path, ...] | None = None,
    dms_base_construct_subject_id: str = _DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
    rt_cds_positions: tuple[int, ...] = (),
    max_dms_variants: int | None = None,
    subject_binding_subject_ids: tuple[str, ...] | None = None,
    allow_partial_byte_resolution: bool = False,
) -> UnifiedConstructSubjectMaterializationReport:
    """Materialize registered subjects; partial byte resolution requires explicit opt-in."""
    context = _load_materialization_context(repo_root)
    selection = _select_unified_construct_subjects(
        context=context,
        include_source_promotions=include_source_promotions,
        include_msd_compiler_promotions=include_msd_compiler_promotions,
        include_rt_cds_dms=include_rt_cds_dms,
        dnadesign_data_root=dnadesign_data_root,
        source_record_resolver=source_record_resolver,
        msd_variant_pool_spec_paths=msd_variant_pool_spec_paths,
        dms_base_construct_subject_id=dms_base_construct_subject_id,
        rt_cds_positions=rt_cds_positions,
        max_dms_variants=max_dms_variants,
        subject_binding_subject_ids=subject_binding_subject_ids,
        allow_partial_byte_resolution=allow_partial_byte_resolution,
    )
    run = _materialize_construct_view_plans(
        work_root=work_root,
        context=context,
        rows=selection.rows,
        plans=selection.run_plans,
    )
    return UnifiedConstructSubjectMaterializationReport(
        usr_root=run.usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_subject_id=run.input_ids_by_subject_id,
        config_paths=run.config_paths,
        run_results=run.run_results,
        template_sequence=context.template_sequence,
        template_context_sequence=context.template_context_sequence,
        expected_sequences=selection.expected_sequences,
        subject_binding_requested_subject_count=selection.subject_binding_requested_subject_count,
        subject_binding_resolved_subject_count=selection.subject_binding_resolved_subject_count,
        subject_binding_blocked_subject_count=selection.subject_binding_blocked_subject_count,
        subject_binding_resolution_complete=selection.subject_binding_resolution_complete,
        blocked_subject_bindings=selection.blocked_subject_bindings,
        crawford_construct_subject_count=selection.crawford_construct_subject_count,
        khan_construct_subject_count=selection.khan_construct_subject_count,
        msd_compiler_construct_subject_count=selection.msd_compiler_construct_subject_count,
        rt_cds_dms_construct_subject_count=selection.rt_cds_dms_construct_subject_count,
        permuter_request_id=selection.permuter_request_id,
        source_promotion_report=selection.source_promotion_report,
    )
