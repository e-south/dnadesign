"""Unified construct-subject source selection for RT-lnRNA materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.permuter import CodingDnaDmsRequest, default_codon_table_path, generate_variants

from ..source_promotions import (
    SourceConstructSubjectPromotion,
    SourcePromotionReport,
    SourceRecordResolver,
    reject_duplicate_msd_compiler_lnrna_sequences,
    resolve_msd_compiler_promotions,
    resolve_source_construct_subject_promotions,
)
from ..variant_genbank_catalog import build_variant_genbank_catalog
from .contracts import (
    _DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
    _DEFAULT_DNADESIGN_DATA_ROOT,
    _DEFAULT_MSD_COMPILER_POOL_SPEC,
    _GENBANK_CATALOG_SOURCE_COLLECTION_ID,
    _PAYLOAD_PROGRAM_ID,
    _RT_CDS_DMS_SOURCE_BASIS,
    MaterializationContractError,
    _ConstructViewRunPlan,
    _MaterializationContext,
)
from .manifest import _source_promotion_window_policy
from .subjects import (
    _candidate_rows,
    _catalog_candidate_rows,
    _catalog_materialization_candidates,
    _construct_subject_row_by_id,
    _construct_subject_row_by_id_or_control,
    _expected_context_sequence,
    _extend_construct_subject_rows,
    _group_by_window_offset,
    _group_source_promotion_rows_by_basis,
    _required_candidate_sequence,
    _rt_cds_dms_construct_subject_rows,
    _source_promotion_rows,
)


@dataclass(frozen=True)
class _UnifiedConstructSubjectSelection:
    rows: list[dict[str, object]]
    expected_sequences: dict[str, str]
    run_plans: tuple[_ConstructViewRunPlan, ...]
    genbank_construct_subject_count: int
    crawford_construct_subject_count: int
    khan_construct_subject_count: int
    msd_compiler_construct_subject_count: int
    rt_cds_dms_construct_subject_count: int
    permuter_request_id: str | None
    source_promotion_report: SourcePromotionReport | None


def _select_unified_construct_subjects(
    *,
    context: _MaterializationContext,
    include_genbank_catalog: bool,
    include_source_promotions: bool,
    include_msd_compiler_promotions: bool,
    include_rt_cds_dms: bool,
    dnadesign_data_root: Path | None,
    source_record_resolver: SourceRecordResolver | None,
    msd_variant_pool_spec_paths: tuple[Path, ...] | None,
    dms_base_construct_subject_id: str,
    rt_cds_positions: tuple[int, ...],
    max_dms_variants: int | None,
) -> _UnifiedConstructSubjectSelection:
    rows: list[dict[str, object]] = []
    expected_sequences: dict[str, str] = {}
    run_plans: list[_ConstructViewRunPlan] = []
    source_promotion_report: SourcePromotionReport | None = None
    permuter_request_id: str | None = None

    genbank_count = 0
    crawford_count = 0
    khan_count = 0
    msd_compiler_count = 0
    rt_cds_dms_count = 0

    if include_genbank_catalog:
        genbank_count = _append_genbank_catalog_subjects(
            context=context,
            rows=rows,
            expected_sequences=expected_sequences,
            run_plans=run_plans,
        )
    if include_source_promotions:
        source_promotion_report = _append_source_promotion_subjects(
            context=context,
            rows=rows,
            expected_sequences=expected_sequences,
            run_plans=run_plans,
            dnadesign_data_root=dnadesign_data_root,
            source_record_resolver=source_record_resolver,
        )
        by_basis = source_promotion_report.candidates_by_basis
        crawford_count = int(by_basis.get("crawford_eco1_lnrna_fixed_wt_rt", 0))
        khan_count = int(by_basis.get("khan_source_rt_lnrna_reference", 0))
    if include_msd_compiler_promotions:
        msd_compiler_count = _append_msd_compiler_subjects(
            context=context,
            rows=rows,
            expected_sequences=expected_sequences,
            run_plans=run_plans,
            msd_variant_pool_spec_paths=msd_variant_pool_spec_paths,
        )
    if include_rt_cds_dms:
        rt_cds_dms_count, permuter_request_id = _append_rt_cds_dms_subjects(
            context=context,
            rows=rows,
            expected_sequences=expected_sequences,
            run_plans=run_plans,
            dms_base_construct_subject_id=dms_base_construct_subject_id,
            rt_cds_positions=rt_cds_positions,
            max_dms_variants=max_dms_variants,
        )
    if not rows:
        raise MaterializationContractError("Unified construct-subject materialization selected no input rows.")

    return _UnifiedConstructSubjectSelection(
        rows=rows,
        expected_sequences=expected_sequences,
        run_plans=tuple(run_plans),
        genbank_construct_subject_count=genbank_count,
        crawford_construct_subject_count=crawford_count,
        khan_construct_subject_count=khan_count,
        msd_compiler_construct_subject_count=msd_compiler_count,
        rt_cds_dms_construct_subject_count=rt_cds_dms_count,
        permuter_request_id=permuter_request_id,
        source_promotion_report=source_promotion_report,
    )


def _append_genbank_catalog_subjects(
    *,
    context: _MaterializationContext,
    rows: list[dict[str, object]],
    expected_sequences: dict[str, str],
    run_plans: list[_ConstructViewRunPlan],
) -> int:
    catalog = build_variant_genbank_catalog(repo_root=context.root)
    if catalog.catalog_id != _GENBANK_CATALOG_SOURCE_COLLECTION_ID:
        raise MaterializationContractError(
            f"Unexpected GenBank catalog id: {catalog.catalog_id}; expected {_GENBANK_CATALOG_SOURCE_COLLECTION_ID}"
        )
    if not catalog.ok:
        joined = "; ".join(catalog.errors)
        raise MaterializationContractError(f"Variant GenBank catalog is invalid: {joined}")
    candidates = _catalog_materialization_candidates(
        repo_root=context.root,
        catalog_genbank_dir=Path(catalog.genbank_dir),
        records=catalog.records,
        target_start=context.target_start,
        target_end=context.target_end,
    )
    catalog_rows, catalog_expected = _catalog_candidate_rows(
        manifest=context.manifest,
        template_sequence=context.template_sequence,
        target_start=context.target_start,
        target_end=context.target_end,
        candidates=candidates,
    )
    _extend_construct_subject_rows(rows, catalog_rows)
    expected_sequences.update(catalog_expected)
    for group_index, (window_offset_bp, group) in enumerate(_group_by_window_offset(candidates).items(), start=1):
        group_name = f"genbank_offset_{group_index:02d}"
        run_plans.append(
            _ConstructViewRunPlan(
                context_job_id=f"rt_lnrna_unified_{group_name}_context_views",
                slot_anchor_job_id=f"rt_lnrna_unified_{group_name}_slot_anchor_views",
                context_config_name=f"construct-{group_name}-context-views.yaml",
                slot_anchor_config_name=f"construct-{group_name}-slot-anchor-views.yaml",
                subject_ids=tuple(candidate.construct_subject_id for candidate in group),
                window_offset_bp=window_offset_bp,
            )
        )
    return len(catalog_rows)


def _append_source_promotion_subjects(
    *,
    context: _MaterializationContext,
    rows: list[dict[str, object]],
    expected_sequences: dict[str, str],
    run_plans: list[_ConstructViewRunPlan],
    dnadesign_data_root: Path | None,
    source_record_resolver: SourceRecordResolver | None,
) -> SourcePromotionReport:
    wt_parent = _parent_construct_subject(
        context=context,
        rows=rows,
        construct_subject_id=_DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
        allow_control_fallback=True,
    )
    report = resolve_source_construct_subject_promotions(
        dnadesign_data_root=(context.root / (dnadesign_data_root or _DEFAULT_DNADESIGN_DATA_ROOT)).resolve(),
        wt_rt_cds_sequence=_required_candidate_sequence(wt_parent, "construct_subject__rt_cds_sequence"),
        window_policy=_source_promotion_window_policy(
            manifest=context.manifest,
            template_sequence=context.template_sequence,
            target_start=context.target_start,
            target_end=context.target_end,
        ),
        source_record_resolver=source_record_resolver,
    )
    source_rows, source_expected = _source_promotion_rows(
        manifest=context.manifest,
        template_sequence=context.template_sequence,
        target_start=context.target_start,
        target_end=context.target_end,
        promotions=report.candidates,
    )
    if source_rows:
        _extend_construct_subject_rows(rows, source_rows)
        expected_sequences.update(source_expected)
        for basis, group in _group_source_promotion_rows_by_basis(source_rows).items():
            run_plans.append(
                _ConstructViewRunPlan(
                    context_job_id=f"rt_lnrna_unified_{basis}_context_views",
                    slot_anchor_job_id=f"rt_lnrna_unified_{basis}_slot_anchor_views",
                    context_config_name=f"construct-{basis}-context-views.yaml",
                    slot_anchor_config_name=f"construct-{basis}-slot-anchor-views.yaml",
                    subject_ids=tuple(str(row["id"]) for row in group),
                    window_offset_bp=None,
                )
            )
    return report


def _append_msd_compiler_subjects(
    *,
    context: _MaterializationContext,
    rows: list[dict[str, object]],
    expected_sequences: dict[str, str],
    run_plans: list[_ConstructViewRunPlan],
    msd_variant_pool_spec_paths: tuple[Path, ...] | None,
) -> int:
    wt_parent = _parent_construct_subject(
        context=context,
        rows=rows,
        construct_subject_id=_DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
        allow_control_fallback=True,
    )
    pool_spec_paths = msd_variant_pool_spec_paths or (_DEFAULT_MSD_COMPILER_POOL_SPEC,)
    promotions: list[SourceConstructSubjectPromotion] = []
    for pool_spec_path in pool_spec_paths:
        promotions.extend(
            resolve_msd_compiler_promotions(
                repo_root=context.root,
                pool_spec_path=context.root / pool_spec_path,
                wt_rt_cds_sequence=_required_candidate_sequence(wt_parent, "construct_subject__rt_cds_sequence"),
                window_policy=_source_promotion_window_policy(
                    manifest=context.manifest,
                    template_sequence=context.template_sequence,
                    target_start=context.target_start,
                    target_end=context.target_end,
                ),
            )
        )
    reject_duplicate_msd_compiler_lnrna_sequences(promotions)
    msd_rows, msd_expected = _source_promotion_rows(
        manifest=context.manifest,
        template_sequence=context.template_sequence,
        target_start=context.target_start,
        target_end=context.target_end,
        promotions=tuple(promotions),
    )
    if not msd_rows:
        return 0
    _extend_construct_subject_rows(rows, msd_rows)
    expected_sequences.update(msd_expected)
    run_plans.append(
        _ConstructViewRunPlan(
            context_job_id="rt_lnrna_unified_compiler_generated_msd_lnrna_variant_context_views",
            slot_anchor_job_id="rt_lnrna_unified_compiler_generated_msd_lnrna_variant_slot_anchor_views",
            context_config_name="construct-compiler_generated_msd_lnrna_variant-context-views.yaml",
            slot_anchor_config_name="construct-compiler_generated_msd_lnrna_variant-slot-anchor-views.yaml",
            subject_ids=tuple(str(row["id"]) for row in msd_rows),
            window_offset_bp=None,
        )
    )
    return len(msd_rows)


def _append_rt_cds_dms_subjects(
    *,
    context: _MaterializationContext,
    rows: list[dict[str, object]],
    expected_sequences: dict[str, str],
    run_plans: list[_ConstructViewRunPlan],
    dms_base_construct_subject_id: str,
    rt_cds_positions: tuple[int, ...],
    max_dms_variants: int | None,
) -> tuple[int, str]:
    parent = _parent_construct_subject(
        context=context,
        rows=rows,
        construct_subject_id=dms_base_construct_subject_id,
        allow_control_fallback=False,
    )
    request = CodingDnaDmsRequest(
        ref_name=f"{dms_base_construct_subject_id}__rt_cds",
        sequence=_required_candidate_sequence(parent, "construct_subject__rt_cds_sequence"),
        codon_table=default_codon_table_path("ecoli"),
        positions=rt_cds_positions,
        max_variants=max_dms_variants,
        metadata={
            "study_id": str(context.manifest["study_id"]),
            "construct_contract": str(context.manifest["construct_contract"]),
            "representation_contract": str(context.manifest["representation_contract"]),
            "payload_program_id": _PAYLOAD_PROGRAM_ID,
            "source_basis": _RT_CDS_DMS_SOURCE_BASIS,
            "parent_construct_subject_id": dms_base_construct_subject_id,
            "slot_id": "rt_cds",
        },
    )
    result = generate_variants(request)
    dms_rows = _rt_cds_dms_construct_subject_rows(
        parent_construct_subject_id=dms_base_construct_subject_id,
        lnrna_sequence=_required_candidate_sequence(parent, "construct_subject__lnrna_sequence"),
        result=result,
    )
    _extend_construct_subject_rows(rows, dms_rows)
    for row in dms_rows:
        subject_id = str(row["id"])
        expected_sequences[subject_id] = _expected_context_sequence(
            template_sequence=context.template_sequence,
            slots=context.slots,
            row=row,
            target_start=context.target_start,
            target_end=context.target_end,
        )
    run_plans.append(
        _ConstructViewRunPlan(
            context_job_id="rt_lnrna_unified_rt_cds_dms_context_views",
            slot_anchor_job_id="rt_lnrna_unified_rt_cds_dms_slot_anchor_views",
            context_config_name="construct-rt_cds_dms-context-views.yaml",
            slot_anchor_config_name="construct-rt_cds_dms-slot-anchor-views.yaml",
            subject_ids=tuple(str(row["id"]) for row in dms_rows),
            window_offset_bp=None,
        )
    )
    return len(dms_rows), result.request_id


def _parent_construct_subject(
    *,
    context: _MaterializationContext,
    rows: list[dict[str, object]],
    construct_subject_id: str,
    allow_control_fallback: bool,
) -> dict[str, object]:
    source_rows = rows
    if not source_rows:
        source_rows, _expected_control_sequences = _candidate_rows(
            manifest=context.manifest,
            authority=context.authority,
            template_sequence=context.template_sequence,
            target_start=context.target_start,
            target_end=context.target_end,
            construct_subject_sequence_overrides={},
            omitted_construct_subject_fields=set(),
        )
    if allow_control_fallback:
        return _construct_subject_row_by_id_or_control(
            source_rows,
            construct_subject_id=construct_subject_id,
            manifest=context.manifest,
            authority=context.authority,
            template_sequence=context.template_sequence,
            target_start=context.target_start,
            target_end=context.target_end,
        )
    return _construct_subject_row_by_id(source_rows, construct_subject_id=construct_subject_id)
