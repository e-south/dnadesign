"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/construct_materialization.py

Construct materialization helpers for the RT-lnRNA sponging construct triage
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from dnadesign.construct import RunResult, run_from_config
from dnadesign.permuter import CodingDnaDmsRequest, default_codon_table_path, generate_variants

from .materialization.common import _list, _mapping, _resolve_repo_root
from .materialization.contracts import (
    _DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
    _DEFAULT_DNADESIGN_DATA_ROOT,
    _DEFAULT_MSD_COMPILER_POOL_SPEC,
    _GENBANK_CATALOG_SOURCE_COLLECTION_ID,
    _INPUT_DATASET,
    _OUTPUT_DATASET,
    _PAYLOAD_PROGRAM_ID,
    _RT_CDS_DMS_SOURCE_BASIS,
    ControlConstructMaterializationReport,
    MaterializationContractError,
    RtCdsDmsConstructMaterializationReport,
    UnifiedConstructSubjectMaterializationReport,
    _ConstructSubjectRunGroup,
)
from .materialization.manifest import (
    _load_projection_manifest,
    _require_genbank_authority,
    _require_valid_projection_manifest,
    _source_promotion_window_policy,
    _target_context_bounds,
    _template_sequence,
)
from .materialization.subjects import (
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
from .materialization.usr_io import (
    _require_construct_infer_ready,
    _write_construct_output_subject_bridge,
    _write_construct_subject_dataset,
)
from .materialization.views import (
    _construct_config,
    _context_output_variants,
    _slot_anchor_output_variants,
    _write_config,
)
from .source_promotions import (
    SourceConstructSubjectPromotion,
    SourcePromotionReport,
    SourceRecordResolver,
    reject_duplicate_msd_compiler_lnrna_sequences,
    resolve_msd_compiler_promotions,
    resolve_source_construct_subject_promotions,
)
from .variant_genbank_catalog import build_variant_genbank_catalog


def materialize_control_construct_contexts(
    *,
    repo_root: Path | None = None,
    work_root: Path,
    construct_subject_sequence_overrides: Mapping[str, Mapping[str, str]] | None = None,
    omitted_construct_subject_fields: tuple[str, ...] = (),
) -> ControlConstructMaterializationReport:
    """Materialize the two checked-in control candidates into temp USR outputs."""
    root = _resolve_repo_root(repo_root)
    manifest = _load_projection_manifest(root)
    _require_valid_projection_manifest(manifest)
    authority = _require_genbank_authority(root)
    template_sequence = _template_sequence(manifest=manifest, authority=authority)
    target_start, target_end = _target_context_bounds(manifest)
    template_context_sequence = template_sequence[target_start:target_end]

    rows, expected_sequences = _candidate_rows(
        manifest=manifest,
        authority=authority,
        template_sequence=template_sequence,
        target_start=target_start,
        target_end=target_end,
        construct_subject_sequence_overrides=construct_subject_sequence_overrides or {},
        omitted_construct_subject_fields=set(omitted_construct_subject_fields),
    )

    work = Path(work_root).resolve()
    usr_root = work / "usr"
    config_dir = work / "construct_configs"
    usr_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    input_ids_by_subject_id = _write_construct_subject_dataset(usr_root=usr_root, rows=rows)

    context_config = _construct_config(
        manifest=manifest,
        template_sequence=template_sequence,
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
        job_id="rt_lnrna_control_context_views",
        output_on_conflict="error",
        output_variants=_context_output_variants(),
    )
    slot_anchor_config = _construct_config(
        manifest=manifest,
        template_sequence=template_sequence,
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
        job_id="rt_lnrna_slot_anchor_views",
        output_on_conflict="ignore",
        output_variants=_slot_anchor_output_variants(),
    )
    context_path = _write_config(config_dir / "construct-context-views.yaml", context_config)
    slot_anchor_path = _write_config(config_dir / "construct-slot-anchor-views.yaml", slot_anchor_config)
    context_result = run_from_config(context_path)
    slot_anchor_result = run_from_config(slot_anchor_path)
    _write_construct_output_subject_bridge(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    _require_construct_infer_ready(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    return ControlConstructMaterializationReport(
        usr_root=usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_subject_id=input_ids_by_subject_id,
        config_paths=(context_path, slot_anchor_path),
        run_results=(context_result, slot_anchor_result),
        template_sequence=template_sequence,
        template_context_sequence=template_context_sequence,
        expected_sequences=expected_sequences,
    )


def materialize_variant_construct_contexts(
    *,
    repo_root: Path | None = None,
    work_root: Path,
) -> ControlConstructMaterializationReport:
    """Materialize all catalog-representable variants into consolidated 2,000 bp views."""
    root = _resolve_repo_root(repo_root)
    manifest = _load_projection_manifest(root)
    _require_valid_projection_manifest(manifest)
    authority = _require_genbank_authority(root)
    template_sequence = _template_sequence(manifest=manifest, authority=authority)
    target_start, target_end = _target_context_bounds(manifest)
    template_context_sequence = template_sequence[target_start:target_end]
    catalog = build_variant_genbank_catalog(repo_root=root)
    if not catalog.ok:
        joined = "; ".join(catalog.errors)
        raise MaterializationContractError(f"Variant GenBank catalog is invalid: {joined}")
    candidates = _catalog_materialization_candidates(
        repo_root=root,
        catalog_genbank_dir=Path(catalog.genbank_dir),
        records=catalog.records,
        target_start=target_start,
        target_end=target_end,
    )
    rows, expected_sequences = _catalog_candidate_rows(
        manifest=manifest,
        template_sequence=template_sequence,
        target_start=target_start,
        target_end=target_end,
        candidates=candidates,
    )

    work = Path(work_root).resolve()
    usr_root = work / "usr"
    config_dir = work / "construct_configs"
    usr_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    input_ids_by_subject_id = _write_construct_subject_dataset(usr_root=usr_root, rows=rows)

    run_results: list[RunResult] = []
    config_paths: list[Path] = []
    for group_index, (window_offset_bp, group) in enumerate(_group_by_window_offset(candidates).items(), start=1):
        construct_subject_ids = tuple(candidate.construct_subject_id for candidate in group)
        context_config = _construct_config(
            manifest=manifest,
            template_sequence=template_sequence,
            usr_root=usr_root,
            input_ids_by_subject_id=input_ids_by_subject_id,
            job_id=f"rt_lnrna_variant_context_views_offset_{group_index}",
            output_on_conflict="error",
            output_variants=_context_output_variants(),
            construct_subject_ids=construct_subject_ids,
            window_offset_bp=window_offset_bp,
        )
        slot_anchor_config = _construct_config(
            manifest=manifest,
            template_sequence=template_sequence,
            usr_root=usr_root,
            input_ids_by_subject_id=input_ids_by_subject_id,
            job_id=f"rt_lnrna_variant_slot_anchor_views_offset_{group_index}",
            output_on_conflict="ignore",
            output_variants=_slot_anchor_output_variants(),
            construct_subject_ids=construct_subject_ids,
            window_offset_bp=window_offset_bp,
        )
        context_path = _write_config(config_dir / f"construct-context-views-{group_index:02d}.yaml", context_config)
        slot_anchor_path = _write_config(
            config_dir / f"construct-slot-anchor-views-{group_index:02d}.yaml",
            slot_anchor_config,
        )
        config_paths.extend([context_path, slot_anchor_path])
        run_results.extend([run_from_config(context_path), run_from_config(slot_anchor_path)])

    _write_construct_output_subject_bridge(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    _require_construct_infer_ready(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    return ControlConstructMaterializationReport(
        usr_root=usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_subject_id=input_ids_by_subject_id,
        config_paths=tuple(config_paths),
        run_results=tuple(run_results),
        template_sequence=template_sequence,
        template_context_sequence=template_context_sequence,
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
    root = _resolve_repo_root(repo_root)
    manifest = _load_projection_manifest(root)
    _require_valid_projection_manifest(manifest)
    authority = _require_genbank_authority(root)
    template_sequence = _template_sequence(manifest=manifest, authority=authority)
    target_start, target_end = _target_context_bounds(manifest)
    template_context_sequence = template_sequence[target_start:target_end]

    parent_rows, _expected_control_sequences = _candidate_rows(
        manifest=manifest,
        authority=authority,
        template_sequence=template_sequence,
        target_start=target_start,
        target_end=target_end,
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
            "study_id": str(manifest["study_id"]),
            "construct_contract": str(manifest["construct_contract"]),
            "representation_contract": str(manifest["representation_contract"]),
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
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    expected_sequences = {
        str(row["id"]): _expected_context_sequence(
            template_sequence=template_sequence,
            slots=slots,
            row=row,
            target_start=target_start,
            target_end=target_end,
        )
        for row in rows
    }

    work = Path(work_root).resolve()
    usr_root = work / "usr"
    config_dir = work / "construct_configs"
    usr_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    input_ids_by_subject_id = _write_construct_subject_dataset(usr_root=usr_root, rows=rows)

    context_config = _construct_config(
        manifest=manifest,
        template_sequence=template_sequence,
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
        job_id="rt_lnrna_rt_cds_dms_context_views",
        output_on_conflict="error",
        output_variants=_context_output_variants(),
        construct_subject_ids=tuple(str(row["id"]) for row in rows),
    )
    slot_anchor_config = _construct_config(
        manifest=manifest,
        template_sequence=template_sequence,
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
        job_id="rt_lnrna_rt_cds_dms_slot_anchor_views",
        output_on_conflict="ignore",
        output_variants=_slot_anchor_output_variants(),
        construct_subject_ids=tuple(str(row["id"]) for row in rows),
    )
    context_path = _write_config(config_dir / "construct-rt-cds-dms-context-views.yaml", context_config)
    slot_anchor_path = _write_config(config_dir / "construct-rt-cds-dms-slot-anchor-views.yaml", slot_anchor_config)
    context_result = run_from_config(context_path)
    slot_anchor_result = run_from_config(slot_anchor_path)
    _write_construct_output_subject_bridge(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    _require_construct_infer_ready(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    return RtCdsDmsConstructMaterializationReport(
        usr_root=usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_subject_id=input_ids_by_subject_id,
        config_paths=(context_path, slot_anchor_path),
        run_results=(context_result, slot_anchor_result),
        template_sequence=template_sequence,
        template_context_sequence=template_context_sequence,
        expected_sequences=expected_sequences,
        base_construct_subject_id=base_construct_subject_id,
        permuter_request_id=result.request_id,
    )


def materialize_unified_construct_subject_contexts(
    *,
    repo_root: Path | None = None,
    work_root: Path,
    include_genbank_catalog: bool = True,
    include_source_promotions: bool = True,
    include_msd_compiler_promotions: bool = True,
    include_rt_cds_dms: bool = True,
    dnadesign_data_root: Path | None = None,
    source_record_resolver: SourceRecordResolver | None = None,
    msd_variant_pool_spec_paths: tuple[Path, ...] | None = None,
    dms_base_construct_subject_id: str = _DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
    rt_cds_positions: tuple[int, ...] = (),
    max_dms_variants: int | None = None,
) -> UnifiedConstructSubjectMaterializationReport:
    """Materialize all first-class RT-lnRNA construct subjects into one Construct output dataset."""
    root = _resolve_repo_root(repo_root)
    manifest = _load_projection_manifest(root)
    _require_valid_projection_manifest(manifest)
    authority = _require_genbank_authority(root)
    template_sequence = _template_sequence(manifest=manifest, authority=authority)
    target_start, target_end = _target_context_bounds(manifest)
    template_context_sequence = template_sequence[target_start:target_end]
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))

    rows: list[dict[str, object]] = []
    expected_sequences: dict[str, str] = {}
    run_groups: list[_ConstructSubjectRunGroup] = []
    genbank_construct_subject_count = 0
    crawford_construct_subject_count = 0
    khan_construct_subject_count = 0
    msd_compiler_construct_subject_count = 0
    rt_cds_dms_construct_subject_count = 0
    permuter_request_id: str | None = None
    source_promotion_report: SourcePromotionReport | None = None

    if include_genbank_catalog:
        catalog = build_variant_genbank_catalog(repo_root=root)
        if catalog.catalog_id != _GENBANK_CATALOG_SOURCE_COLLECTION_ID:
            raise MaterializationContractError(
                f"Unexpected GenBank catalog id: {catalog.catalog_id}; expected {_GENBANK_CATALOG_SOURCE_COLLECTION_ID}"
            )
        if not catalog.ok:
            joined = "; ".join(catalog.errors)
            raise MaterializationContractError(f"Variant GenBank catalog is invalid: {joined}")
        catalog_candidates = _catalog_materialization_candidates(
            repo_root=root,
            catalog_genbank_dir=Path(catalog.genbank_dir),
            records=catalog.records,
            target_start=target_start,
            target_end=target_end,
        )
        catalog_rows, catalog_expected = _catalog_candidate_rows(
            manifest=manifest,
            template_sequence=template_sequence,
            target_start=target_start,
            target_end=target_end,
            candidates=catalog_candidates,
        )
        _extend_construct_subject_rows(rows, catalog_rows)
        expected_sequences.update(catalog_expected)
        genbank_construct_subject_count = len(catalog_rows)
        for group_index, (window_offset_bp, group) in enumerate(
            _group_by_window_offset(catalog_candidates).items(),
            start=1,
        ):
            run_groups.append(
                _ConstructSubjectRunGroup(
                    name=f"genbank_offset_{group_index:02d}",
                    subject_ids=tuple(candidate.construct_subject_id for candidate in group),
                    window_offset_bp=window_offset_bp,
                )
            )

    if include_source_promotions:
        source_rows = rows
        if not source_rows:
            source_rows, _expected_control_sequences = _candidate_rows(
                manifest=manifest,
                authority=authority,
                template_sequence=template_sequence,
                target_start=target_start,
                target_end=target_end,
                construct_subject_sequence_overrides={},
                omitted_construct_subject_fields=set(),
            )
        wt_parent = _construct_subject_row_by_id_or_control(
            source_rows,
            construct_subject_id=_DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
            manifest=manifest,
            authority=authority,
            template_sequence=template_sequence,
            target_start=target_start,
            target_end=target_end,
        )
        source_promotion_report = resolve_source_construct_subject_promotions(
            dnadesign_data_root=(root / (dnadesign_data_root or _DEFAULT_DNADESIGN_DATA_ROOT)).resolve(),
            wt_rt_cds_sequence=_required_candidate_sequence(wt_parent, "construct_subject__rt_cds_sequence"),
            window_policy=_source_promotion_window_policy(
                manifest=manifest,
                template_sequence=template_sequence,
                target_start=target_start,
                target_end=target_end,
            ),
            source_record_resolver=source_record_resolver,
        )
        source_promotion_rows, source_promotion_expected = _source_promotion_rows(
            manifest=manifest,
            template_sequence=template_sequence,
            target_start=target_start,
            target_end=target_end,
            promotions=source_promotion_report.candidates,
        )
        if source_promotion_rows:
            _extend_construct_subject_rows(rows, source_promotion_rows)
            expected_sequences.update(source_promotion_expected)
            by_basis = source_promotion_report.candidates_by_basis
            crawford_construct_subject_count = int(by_basis.get("crawford_eco1_lnrna_fixed_wt_rt", 0))
            khan_construct_subject_count = int(by_basis.get("khan_source_rt_lnrna_reference", 0))
            for basis, group in _group_source_promotion_rows_by_basis(source_promotion_rows).items():
                run_groups.append(
                    _ConstructSubjectRunGroup(
                        name=basis,
                        subject_ids=tuple(str(row["id"]) for row in group),
                        window_offset_bp=None,
                    )
                )

    if include_msd_compiler_promotions:
        source_rows = rows
        if not source_rows:
            source_rows, _expected_control_sequences = _candidate_rows(
                manifest=manifest,
                authority=authority,
                template_sequence=template_sequence,
                target_start=target_start,
                target_end=target_end,
                construct_subject_sequence_overrides={},
                omitted_construct_subject_fields=set(),
            )
        wt_parent = _construct_subject_row_by_id_or_control(
            source_rows,
            construct_subject_id=_DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID,
            manifest=manifest,
            authority=authority,
            template_sequence=template_sequence,
            target_start=target_start,
            target_end=target_end,
        )
        pool_spec_paths = msd_variant_pool_spec_paths or (_DEFAULT_MSD_COMPILER_POOL_SPEC,)
        msd_promotions: list[SourceConstructSubjectPromotion] = []
        for pool_spec_path in pool_spec_paths:
            msd_promotions.extend(
                resolve_msd_compiler_promotions(
                    repo_root=root,
                    pool_spec_path=root / pool_spec_path,
                    wt_rt_cds_sequence=_required_candidate_sequence(wt_parent, "construct_subject__rt_cds_sequence"),
                    window_policy=_source_promotion_window_policy(
                        manifest=manifest,
                        template_sequence=template_sequence,
                        target_start=target_start,
                        target_end=target_end,
                    ),
                )
            )
        reject_duplicate_msd_compiler_lnrna_sequences(msd_promotions)
        msd_rows, msd_expected = _source_promotion_rows(
            manifest=manifest,
            template_sequence=template_sequence,
            target_start=target_start,
            target_end=target_end,
            promotions=tuple(msd_promotions),
        )
        if msd_rows:
            _extend_construct_subject_rows(rows, msd_rows)
            expected_sequences.update(msd_expected)
            msd_compiler_construct_subject_count = len(msd_rows)
            run_groups.append(
                _ConstructSubjectRunGroup(
                    name="compiler_generated_msd_lnrna_variant",
                    subject_ids=tuple(str(row["id"]) for row in msd_rows),
                    window_offset_bp=None,
                )
            )

    if include_rt_cds_dms:
        source_rows = rows
        if not source_rows:
            source_rows, _expected_control_sequences = _candidate_rows(
                manifest=manifest,
                authority=authority,
                template_sequence=template_sequence,
                target_start=target_start,
                target_end=target_end,
                construct_subject_sequence_overrides={},
                omitted_construct_subject_fields=set(),
            )
        parent = _construct_subject_row_by_id(source_rows, construct_subject_id=dms_base_construct_subject_id)
        request = CodingDnaDmsRequest(
            ref_name=f"{dms_base_construct_subject_id}__rt_cds",
            sequence=_required_candidate_sequence(parent, "construct_subject__rt_cds_sequence"),
            codon_table=default_codon_table_path("ecoli"),
            positions=rt_cds_positions,
            max_variants=max_dms_variants,
            metadata={
                "study_id": str(manifest["study_id"]),
                "construct_contract": str(manifest["construct_contract"]),
                "representation_contract": str(manifest["representation_contract"]),
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
                template_sequence=template_sequence,
                slots=slots,
                row=row,
                target_start=target_start,
                target_end=target_end,
            )
        run_groups.append(
            _ConstructSubjectRunGroup(
                name="rt_cds_dms",
                subject_ids=tuple(str(row["id"]) for row in dms_rows),
                window_offset_bp=None,
            )
        )
        rt_cds_dms_construct_subject_count = len(dms_rows)
        permuter_request_id = result.request_id

    if not rows:
        raise MaterializationContractError("Unified construct-subject materialization selected no input rows.")

    work = Path(work_root).resolve()
    usr_root = work / "usr"
    config_dir = work / "construct_configs"
    usr_root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    input_ids_by_subject_id = _write_construct_subject_dataset(usr_root=usr_root, rows=rows)

    run_results: list[RunResult] = []
    config_paths: list[Path] = []
    for group in run_groups:
        context_config = _construct_config(
            manifest=manifest,
            template_sequence=template_sequence,
            usr_root=usr_root,
            input_ids_by_subject_id=input_ids_by_subject_id,
            job_id=f"rt_lnrna_unified_{group.name}_context_views",
            output_on_conflict="error",
            output_variants=_context_output_variants(),
            construct_subject_ids=group.subject_ids,
            window_offset_bp=group.window_offset_bp,
        )
        slot_anchor_config = _construct_config(
            manifest=manifest,
            template_sequence=template_sequence,
            usr_root=usr_root,
            input_ids_by_subject_id=input_ids_by_subject_id,
            job_id=f"rt_lnrna_unified_{group.name}_slot_anchor_views",
            output_on_conflict="ignore",
            output_variants=_slot_anchor_output_variants(),
            construct_subject_ids=group.subject_ids,
            window_offset_bp=group.window_offset_bp,
        )
        context_path = _write_config(config_dir / f"construct-{group.name}-context-views.yaml", context_config)
        slot_anchor_path = _write_config(
            config_dir / f"construct-{group.name}-slot-anchor-views.yaml",
            slot_anchor_config,
        )
        config_paths.extend([context_path, slot_anchor_path])
        run_results.extend([run_from_config(context_path), run_from_config(slot_anchor_path)])

    _write_construct_output_subject_bridge(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    _require_construct_infer_ready(
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
    )
    return UnifiedConstructSubjectMaterializationReport(
        usr_root=usr_root,
        input_dataset=_INPUT_DATASET,
        output_dataset=_OUTPUT_DATASET,
        input_ids_by_subject_id=input_ids_by_subject_id,
        config_paths=tuple(config_paths),
        run_results=tuple(run_results),
        template_sequence=template_sequence,
        template_context_sequence=template_context_sequence,
        expected_sequences=expected_sequences,
        genbank_construct_subject_count=genbank_construct_subject_count,
        crawford_construct_subject_count=crawford_construct_subject_count,
        khan_construct_subject_count=khan_construct_subject_count,
        msd_compiler_construct_subject_count=msd_compiler_construct_subject_count,
        rt_cds_dms_construct_subject_count=rt_cds_dms_construct_subject_count,
        permuter_request_id=permuter_request_id,
        source_promotion_report=source_promotion_report,
    )
