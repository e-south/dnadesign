"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/construct_materialization.py

Construct materialization helpers for the RT-lnRNA sponging construct triage
study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pyarrow as pa
import yaml

from dnadesign.construct import RunResult, run_from_config
from dnadesign.permuter import (
    CodingDnaDmsRequest,
    CodingDnaDmsVariantMetadata,
    PermuterResult,
    default_codon_table_path,
    generate_variants,
)
from dnadesign.usr import BiopythonGenBankParser, Dataset, ensure_sequence_contract_namespaces

from .construct_projection import validate_projection_manifest_payload
from .genbank_authority import GenBankAuthorityAudit, run_default_authority_audit
from .source_promotions import (
    ConstructWindowPolicy,
    SourceConstructSubjectPromotion,
    SourcePromotionReport,
    resolve_msd_compiler_promotions,
    resolve_source_construct_subject_promotions,
)
from .variant_genbank_catalog import (
    ExtractedSequenceAuthority,
    VariantGenBankCatalogRecord,
    build_variant_genbank_catalog,
)

_STUDY_DIR = Path("docs/studies/rt_lnrna_sponging_construct_triage")
_PROJECTION_MANIFEST_PATH = _STUDY_DIR / "operations/contract/fixtures/construct/construct-projection-manifest.yaml"
_DEFAULT_MSD_COMPILER_POOL_SPEC = _STUDY_DIR / "operations/contract/fixtures/source-promotions/msd-compiler-pool.yaml"
_STUDY_ID = "rt_lnrna_sponging_construct_triage"
_INPUT_DATASET = "rt_lnrna_sponging_construct_triage_construct_slot_inputs_v1"
_OUTPUT_DATASET = "rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1"
_MATERIALIZATION_SOURCE = "rt_lnrna_sponging_construct_triage construct materialization"
_PAYLOAD_PROGRAM_ID = "tetO_sponging_v1"
_RT_CDS_DMS_SOURCE_BASIS = "in_silico_rt_cds_dms"
_GENBANK_CATALOG_SOURCE_BASIS = "genbank_variant_catalog"
_GENBANK_CATALOG_SOURCE_COLLECTION_ID = "rt_lnrna_sponging_construct_triage_retron_variant_genbank_catalog_v1"
_DEFAULT_DNADESIGN_DATA_ROOT = Path("../dnadesign-data")
_DEFAULT_DMS_BASE_CONSTRUCT_SUBJECT_ID = "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO"
_CONSTRUCT_SUBJECT_OVERLAY = "construct_subject"
_CONSTRUCT_SUBJECT_BIOLOGICAL_SEQUENCE_FIELDS = (
    "construct_subject__lnrna_sequence",
    "construct_subject__rt_cds_sequence",
)
_CONSTRUCT_SUBJECT_INT_FIELDS = {
    "construct_subject__msd_product_length_nt",
    "construct_subject__rt_cds_dms_aa_pos",
    "construct_subject__rt_cds_dms_codon_index",
    "construct_subject__source_record_count",
    "construct_subject__source_reference_record_count",
    "construct_subject__source_abundance_record_count",
}
_REQUIRED_SLOT_IDS = ("lnrna", "rt_cds")
_BASE_TEMPLATE_LNRNA_SPAN_0 = (186, 359)
_TARGET_CONTEXT_START_0 = 56
_SEQUENCE_ID_SOURCE_MAP = {
    "2000bp-region.gb": "dual_cassette_2000bp_region",
    "pes-retron-26.gb": "pes_retron_26_vector",
    "pes-retron-26-a1-a2.gb": "pes_retron_26_lnrna_a1_a2",
    "retron-eco1-rt.gb": "retron_eco1_rt",
    "pes-retron-43.gb": "pes_retron_43_vector",
}


class MaterializationContractError(ValueError):
    """Raised when study-owned projection inputs cannot safely materialize."""


@dataclass(frozen=True)
class ControlConstructMaterializationReport:
    usr_root: Path
    input_dataset: str
    output_dataset: str
    input_ids_by_subject_id: dict[str, str]
    config_paths: tuple[Path, ...]
    run_results: tuple[RunResult, ...]
    template_sequence: str
    template_context_sequence: str
    expected_sequences: dict[str, str]


@dataclass(frozen=True)
class RtCdsDmsConstructMaterializationReport(ControlConstructMaterializationReport):
    base_construct_subject_id: str
    permuter_request_id: str


@dataclass(frozen=True)
class UnifiedConstructSubjectMaterializationReport(ControlConstructMaterializationReport):
    genbank_construct_subject_count: int
    crawford_construct_subject_count: int
    khan_construct_subject_count: int
    msd_compiler_construct_subject_count: int
    rt_cds_dms_construct_subject_count: int
    permuter_request_id: str | None
    source_promotion_report: SourcePromotionReport | None


@dataclass(frozen=True)
class _CatalogMaterializationCandidate:
    construct_subject_id: str
    lnrna_sequence: str
    rt_cds_sequence: str
    window_start: int
    window_offset_bp: int
    source_variant_id: str
    source_variant_class: str
    reader_design_id: str
    lnrna_authority_kind: str
    rt_cds_authority_kind: str


@dataclass(frozen=True)
class _ConstructSubjectRunGroup:
    name: str
    subject_ids: tuple[str, ...]
    window_offset_bp: int | None


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
        output_variants=[
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "forward",
                "recommended_pooling": "seq_mean",
                "view_name": "dual_cassette_2000bp_seq_mean",
            },
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "reverse_complement",
                "recommended_pooling": "seq_mean",
                "view_name": "dual_cassette_2000bp_reverse_complement_seq_mean",
            },
        ],
    )
    slot_anchor_config = _construct_config(
        manifest=manifest,
        template_sequence=template_sequence,
        usr_root=usr_root,
        input_ids_by_subject_id=input_ids_by_subject_id,
        job_id="rt_lnrna_slot_anchor_views",
        output_on_conflict="ignore",
        output_variants=[
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "forward",
                "recommended_pooling": "anchor_mean",
                "anchor_part": "lnrna",
                "view_name": "lnrna_span_in_construct_anchor_mean",
            },
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "reverse_complement",
                "recommended_pooling": "anchor_mean",
                "anchor_part": "lnrna",
                "view_name": "lnrna_span_in_construct_reverse_complement_anchor_mean",
            },
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "forward",
                "recommended_pooling": "anchor_mean",
                "anchor_part": "rt_cds",
                "view_name": "rt_cds_span_in_construct_anchor_mean",
            },
            {
                "product_kind": "realized_context",
                "context_kind": "template_custom",
                "orientation": "reverse_complement",
                "recommended_pooling": "anchor_mean",
                "anchor_part": "rt_cds",
                "view_name": "rt_cds_span_in_construct_reverse_complement_anchor_mean",
            },
        ],
    )
    context_path = _write_config(config_dir / "construct-context-views.yaml", context_config)
    slot_anchor_path = _write_config(config_dir / "construct-slot-anchor-views.yaml", slot_anchor_config)
    context_result = run_from_config(context_path)
    slot_anchor_result = run_from_config(slot_anchor_path)
    _write_construct_output_subject_bridge(
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


def _load_projection_manifest(repo_root: Path) -> dict[str, object]:
    payload = yaml.safe_load((repo_root / _PROJECTION_MANIFEST_PATH).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MaterializationContractError("Construct projection manifest must be a mapping.")
    return payload


def _require_valid_projection_manifest(manifest: dict[str, object]) -> None:
    audit = validate_projection_manifest_payload(manifest)
    if not audit.ok:
        joined = "; ".join(audit.errors)
        raise MaterializationContractError(f"Construct projection manifest is invalid: {joined}")


def _require_genbank_authority(repo_root: Path) -> GenBankAuthorityAudit:
    audit = run_default_authority_audit(repo_root=repo_root)
    if not audit.ok:
        joined = "; ".join(audit.errors)
        raise MaterializationContractError(f"GenBank source authority is invalid: {joined}")
    return audit


def _template_sequence(*, manifest: dict[str, object], authority: GenBankAuthorityAudit) -> str:
    template = _mapping(manifest["construct_template"], label="construct_template")
    source_id = str(template["source_authority_id"])
    return authority.source(source_id).sequence


def _target_context_bounds(manifest: dict[str, object]) -> tuple[int, int]:
    template = _mapping(manifest["construct_template"], label="construct_template")
    target = _mapping(template["target_context"], label="construct_template.target_context")
    start = int(target["window_start_0"])
    end = int(target["window_end_0"])
    if end <= start:
        raise MaterializationContractError("target_context.window_end_0 must be greater than window_start_0.")
    expected_length = int(target.get("length_nt", end - start))
    if end - start != expected_length:
        raise MaterializationContractError("target_context window span must equal target_context.length_nt.")
    return start, end


def _source_promotion_window_policy(
    *,
    manifest: dict[str, object],
    template_sequence: str,
    target_start: int,
    target_end: int,
) -> ConstructWindowPolicy:
    template = _mapping(manifest["construct_template"], label="construct_template")
    target = _mapping(template["target_context"], label="construct_template.target_context")
    slots = {str(slot["slot_id"]): slot for slot in _list(manifest["slots"], label="slots")}
    missing = sorted(set(_REQUIRED_SLOT_IDS) - set(slots))
    if missing:
        joined = ", ".join(missing)
        raise MaterializationContractError(f"Source promotion window policy missing required slot(s): {joined}")
    return ConstructWindowPolicy(
        context_id=str(target["context_id"]),
        target_start_0=target_start,
        target_length_nt=target_end - target_start,
        template_length_nt=len(template_sequence),
        lnrna_template_span_0=_span_0(slots["lnrna"]["template_span_0"], label="lnrna.template_span_0"),
        rt_cds_template_span_0=_span_0(slots["rt_cds"]["template_span_0"], label="rt_cds.template_span_0"),
    )


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


def _write_construct_subject_dataset(*, usr_root: Path, rows: list[dict[str, object]]) -> dict[str, str]:
    field_names = _construct_subject_overlay_fields(rows)
    _ensure_construct_subject_overlay_namespace(usr_root, field_names=field_names)
    dataset = Dataset(usr_root, _INPUT_DATASET)
    dataset.init(source=_MATERIALIZATION_SOURCE, notes="Temp RT-lnRNA Construct materialization inputs.")
    carrier_sequences = [
        _construct_subject_envelope_carrier_sequence(index) for index, _row in enumerate(rows, start=1)
    ]
    add_result = dataset.add_sequences(
        carrier_sequences,
        bio_type="dna",
        alphabet="dna_4",
        source=_MATERIALIZATION_SOURCE,
    )
    input_ids_by_subject_id = {str(row["id"]): input_id for row, input_id in zip(rows, add_result.ids, strict=True)}
    input_ids = [input_ids_by_subject_id[str(row["id"])] for row in rows]
    columns: dict[str, pa.Array] = {
        "id": pa.array(input_ids, type=pa.string()),
        "construct_subject__id": pa.array([str(row["id"]) for row in rows], type=pa.string()),
    }
    for field_name in field_names:
        columns[field_name] = pa.array([row.get(field_name) for row in rows])
    dataset.write_overlay(_CONSTRUCT_SUBJECT_OVERLAY, pa.table(columns), key="id", overwrite=True)
    dataset.write_overlay(
        "usr_label",
        pa.table(
            {
                "id": pa.array(input_ids, type=pa.string()),
                "usr_label__primary": pa.array([str(row["id"]) for row in rows], type=pa.string()),
                "usr_label__aliases": pa.array([[] for _row in rows], type=pa.list_(pa.string())),
            }
        ),
        key="id",
        overwrite=True,
    )
    return input_ids_by_subject_id


def _construct_subject_envelope_carrier_sequence(index: int) -> str:
    if index < 1:
        raise MaterializationContractError("Construct-subject envelope carrier index must be positive.")
    alphabet = "ACGT"
    n = index - 1
    encoded: list[str] = []
    for _digit in range(10):
        n, remainder = divmod(n, len(alphabet))
        encoded.append(alphabet[remainder])
    if n:
        raise MaterializationContractError(
            "Construct-subject envelope carrier index exceeds synthetic policy capacity."
        )
    return "ACGT" + "".join(reversed(encoded))


def _write_construct_output_subject_bridge(
    *,
    usr_root: Path,
    input_ids_by_subject_id: Mapping[str, str],
) -> None:
    if not input_ids_by_subject_id:
        raise MaterializationContractError("Cannot bridge Construct outputs without construct-subject input ids.")

    input_dataset = Dataset(usr_root, _INPUT_DATASET)
    output_dataset = Dataset(usr_root, _OUTPUT_DATASET)
    input_frame = input_dataset.head(n=max(len(input_ids_by_subject_id) + 20, 1000))
    output_frame = output_dataset.head(n=max(len(input_ids_by_subject_id) * 4 + 20, 1000))
    construct_subject_columns = tuple(
        column for column in input_frame.columns if column.startswith("construct_subject__")
    )
    if "construct_subject__id" not in construct_subject_columns:
        raise MaterializationContractError("Construct input dataset is missing construct_subject__id overlay.")
    if output_frame.empty:
        raise MaterializationContractError("Construct output dataset has no rows to bridge.")

    input_by_id = {str(row["id"]): row for row in input_frame.to_dict(orient="records")}
    expected_input_ids = set(input_ids_by_subject_id.values())
    missing_inputs = sorted(expected_input_ids - set(input_by_id))
    if missing_inputs:
        raise MaterializationContractError(
            "Construct input dataset is missing construct subject bridge row(s): " + ", ".join(missing_inputs)
        )

    for construct_subject_id, input_id in input_ids_by_subject_id.items():
        input_construct_subject_id = str(input_by_id[input_id].get("construct_subject__id") or "")
        if input_construct_subject_id != construct_subject_id:
            raise MaterializationContractError(
                f"Construct input construct subject bridge mismatch for {input_id}: "
                f"expected {construct_subject_id}, found {input_construct_subject_id or '<missing>'}."
            )

    bridge_rows: list[dict[str, object]] = []
    seen_output_ids: set[str] = set()
    seen_input_ids: set[str] = set()
    for output_row in output_frame.to_dict(orient="records"):
        output_id = str(output_row.get("id") or "")
        input_id = str(output_row.get("construct__input_id") or "")
        if not output_id:
            raise MaterializationContractError("Construct output row is missing id.")
        if output_id in seen_output_ids:
            raise MaterializationContractError(
                f"Duplicate Construct output id while bridging construct subjects: {output_id}"
            )
        seen_output_ids.add(output_id)
        input_row = input_by_id.get(input_id)
        if input_row is None:
            raise MaterializationContractError(
                f"Construct output row {output_id} references unknown construct__input_id {input_id or '<missing>'}."
            )
        seen_input_ids.add(input_id)
        bridge_row = {"id": output_id}
        for column in construct_subject_columns:
            bridge_row[column] = _clean_overlay_value(input_row.get(column), field_name=column)
        bridge_row["construct_subject__record_kind"] = "construct_output"
        bridge_row["construct_subject__sequence_authority"] = "realized_construct_sequence"
        bridge_rows.append(bridge_row)

    missing_outputs = sorted(expected_input_ids - seen_input_ids)
    if missing_outputs:
        raise MaterializationContractError(
            "Construct output construct subject bridge has no realized output row(s) for input id(s): "
            + ", ".join(missing_outputs)
        )

    output_dataset.write_overlay(
        _CONSTRUCT_SUBJECT_OVERLAY,
        pa.table(
            {
                column: pa.array([row.get(column) for row in bridge_rows])
                for column in ("id", *construct_subject_columns)
            }
        ),
        key="id",
        overwrite=True,
    )
    with input_dataset.maintenance(reason="construct_output_subject_bridge_registry_refresh"):
        input_dataset.refresh_overlay_metadata(_CONSTRUCT_SUBJECT_OVERLAY)
        input_dataset.refresh_overlay_metadata("usr_label")
    with output_dataset.maintenance(reason="construct_output_subject_bridge_registry_refresh"):
        output_dataset.refresh_overlay_metadata(_CONSTRUCT_SUBJECT_OVERLAY)


def _clean_overlay_value(value: object, *, field_name: str) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is not None and field_name in _CONSTRUCT_SUBJECT_INT_FIELDS:
        return int(value)
    return value


def _construct_subject_overlay_fields(rows: list[dict[str, object]]) -> tuple[str, ...]:
    required = _CONSTRUCT_SUBJECT_BIOLOGICAL_SEQUENCE_FIELDS
    extras = tuple(
        sorted(
            {
                key
                for row in rows
                for key in row
                if key.startswith("construct_subject__") and key not in {*required, "construct_subject__id"}
            }
        )
    )
    return (*required, *extras)


def _ensure_construct_subject_overlay_namespace(usr_root: Path, *, field_names: tuple[str, ...]) -> None:
    ensure_sequence_contract_namespaces(usr_root)
    registry_path = usr_root / "registry.yaml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise MaterializationContractError(f"{registry_path}: expected registry mapping")
    namespaces = payload.setdefault("namespaces", {})
    if not isinstance(namespaces, dict):
        raise MaterializationContractError(f"{registry_path}: namespaces must be a mapping")
    namespaces[_CONSTRUCT_SUBJECT_OVERLAY] = {
        "owner": "study",
        "description": "RT-lnRNA construct subjects and their slot sequences.",
        "columns": [{"name": "construct_subject__id", "type": "string"}]
        + [{"name": field_name, "type": _construct_subject_field_type(field_name)} for field_name in field_names],
    }
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _construct_subject_field_type(field_name: str) -> str:
    if field_name in _CONSTRUCT_SUBJECT_INT_FIELDS:
        return "int64"
    if field_name in {"construct_subject__biological_sequence_fields", "construct_subject__permuter_modifications"}:
        return "list<string>"
    return "string"


def _context_output_variants() -> list[dict[str, object]]:
    return [
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "forward",
            "recommended_pooling": "seq_mean",
            "view_name": "dual_cassette_2000bp_seq_mean",
        },
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "reverse_complement",
            "recommended_pooling": "seq_mean",
            "view_name": "dual_cassette_2000bp_reverse_complement_seq_mean",
        },
    ]


def _slot_anchor_output_variants() -> list[dict[str, object]]:
    return [
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "forward",
            "recommended_pooling": "anchor_mean",
            "anchor_part": "lnrna",
            "view_name": "lnrna_span_in_construct_anchor_mean",
        },
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "reverse_complement",
            "recommended_pooling": "anchor_mean",
            "anchor_part": "lnrna",
            "view_name": "lnrna_span_in_construct_reverse_complement_anchor_mean",
        },
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "forward",
            "recommended_pooling": "anchor_mean",
            "anchor_part": "rt_cds",
            "view_name": "rt_cds_span_in_construct_anchor_mean",
        },
        {
            "product_kind": "realized_context",
            "context_kind": "template_custom",
            "orientation": "reverse_complement",
            "recommended_pooling": "anchor_mean",
            "anchor_part": "rt_cds",
            "view_name": "rt_cds_span_in_construct_reverse_complement_anchor_mean",
        },
    ]


def _construct_config(
    *,
    manifest: dict[str, object],
    template_sequence: str,
    usr_root: Path,
    input_ids_by_subject_id: Mapping[str, str],
    job_id: str,
    output_on_conflict: str,
    output_variants: list[dict[str, object]],
    construct_subject_ids: tuple[str, ...] | None = None,
    window_offset_bp: int | None = None,
) -> dict[str, object]:
    slots = tuple(_mapping(slot, label="slots[]") for slot in _list(manifest["slots"], label="slots"))
    target_start, target_end = _target_context_bounds(manifest)
    resolved_window_offset_bp = (
        _centered_window_offset_bp(slots=slots, target_start=target_start, target_end=target_end)
        if window_offset_bp is None
        else window_offset_bp
    )
    resolved_construct_subject_ids = construct_subject_ids or tuple(
        str(candidate["construct_subject_id"]) for candidate in _list(manifest["candidates"], label="candidates")
    )
    return {
        "job": {
            "id": job_id,
            "input": {
                "source": {
                    "kind": "usr",
                    "dataset": _INPUT_DATASET,
                    "root": str(usr_root),
                },
                "field": None,
                "ids": [input_ids_by_subject_id[subject_id] for subject_id in resolved_construct_subject_ids],
            },
            "template": {
                "id": str(
                    _mapping(manifest["construct_template"], label="construct_template")["construct_template_id"]
                ),
                "source": {
                    "kind": "literal",
                    "sequence": template_sequence,
                    "label": "genbank:pes-retron-26.gb#record",
                },
                "circular": True,
            },
            "parts": [_part_config(slot=slot, template_sequence=template_sequence) for slot in slots],
            "realize": {
                "mode": "window",
                "focal_part": "lnrna",
                "required_slots": list(_REQUIRED_SLOT_IDS),
                "window": {
                    "semantics": "fixed_total",
                    "reference": "center",
                    "direction": "symmetric",
                    "size_bp": target_end - target_start,
                    "offset_bp": resolved_window_offset_bp,
                },
            },
            "output_variants": output_variants,
            "output": {
                "record_source": _MATERIALIZATION_SOURCE,
                "on_conflict": output_on_conflict,
                "target": {
                    "kind": "usr",
                    "dataset": _OUTPUT_DATASET,
                    "root": str(usr_root),
                },
            },
        }
    }


def _centered_window_offset_bp(
    *,
    slots: tuple[dict[str, object], ...],
    target_start: int,
    target_end: int,
) -> int:
    lnrna_slot = next((slot for slot in slots if str(slot["slot_id"]) == "lnrna"), None)
    if lnrna_slot is None:
        raise MaterializationContractError("Centered RT-lnRNA window requires an lnrna slot.")
    lnrna_start, lnrna_end = _span_0(lnrna_slot["template_span_0"], label="lnrna.template_span_0")
    base_center = lnrna_start + ((lnrna_end - lnrna_start) // 2)
    window_length = target_end - target_start
    return target_start - (base_center - (window_length // 2))


def _candidate_window_bounds(
    *,
    slots: tuple[dict[str, object], ...],
    realized_spans: dict[str, tuple[int, int]],
    target_start: int,
    target_end: int,
) -> tuple[int, int]:
    lnrna_slot = next((slot for slot in slots if str(slot["slot_id"]) == "lnrna"), None)
    if lnrna_slot is None:
        raise MaterializationContractError("Centered RT-lnRNA window requires an lnrna slot.")
    base_start, base_end = _span_0(lnrna_slot["template_span_0"], label="lnrna.template_span_0")
    realized_start, realized_end = realized_spans["lnrna"]
    base_center = base_start + ((base_end - base_start) // 2)
    realized_center = realized_start + ((realized_end - realized_start) // 2)
    window_start = target_start + (realized_center - base_center)
    return window_start, window_start + (target_end - target_start)


def _part_config(*, slot: dict[str, object], template_sequence: str) -> dict[str, object]:
    start, end = _span_0(slot["template_span_0"], label=f"{slot['slot_id']}.template_span_0")
    return {
        "name": str(slot["slot_id"]),
        "role": str(slot["role"]),
        "sequence": {
            "source": "input_field",
            "field": str(slot["sequence_field"]),
        },
        "placement": {
            "kind": "replace",
            "orientation": str(slot["orientation"]),
            "locator": {
                "kind": "coordinates",
                "start": start,
                "end": end,
            },
            "guards": {
                "replaced_sequence": template_sequence[start:end],
                "replaced_span_bp": end - start,
            },
        },
    }


def _write_config(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MaterializationContractError(f"{label} must be a mapping.")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise MaterializationContractError(f"{label} must be a list.")
    return value


def _span_0(value: object, *, label: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise MaterializationContractError(f"{label} must be [start, end].")
    start = int(value[0])
    end = int(value[1])
    if start < 0 or end <= start:
        raise MaterializationContractError(f"{label} must be a valid zero-based half-open span.")
    return start, end


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")
