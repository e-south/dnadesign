"""Contracts and constants for RT-lnRNA Construct materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.construct import RunResult

from ..source_promotions import SourcePromotionReport

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
