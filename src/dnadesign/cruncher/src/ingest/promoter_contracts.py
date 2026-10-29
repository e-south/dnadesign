"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/ingest/promoter_contracts.py

Promoter ingest data contracts for Cruncher source normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

PROMOTER_EXPORT_SCHEMA_VERSION = "cruncher.regulondb.promoter_export.v2"
PROMOTER_PARSER_VERSION = "cruncher.regulondb.promoter_parser.v1"
_STRICT_DNA_RE = re.compile(r"^[ACGT]+$")
_RELATION_NAMES = (
    "promoter_aliases",
    "sigma_affiliations",
    "regulatory_interactions",
    "tfbs_sites",
    "promoter_boxes",
    "evidence_citations",
    "coordinate_features",
    "source_conflicts",
    "source_rows",
)


class PromoterSchemaError(ValueError):
    """Raised when a source payload cannot satisfy the promoter contract."""


@dataclass(frozen=True, slots=True)
class PromoterQuery:
    source_release_policy: str = "reported"
    source_release: str | None = None
    routes: tuple[str, ...] = ()
    limit: int | None = None
    page_size: int = 100
    include_relations: bool = True
    timeout_seconds: int = 30
    source_stratum: str = "curated"


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    source: str
    source_release: str
    source_route: str
    fetched_at: datetime
    raw_payload_sha256: str
    query_sha256: str
    parser_version: str = PROMOTER_PARSER_VERSION
    export_schema_version: str = PROMOTER_EXPORT_SCHEMA_VERSION
    source_release_date: str | None = None
    source_url: str | None = None
    raw_payload_ref: str | None = None
    source_table: str | None = None
    source_stratum: str = "curated"


@dataclass(frozen=True, slots=True)
class PromoterDescriptor:
    source: str
    source_release: str
    source_route: str
    promoter_id: str
    promoter_name: str | None
    sequence_present: bool
    tss_present: bool
    sigma_present: bool
    confidence_present: bool
    box_annotation_present: bool
    sigma_factor_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromoterSourceFile:
    source_id: str
    source: str
    release: str
    path: str
    table: str
    stratum: str
    role: str
    file_format: str
    parser_hint: str
    creates_base_rows: bool


@dataclass(frozen=True, slots=True)
class PromoterAssociationSourceFile:
    source_id: str
    source: str
    release: str
    path: str
    table: str
    stratum: str
    role: str
    file_format: str
    parser_hint: str


@dataclass(frozen=True, slots=True)
class SkippedPromoterSourceRow:
    source: str
    source_release: str
    source_route: str
    source_table: str | None
    source_stratum: str
    promoter_id: str | None
    promoter_name: str | None
    raw_sequence: str | None
    skip_reason: str
    source_row_ref: str
    raw_payload_sha256: str
    query_sha256: str
    parser_version: str = PROMOTER_PARSER_VERSION
    export_schema_version: str = PROMOTER_EXPORT_SCHEMA_VERSION
    source_release_date: str | None = None


@dataclass(frozen=True, slots=True)
class PromoterBox:
    kind: str
    sequence: str | None
    raw_coordinates: dict[str, Any]
    interval_0based: tuple[int, int] | None
    strand: str | None
    source_route: str


@dataclass(frozen=True, slots=True)
class PromoterSigmaAffiliation:
    sigma_id: str | None
    name: str | None
    abbrev: str | None
    gene_id: str | None
    gene_name: str | None
    source_route: str
    evidence: tuple[str, ...] = ()
    confidence: str | None = None
    citation_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromoterRegulatorySite:
    regulatory_interaction_id: str | None
    binding_site_id: str | None
    regulator_id: str | None
    regulator_name: str | None
    regulator_abbrev: str | None
    regulon_id: str | None
    regulon_name: str | None
    target_type: str | None
    function: str | None
    mechanism: str | None
    raw_coordinates: dict[str, Any]
    interval_0based: tuple[int, int] | None
    strand: str | None
    sequence: str | None
    confidence: str | None
    evidence: tuple[str, ...] = ()
    citation_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PromoterRegulatoryAssociation:
    source: str
    source_release: str
    source_route: str
    source_table: str | None
    source_stratum: str
    source_row_ref: str
    regulatory_interaction_id: str
    promoter_id: str | None
    promoter_name: str | None
    regulated_entity_name: str | None
    regulator_id: str | None
    regulator_name: str | None
    regulator_abbrev: str | None
    regulon_id: str | None
    regulon_name: str | None
    target_type: str | None
    function: str | None
    mechanism: str | None
    confidence: str | None
    evidence: tuple[str, ...] = ()
    citation_refs: tuple[str, ...] = ()
    binding_site_id: str | None = None
    binding_site_sequence: str | None = None
    binding_site_strand: str | None = None
    binding_interval_0based: tuple[int, int] | None = None
    binding_raw_coordinates: dict[str, Any] | None = None
    raw_payload_sha256: str | None = None
    query_sha256: str | None = None
    parser_version: str = PROMOTER_PARSER_VERSION
    export_schema_version: str = PROMOTER_EXPORT_SCHEMA_VERSION
    source_release_date: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionUnitRef:
    tu_id: str | None
    name: str | None


@dataclass(frozen=True, slots=True)
class OperonRef:
    operon_id: str | None
    name: str | None


@dataclass(frozen=True, slots=True)
class GeneRef:
    gene_id: str | None
    name: str | None


@dataclass(frozen=True, slots=True)
class PromoterRecord:
    source: str
    source_release: str
    source_route: str
    promoter_id: str
    promoter_name: str | None
    sequence: str
    raw_sequence: str
    sequence_case_policy: str
    sequence_length: int
    strand: str | None
    genome_accession: str | None
    tss_position_raw: str | None
    tss_interval_0based: tuple[int, int] | None
    confidence_level: str | None
    score: float | None
    evidence: tuple[str, ...]
    citations: tuple[str, ...]
    sigma_affiliations: tuple[PromoterSigmaAffiliation, ...]
    boxes: tuple[PromoterBox, ...]
    regulatory_sites: tuple[PromoterRegulatorySite, ...]
    transcription_units: tuple[TranscriptionUnitRef, ...]
    operon: OperonRef | None
    first_gene: GeneRef | None
    provenance: SourceProvenance


@dataclass(frozen=True, slots=True)
class PromoterSourceInventory:
    source_releases: tuple[str, ...]
    source_routes: tuple[str, ...]
    promoter_row_count: int
    sequence_present_rate: float
    promoter_id_present_rate: float
    tss_present_rate: float
    sigma_present_rate: float
    box_annotation_rate: float
    confidence_present_rate: float
    regulatory_context_rate: float
    duplicate_sequence_count: int
    conflict_count: int
    route_failure_count: int = 0


@dataclass(frozen=True, slots=True)
class PromoterCollectionSummary:
    record_count: int
    unique_promoter_count: int
    duplicate_promoter_id_count: int
    missing_sigma_count: int
    multi_sigma_count: int
    sigma_factor_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class PromoterSourceTriageReport:
    primary_source: str | None
    supplemental_sources: tuple[str, ...]
    blocked: bool
    blocking_reasons: tuple[str, ...]
    candidate_status: dict[str, str]


@dataclass(frozen=True, slots=True)
class PromoterExportManifest:
    schema_version: str
    parser_version: str
    export_created_at: datetime
    complete: bool
    record_count: int
    source_selection_status: str
    source_inventory: PromoterSourceInventory
    query: PromoterQuery
    artifacts: dict[str, str]
