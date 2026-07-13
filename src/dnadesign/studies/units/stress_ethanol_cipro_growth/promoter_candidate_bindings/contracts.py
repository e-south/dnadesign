"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/contracts.py

Contracts for study-owned promoter candidate identity bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STUDY_ID = "stress_ethanol_cipro_growth"
SCHEMA_ID = "dnadesign.study.promoter_candidate_bindings.v1"
SCHEMA_VERSION = "1"
BINDINGS_RECORD_ID = "promoter_candidate_bindings/bindings"
BINDINGS_FILENAME = "bindings.parquet"
READER_ALIAS_NAMESPACE = "reader.design_id"
SYNTHESIS_ALIAS_NAMESPACE = "synthesis.name"
SOURCE_ALIAS_NAMESPACE = "source.alias"

DENSEGEN_RENDER_ANNOTATION_KEYS: tuple[str, ...] = (
    "part_kind",
    "role",
    "constraint_name",
    "sequence",
    "core_sequence",
    "variant_id",
    "spacer_length",
    "placement_index",
    "part_index",
    "regulator",
    "motif_id",
    "tfbs_id",
    "orientation",
    "offset",
    "offset_raw",
    "length",
    "end",
    "pad_left",
    "site_id",
)
GENBANK_RENDER_ANNOTATION_KEYS: tuple[str, ...] = (
    "feature_id",
    "feature_order",
    "feature_type",
    "label",
    "role_hint",
    "location_raw",
    "start_0",
    "end_0",
    "strand",
    "confidence",
)


class PromoterCandidateBindingsError(ValueError):
    """Raised when candidate identity or binding provenance violates the contract."""


@dataclass(frozen=True)
class ExactPromoterCandidateIdentity:
    """One exact namespace-qualified alias bound to canonical candidate identity."""

    alias_namespace: str
    alias: str
    candidate_id: str
    canonical_sequence: str
    sequence_sha256: str
    binding_status: str = "resolved"
    binding_method: str = "exact_alias"


@dataclass(frozen=True)
class BindingSourceArtifact:
    artifact_id: str
    path: str
    sha256: str


@dataclass(frozen=True)
class PromoterCandidateBindingsPreview:
    bindings: object
    candidate_table_id: str
    candidate_selection_sha256: str
    source_artifacts: tuple[BindingSourceArtifact, ...]


@dataclass(frozen=True)
class PromoterCandidateBindingsWriteResult:
    manifest_json: Path
    bindings_parquet: Path
    binding_count: int
    candidate_count: int


@dataclass(frozen=True)
class PromoterCandidateBindingsVerification:
    schema_id: str
    schema_version: str
    study_id: str
    binding_count: int
    candidate_count: int
    manifest_json: Path
    bindings_parquet: Path
