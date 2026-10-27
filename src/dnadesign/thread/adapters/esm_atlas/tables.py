"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/esm_atlas/tables.py

Parquet writers and validators for normalized ESM Atlas artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.thread.adapters.esm_atlas.models import AtlasIssue

ATLAS_PROFILE_SCHEMA_ID = "thread.esm_atlas.semantic_profile"
ATLAS_PROTEIN_ACTIVATIONS_SCHEMA_ID = "thread.esm_atlas.protein_activations"
ATLAS_RESIDUE_ACTIVATIONS_SCHEMA_ID = "thread.esm_atlas.residue_activations"
ATLAS_FEATURE_CATALOG_SCHEMA_ID = "thread.esm_atlas.feature_catalog"
PROFILE_FILE_NAME = "atlas_semantic_profile.parquet"
PROTEIN_ACTIVATIONS_FILE_NAME = "atlas_protein_activations.parquet"
RESIDUE_ACTIVATIONS_FILE_NAME = "atlas_residue_activations.parquet"
FEATURE_CATALOG_FILE_NAME = "atlas_feature_catalog.parquet"
_ALLOWED_PROFILE_STATUSES = {"accepted", "errored"}


@dataclass(frozen=True)
class AtlasSemanticArtifacts:
    """Paths emitted by one ESM Atlas semantic-profile materialization."""

    profile_path: Path
    protein_activations_path: Path
    residue_activations_path: Path
    feature_catalog_path: Path


_PROFILE_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("source_request_hash", pa.string()),
        ("atlas_request_hash", pa.string()),
        ("atlas_query_hash", pa.string()),
        ("atlas_api_base_url", pa.string()),
        ("atlas_api_version", pa.string()),
        ("query_md5", pa.string()),
        ("atlas_hash", pa.string()),
        ("atlas_accession", pa.string()),
        ("atlas_source", pa.string()),
        ("sequence_length", pa.int64()),
        ("status", pa.string()),
        ("folded_on_demand", pa.bool_()),
        ("restricted_count", pa.int64()),
        ("top_feature_indices", pa.list_(pa.int64())),
        ("top_feature_values", pa.list_(pa.float64())),
        ("top_feature_labels", pa.list_(pa.string())),
        ("nearest_hits_json", pa.string()),
        ("raw_response_hash", pa.string()),
        ("retrieved_at", pa.string()),
        ("failure_reason", pa.string()),
    ]
)
_PROTEIN_ACTIVATIONS_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("feature_index", pa.int32()),
        ("value", pa.float64()),
    ]
)
_RESIDUE_ACTIVATIONS_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("residue_index_zero_based", pa.int32()),
        ("sequence_position_one_based", pa.int32()),
        ("feature_index", pa.int32()),
        ("value", pa.float64()),
    ]
)
_FEATURE_CATALOG_SCHEMA = pa.schema(
    [
        ("feature_index", pa.int32()),
        ("label", pa.string()),
        ("description", pa.string()),
        ("source_retrieved_at", pa.string()),
        ("raw_feature_hash", pa.string()),
    ]
)


def write_atlas_semantic_artifacts(
    *,
    output_root: Path,
    profile_rows: Sequence[Mapping[str, Any]],
    protein_activation_rows: Sequence[Mapping[str, Any]],
    residue_activation_rows: Sequence[Mapping[str, Any]],
    feature_catalog_rows: Sequence[Mapping[str, Any]],
    request_hash: str,
) -> AtlasSemanticArtifacts:
    """Write normalized Atlas semantic artifacts to compact Parquet tables."""

    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = AtlasSemanticArtifacts(
        profile_path=output_root / PROFILE_FILE_NAME,
        protein_activations_path=output_root / PROTEIN_ACTIVATIONS_FILE_NAME,
        residue_activations_path=output_root / RESIDUE_ACTIVATIONS_FILE_NAME,
        feature_catalog_path=output_root / FEATURE_CATALOG_FILE_NAME,
    )
    _write_table(
        artifacts.profile_path,
        profile_rows,
        schema=_PROFILE_SCHEMA,
        schema_id=ATLAS_PROFILE_SCHEMA_ID,
        request_hash=request_hash,
    )
    _write_table(
        artifacts.protein_activations_path,
        protein_activation_rows,
        schema=_PROTEIN_ACTIVATIONS_SCHEMA,
        schema_id=ATLAS_PROTEIN_ACTIVATIONS_SCHEMA_ID,
        request_hash=request_hash,
    )
    _write_table(
        artifacts.residue_activations_path,
        residue_activation_rows,
        schema=_RESIDUE_ACTIVATIONS_SCHEMA,
        schema_id=ATLAS_RESIDUE_ACTIVATIONS_SCHEMA_ID,
        request_hash=request_hash,
    )
    unique_feature_rows = _deduplicate_feature_rows(feature_catalog_rows)
    _write_table(
        artifacts.feature_catalog_path,
        unique_feature_rows,
        schema=_FEATURE_CATALOG_SCHEMA,
        schema_id=ATLAS_FEATURE_CATALOG_SCHEMA_ID,
        request_hash=request_hash,
    )
    return artifacts


def validate_atlas_semantic_artifacts(
    *,
    output_root: Path,
    expected_candidate_ids: set[str],
    request_hash: str,
    allow_fold_on_miss: bool = False,
) -> list[AtlasIssue]:
    """Validate normalized Atlas semantic artifacts without study-specific biology."""

    artifacts = AtlasSemanticArtifacts(
        profile_path=output_root / PROFILE_FILE_NAME,
        protein_activations_path=output_root / PROTEIN_ACTIVATIONS_FILE_NAME,
        residue_activations_path=output_root / RESIDUE_ACTIVATIONS_FILE_NAME,
        feature_catalog_path=output_root / FEATURE_CATALOG_FILE_NAME,
    )
    issues: list[AtlasIssue] = []
    table_specs = (
        (artifacts.profile_path, ATLAS_PROFILE_SCHEMA_ID, set(_PROFILE_SCHEMA.names)),
        (
            artifacts.protein_activations_path,
            ATLAS_PROTEIN_ACTIVATIONS_SCHEMA_ID,
            set(_PROTEIN_ACTIVATIONS_SCHEMA.names),
        ),
        (
            artifacts.residue_activations_path,
            ATLAS_RESIDUE_ACTIVATIONS_SCHEMA_ID,
            set(_RESIDUE_ACTIVATIONS_SCHEMA.names),
        ),
        (artifacts.feature_catalog_path, ATLAS_FEATURE_CATALOG_SCHEMA_ID, set(_FEATURE_CATALOG_SCHEMA.names)),
    )
    for path, schema_id, columns in table_specs:
        issues.extend(
            _validate_table_header(path, schema_id=schema_id, required_columns=columns, request_hash=request_hash)
        )
    if issues:
        return issues

    profile_rows = pq.read_table(artifacts.profile_path).to_pylist()
    observed_candidate_ids = {str(row["candidate_id"]) for row in profile_rows}
    missing = sorted(expected_candidate_ids - observed_candidate_ids)
    unexpected = sorted(observed_candidate_ids - expected_candidate_ids)
    if missing:
        issues.append(
            AtlasIssue(
                check_id="thread.esm_atlas.profile_missing_candidates",
                message=f"Atlas profile is missing candidate ids: {missing}",
                path=str(artifacts.profile_path),
            )
        )
    if unexpected:
        issues.append(
            AtlasIssue(
                check_id="thread.esm_atlas.profile_unexpected_candidates",
                message=f"Atlas profile contains unexpected candidate ids: {unexpected}",
                path=str(artifacts.profile_path),
            )
        )
    for index, row in enumerate(profile_rows):
        row_path = f"{artifacts.profile_path}:row[{index}]"
        status = str(row.get("status", ""))
        if status not in _ALLOWED_PROFILE_STATUSES:
            issues.append(
                AtlasIssue(
                    check_id="thread.esm_atlas.profile_invalid_status",
                    message=f"Atlas profile status must be one of {sorted(_ALLOWED_PROFILE_STATUSES)}",
                    path=row_path,
                )
            )
        if not _is_sha256_uri(str(row.get("atlas_query_hash", ""))):
            issues.append(
                AtlasIssue(
                    check_id="thread.esm_atlas.profile_missing_query_hash",
                    message="Atlas profile rows must carry atlas_query_hash",
                    path=row_path,
                )
            )
        if bool(row.get("folded_on_demand", False)) and not allow_fold_on_miss:
            issues.append(
                AtlasIssue(
                    check_id="thread.esm_atlas.profile_folded_on_demand",
                    message="Atlas profile rows must not use on-demand folding unless a later contract permits it",
                    path=row_path,
                )
            )
        if status == "accepted":
            if not _is_md5(str(row.get("atlas_hash", ""))):
                issues.append(
                    AtlasIssue(
                        check_id="thread.esm_atlas.profile_invalid_atlas_hash",
                        message="Accepted Atlas profile rows must carry a 32-character atlas_hash",
                        path=row_path,
                    )
                )
            if not _is_sha256_uri(str(row.get("raw_response_hash", ""))):
                issues.append(
                    AtlasIssue(
                        check_id="thread.esm_atlas.profile_missing_raw_hash",
                        message="Accepted Atlas profile rows must carry a raw_response_hash",
                        path=row_path,
                    )
                )
            if not row.get("top_feature_indices"):
                issues.append(
                    AtlasIssue(
                        check_id="thread.esm_atlas.profile_missing_features",
                        message="Accepted Atlas profile rows must carry top_feature_indices",
                        path=row_path,
                    )
                )
        elif not str(row.get("failure_reason", "")).strip():
            issues.append(
                AtlasIssue(
                    check_id="thread.esm_atlas.profile_missing_failure_reason",
                    message="Errored Atlas profile rows must carry failure_reason",
                    path=row_path,
                )
            )
    return issues


def _write_table(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: pa.Schema,
    schema_id: str,
    request_hash: str,
) -> None:
    metadata = {
        b"schema_id": schema_id.encode("utf-8"),
        b"schema_version": b"1",
        b"status": b"materialized",
        b"request_hash": request_hash.encode("utf-8"),
    }
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _validate_table_header(
    path: Path,
    *,
    schema_id: str,
    required_columns: set[str],
    request_hash: str,
) -> list[AtlasIssue]:
    if not path.exists():
        return [
            AtlasIssue(
                check_id="thread.esm_atlas.artifact_missing",
                message=f"Atlas semantic artifact is missing: {path.name}",
                path=str(path),
            )
        ]
    table = pq.read_table(path)
    missing_columns = sorted(required_columns - set(table.column_names))
    if missing_columns:
        return [
            AtlasIssue(
                check_id="thread.esm_atlas.artifact_missing_columns",
                message=f"Atlas artifact {path.name} is missing required columns: {missing_columns}",
                path=str(path),
            )
        ]
    issues: list[AtlasIssue] = []
    metadata = table.schema.metadata or {}
    if metadata.get(b"schema_id") != schema_id.encode("utf-8"):
        issues.append(
            AtlasIssue(
                check_id="thread.esm_atlas.artifact_schema_mismatch",
                message=f"Atlas artifact {path.name} must declare schema id {schema_id}",
                path=str(path),
            )
        )
    if metadata.get(b"request_hash") != request_hash.encode("utf-8"):
        issues.append(
            AtlasIssue(
                check_id="thread.esm_atlas.artifact_request_hash_mismatch",
                message=f"Atlas artifact {path.name} must carry request hash {request_hash}",
                path=str(path),
            )
        )
    return issues


def _deduplicate_feature_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        by_index.setdefault(int(row["feature_index"]), row)
    return [by_index[index] for index in sorted(by_index)]


def _is_md5(value: str) -> bool:
    return len(value) == 32 and all(character in "0123456789abcdef" for character in value.lower())


def _is_sha256_uri(value: str) -> bool:
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest.lower())
