"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/tables.py

Parquet writers and validators for Biohub ESMC SAE artifacts.

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
import yaml

from dnadesign.thread.adapters.biohub_esmc.models import BiohubEsmcIssue

BIOHUB_ESMC_PROFILE_SCHEMA_ID = "thread.biohub_esmc.sae_profile"
BIOHUB_ESMC_PROTEIN_FEATURES_SCHEMA_ID = "thread.biohub_esmc.protein_features"
BIOHUB_ESMC_RESIDUE_FEATURES_SCHEMA_ID = "thread.biohub_esmc.residue_features"
BIOHUB_ESMC_FEATURE_CATALOG_SCHEMA_ID = "thread.biohub_esmc.feature_catalog"
PROFILE_FILE_NAME = "biohub_esmc_sae_profile.parquet"
PROTEIN_FEATURES_FILE_NAME = "biohub_esmc_protein_features.parquet"
RESIDUE_FEATURES_FILE_NAME = "biohub_esmc_residue_features.parquet"
FEATURE_CATALOG_FILE_NAME = "biohub_esmc_feature_catalog.parquet"
REQUEST_MANIFEST_FILE_NAME = "biohub_esmc_request_manifest.yaml"
_ALLOWED_PROFILE_STATUSES = {"accepted", "errored"}


@dataclass(frozen=True)
class BiohubEsmcArtifacts:
    """Paths emitted by one Biohub ESMC materialization."""

    profile_path: Path
    protein_features_path: Path
    residue_features_path: Path
    feature_catalog_path: Path
    request_manifest_path: Path


_PROFILE_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("source_request_hash", pa.string()),
        ("biohub_request_hash", pa.string()),
        ("biohub_query_hash", pa.string()),
        ("biohub_api_base_url", pa.string()),
        ("biohub_api_version", pa.string()),
        ("model", pa.string()),
        ("sae_model", pa.string()),
        ("normalize_features", pa.bool_()),
        ("key_label", pa.string()),
        ("sequence_length", pa.int64()),
        ("token_count", pa.int64()),
        ("feature_dictionary_size", pa.int64()),
        ("status", pa.string()),
        ("protein_feature_count", pa.int64()),
        ("residue_feature_count", pa.int64()),
        ("encoded_sae_bytes", pa.int64()),
        ("raw_encode_response_hash", pa.string()),
        ("raw_logits_response_hash", pa.string()),
        ("retrieved_at", pa.string()),
        ("failure_reason", pa.string()),
    ]
)
_PROTEIN_FEATURES_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("sae_model", pa.string()),
        ("feature_index", pa.int32()),
        ("sequence_residue_count", pa.int32()),
        ("nonzero_residue_count", pa.int32()),
        ("activation_sum", pa.float64()),
        ("activation_mean", pa.float64()),
        ("activation_max", pa.float64()),
    ]
)
_RESIDUE_FEATURES_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("sae_model", pa.string()),
        ("residue_index_zero_based", pa.int32()),
        ("sequence_position_one_based", pa.int32()),
        ("feature_index", pa.int32()),
        ("value", pa.float64()),
    ]
)
_FEATURE_CATALOG_SCHEMA = pa.schema(
    [
        ("sae_model", pa.string()),
        ("feature_index", pa.int32()),
        ("label", pa.string()),
        ("description", pa.string()),
        ("source_retrieved_at", pa.string()),
        ("raw_feature_hash", pa.string()),
    ]
)


def write_biohub_esmc_artifacts(
    *,
    output_root: Path,
    profile_rows: Sequence[Mapping[str, Any]],
    protein_feature_rows: Sequence[Mapping[str, Any]],
    residue_feature_rows: Sequence[Mapping[str, Any]],
    feature_catalog_rows: Sequence[Mapping[str, Any]],
    request_manifest: Mapping[str, Any],
    request_hash: str,
) -> BiohubEsmcArtifacts:
    """Write normalized Biohub ESMC artifacts to compact Parquet tables."""

    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = BiohubEsmcArtifacts(
        profile_path=output_root / PROFILE_FILE_NAME,
        protein_features_path=output_root / PROTEIN_FEATURES_FILE_NAME,
        residue_features_path=output_root / RESIDUE_FEATURES_FILE_NAME,
        feature_catalog_path=output_root / FEATURE_CATALOG_FILE_NAME,
        request_manifest_path=output_root / REQUEST_MANIFEST_FILE_NAME,
    )
    _write_table(
        artifacts.profile_path,
        profile_rows,
        schema=_PROFILE_SCHEMA,
        schema_id=BIOHUB_ESMC_PROFILE_SCHEMA_ID,
        request_hash=request_hash,
    )
    _write_table(
        artifacts.protein_features_path,
        protein_feature_rows,
        schema=_PROTEIN_FEATURES_SCHEMA,
        schema_id=BIOHUB_ESMC_PROTEIN_FEATURES_SCHEMA_ID,
        request_hash=request_hash,
    )
    _write_table(
        artifacts.residue_features_path,
        residue_feature_rows,
        schema=_RESIDUE_FEATURES_SCHEMA,
        schema_id=BIOHUB_ESMC_RESIDUE_FEATURES_SCHEMA_ID,
        request_hash=request_hash,
    )
    _write_table(
        artifacts.feature_catalog_path,
        _deduplicate_feature_rows(feature_catalog_rows),
        schema=_FEATURE_CATALOG_SCHEMA,
        schema_id=BIOHUB_ESMC_FEATURE_CATALOG_SCHEMA_ID,
        request_hash=request_hash,
    )
    artifacts.request_manifest_path.write_text(
        yaml.safe_dump(dict(request_manifest), sort_keys=False), encoding="utf-8"
    )
    return artifacts


def validate_biohub_esmc_artifacts(
    *,
    output_root: Path,
    expected_candidate_ids: set[str],
    request_hash: str,
) -> list[BiohubEsmcIssue]:
    """Validate Biohub ESMC artifacts without study-specific interpretation."""

    artifacts = BiohubEsmcArtifacts(
        profile_path=output_root / PROFILE_FILE_NAME,
        protein_features_path=output_root / PROTEIN_FEATURES_FILE_NAME,
        residue_features_path=output_root / RESIDUE_FEATURES_FILE_NAME,
        feature_catalog_path=output_root / FEATURE_CATALOG_FILE_NAME,
        request_manifest_path=output_root / REQUEST_MANIFEST_FILE_NAME,
    )
    issues: list[BiohubEsmcIssue] = []
    for path, schema_id, columns in (
        (artifacts.profile_path, BIOHUB_ESMC_PROFILE_SCHEMA_ID, set(_PROFILE_SCHEMA.names)),
        (artifacts.protein_features_path, BIOHUB_ESMC_PROTEIN_FEATURES_SCHEMA_ID, set(_PROTEIN_FEATURES_SCHEMA.names)),
        (artifacts.residue_features_path, BIOHUB_ESMC_RESIDUE_FEATURES_SCHEMA_ID, set(_RESIDUE_FEATURES_SCHEMA.names)),
        (artifacts.feature_catalog_path, BIOHUB_ESMC_FEATURE_CATALOG_SCHEMA_ID, set(_FEATURE_CATALOG_SCHEMA.names)),
    ):
        issues.extend(
            _validate_table_header(path, schema_id=schema_id, required_columns=columns, request_hash=request_hash)
        )
    if not artifacts.request_manifest_path.exists():
        issues.append(
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.request_manifest_missing",
                message="Biohub ESMC request manifest is missing",
                path=str(artifacts.request_manifest_path),
            )
        )
    if issues:
        return issues
    profile_rows = pq.read_table(artifacts.profile_path).to_pylist()
    observed = {str(row["candidate_id"]) for row in profile_rows}
    missing = sorted(expected_candidate_ids - observed)
    unexpected = sorted(observed - expected_candidate_ids)
    if missing:
        issues.append(
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.profile_missing_candidates",
                message=f"Biohub ESMC profile is missing candidate ids: {missing}",
                path=str(artifacts.profile_path),
            )
        )
    if unexpected:
        issues.append(
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.profile_unexpected_candidates",
                message=f"Biohub ESMC profile contains unexpected candidate ids: {unexpected}",
                path=str(artifacts.profile_path),
            )
        )
    for index, row in enumerate(profile_rows):
        row_path = f"{artifacts.profile_path}:row[{index}]"
        status = str(row.get("status", ""))
        if status not in _ALLOWED_PROFILE_STATUSES:
            issues.append(
                BiohubEsmcIssue(
                    check_id="thread.biohub_esmc.profile_invalid_status",
                    message=f"Biohub ESMC status must be one of {sorted(_ALLOWED_PROFILE_STATUSES)}",
                    path=row_path,
                )
            )
        if not _is_sha256_uri(str(row.get("biohub_query_hash", ""))):
            issues.append(
                BiohubEsmcIssue(
                    check_id="thread.biohub_esmc.profile_missing_query_hash",
                    message="Biohub ESMC rows must carry biohub_query_hash",
                    path=row_path,
                )
            )
        if status == "accepted":
            if int(row.get("residue_feature_count") or 0) <= 0:
                issues.append(
                    BiohubEsmcIssue(
                        check_id="thread.biohub_esmc.profile_missing_residue_features",
                        message="Accepted Biohub ESMC rows must carry residue SAE features",
                        path=row_path,
                    )
                )
            if not _is_sha256_uri(str(row.get("raw_logits_response_hash", ""))):
                issues.append(
                    BiohubEsmcIssue(
                        check_id="thread.biohub_esmc.profile_missing_raw_logits_hash",
                        message="Accepted Biohub ESMC rows must carry raw_logits_response_hash",
                        path=row_path,
                    )
                )
        elif not str(row.get("failure_reason", "")).strip():
            issues.append(
                BiohubEsmcIssue(
                    check_id="thread.biohub_esmc.profile_missing_failure_reason",
                    message="Errored Biohub ESMC rows must carry failure_reason",
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
) -> list[BiohubEsmcIssue]:
    if not path.exists():
        return [
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.artifact_missing",
                message=f"Biohub ESMC artifact is missing: {path.name}",
                path=str(path),
            )
        ]
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    missing = sorted(required_columns - set(schema.names))
    if missing:
        return [
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.artifact_missing_columns",
                message=f"Biohub ESMC artifact {path.name} is missing required columns: {missing}",
                path=str(path),
            )
        ]
    issues: list[BiohubEsmcIssue] = []
    metadata = schema.metadata or {}
    if metadata.get(b"schema_id") != schema_id.encode("utf-8"):
        issues.append(
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.artifact_schema_mismatch",
                message=f"Biohub ESMC artifact {path.name} must declare schema id {schema_id}",
                path=str(path),
            )
        )
    if metadata.get(b"request_hash") != request_hash.encode("utf-8"):
        issues.append(
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.artifact_request_hash_mismatch",
                message=f"Biohub ESMC artifact {path.name} must carry request hash {request_hash}",
                path=str(path),
            )
        )
    return issues


def _deduplicate_feature_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        by_key.setdefault((str(row["sae_model"]), int(row["feature_index"])), row)
    return [by_key[key] for key in sorted(by_key)]


def _is_sha256_uri(value: str) -> bool:
    return (
        len(value) == len("sha256:") + 64
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value.removeprefix("sha256:"))
    )
