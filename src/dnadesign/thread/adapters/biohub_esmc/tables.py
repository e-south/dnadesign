"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/tables.py

Parquet writers and validators for Biohub ESMC SAE artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections import Counter
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


def write_biohub_esmc_feature_catalog(
    path: Path,
    feature_catalog_rows: Sequence[Mapping[str, Any]],
    *,
    request_hash: str,
) -> Path:
    """Write only the Biohub ESMC feature-catalog table."""

    _write_table(
        path,
        _deduplicate_feature_rows(feature_catalog_rows),
        schema=_FEATURE_CATALOG_SCHEMA,
        schema_id=BIOHUB_ESMC_FEATURE_CATALOG_SCHEMA_ID,
        request_hash=request_hash,
    )
    return path


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
    candidate_ids = [str(row["candidate_id"]) for row in profile_rows]
    observed = set(candidate_ids)
    for candidate_id, count in sorted(Counter(candidate_ids).items()):
        if count > 1:
            issues.append(
                BiohubEsmcIssue(
                    check_id="thread.biohub_esmc.profile_duplicate_candidate_id",
                    message=f"Biohub ESMC profile contains {count} rows for candidate id {candidate_id!r}",
                    path=str(artifacts.profile_path),
                )
            )
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
            issues.extend(_validate_accepted_profile_shape(row, path=row_path))
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
    issues.extend(
        _validate_feature_table_candidate_hashes(
            artifacts.protein_features_path,
            profile_rows,
            table_label="protein_features",
            expected_count_field="protein_feature_count",
        )
    )
    issues.extend(
        _validate_feature_table_candidate_hashes(
            artifacts.residue_features_path,
            profile_rows,
            table_label="residue_features",
            expected_count_field="residue_feature_count",
        )
    )
    if not issues:
        issues.extend(_validate_residue_feature_shape(artifacts.residue_features_path, profile_rows))
    return issues


def _validate_accepted_profile_shape(row: Mapping[str, Any], *, path: str) -> list[BiohubEsmcIssue]:
    issues: list[BiohubEsmcIssue] = []
    sequence_length = int(row.get("sequence_length") or 0)
    token_count = int(row.get("token_count") or 0)
    if token_count not in {sequence_length, sequence_length + 2}:
        issues.append(
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.profile_token_count_mismatch",
                message=(
                    "Accepted Biohub ESMC rows must have token_count equal to sequence_length or sequence_length + 2"
                ),
                path=path,
            )
        )
    expected_k, expected_codebook = _parse_sae_model_shape(str(row.get("sae_model") or ""))
    feature_dictionary_size = int(row.get("feature_dictionary_size") or 0)
    if expected_codebook is not None and feature_dictionary_size != expected_codebook:
        issues.append(
            BiohubEsmcIssue(
                check_id="thread.biohub_esmc.profile_codebook_mismatch",
                message=f"feature_dictionary_size must match SAE model codebook{expected_codebook}",
                path=path,
            )
        )
    if expected_k is not None:
        expected_rows = sequence_length * expected_k
        residue_feature_count = int(row.get("residue_feature_count") or 0)
        if residue_feature_count != expected_rows:
            issues.append(
                BiohubEsmcIssue(
                    check_id="thread.biohub_esmc.profile_residue_feature_count_mismatch",
                    message=f"residue_feature_count must equal sequence_length * k ({expected_rows})",
                    path=path,
                )
            )
    return issues


def _validate_residue_feature_shape(path: Path, profile_rows: Sequence[Mapping[str, Any]]) -> list[BiohubEsmcIssue]:
    accepted = {
        str(row["candidate_id"]): (
            int(row.get("sequence_length") or 0),
            _parse_sae_model_shape(str(row.get("sae_model") or ""))[0],
        )
        for row in profile_rows
        if str(row.get("status") or "") == "accepted"
    }
    if not accepted:
        return []
    table = pq.read_table(
        path,
        columns=["candidate_id", "residue_index_zero_based", "sequence_position_one_based", "feature_index"],
    )
    grouped = table.group_by(["candidate_id", "residue_index_zero_based", "sequence_position_one_based"]).aggregate(
        [("feature_index", "count")]
    )
    issues: list[BiohubEsmcIssue] = []
    observed_position_counts: dict[str, int] = {}
    for row in grouped.to_pylist():
        candidate_id = str(row["candidate_id"])
        if candidate_id not in accepted:
            continue
        sequence_length, expected_k = accepted[candidate_id]
        zero_based = int(row["residue_index_zero_based"])
        one_based = int(row["sequence_position_one_based"])
        if expected_k is not None:
            observed_position_counts[candidate_id] = observed_position_counts.get(candidate_id, 0) + 1
        if one_based != zero_based + 1 or zero_based < 0 or one_based < 1 or one_based > sequence_length:
            issues.append(
                BiohubEsmcIssue(
                    check_id="thread.biohub_esmc.residue_position_out_of_bounds",
                    message="Residue feature rows must be residue-only positions with aligned zero/one-based indices",
                    path=f"{path}:{candidate_id}:{zero_based}:{one_based}",
                )
            )
            continue
        if expected_k is not None and int(row["feature_index_count"]) != expected_k:
            issues.append(
                BiohubEsmcIssue(
                    check_id="thread.biohub_esmc.residue_active_feature_count_mismatch",
                    message=f"Each residue must have exactly k={expected_k} nonzero SAE features",
                    path=f"{path}:{candidate_id}:{one_based}",
                )
            )
    for candidate_id, (sequence_length, expected_k) in accepted.items():
        if expected_k is None:
            continue
        if observed_position_counts.get(candidate_id, 0) != sequence_length:
            issues.append(
                BiohubEsmcIssue(
                    check_id="thread.biohub_esmc.residue_position_coverage_mismatch",
                    message=f"Accepted Biohub ESMC rows must cover all {sequence_length} sequence residues",
                    path=f"{path}:{candidate_id}",
                )
            )
    return issues


def _validate_feature_table_candidate_hashes(
    path: Path,
    profile_rows: Sequence[Mapping[str, Any]],
    *,
    table_label: str,
    expected_count_field: str,
) -> list[BiohubEsmcIssue]:
    accepted_hashes: dict[str, str] = {}
    accepted_sae_models: dict[str, str] = {}
    expected_counts: dict[str, int] = {}
    for row in profile_rows:
        if str(row.get("status") or "") != "accepted":
            continue
        candidate_id = str(row["candidate_id"])
        accepted_hashes[candidate_id] = str(row["sequence_hash"])
        accepted_sae_models[candidate_id] = str(row["sae_model"])
        expected_counts[candidate_id] = int(row.get(expected_count_field) or 0)
    if not accepted_hashes:
        return []
    table = pq.read_table(path, columns=["candidate_id", "sequence_hash", "sae_model"])
    grouped = table.group_by(["candidate_id", "sequence_hash", "sae_model"]).aggregate([("sequence_hash", "count")])
    issues: list[BiohubEsmcIssue] = []
    observed_counts: dict[str, int] = {}
    for row in grouped.to_pylist():
        candidate_id = str(row["candidate_id"])
        observed_counts[candidate_id] = observed_counts.get(candidate_id, 0) + int(row["sequence_hash_count"])
        expected_hash = accepted_hashes.get(candidate_id)
        if expected_hash is None:
            issues.append(
                BiohubEsmcIssue(
                    check_id=f"thread.biohub_esmc.{table_label}_unexpected_candidate",
                    message=f"{table_label} contains rows for non-accepted candidate id {candidate_id!r}",
                    path=f"{path}:{candidate_id}",
                )
            )
            continue
        observed_hash = str(row["sequence_hash"])
        if observed_hash != expected_hash:
            issues.append(
                BiohubEsmcIssue(
                    check_id=f"thread.biohub_esmc.{table_label}_sequence_hash_mismatch",
                    message=f"{table_label} sequence_hash for {candidate_id!r} does not match the profile row",
                    path=f"{path}:{candidate_id}",
                )
            )
        observed_sae_model = str(row["sae_model"])
        if observed_sae_model != accepted_sae_models[candidate_id]:
            issues.append(
                BiohubEsmcIssue(
                    check_id=f"thread.biohub_esmc.{table_label}_sae_model_mismatch",
                    message=f"{table_label} sae_model for {candidate_id!r} does not match the profile row",
                    path=f"{path}:{candidate_id}",
                )
            )
    for candidate_id, expected_count in expected_counts.items():
        observed_count = observed_counts.get(candidate_id, 0)
        if observed_count != expected_count:
            issues.append(
                BiohubEsmcIssue(
                    check_id=f"thread.biohub_esmc.{table_label}_row_count_mismatch",
                    message=(
                        f"{table_label} row count for {candidate_id!r} must equal "
                        f"{expected_count_field} ({expected_count}); got {observed_count}"
                    ),
                    path=f"{path}:{candidate_id}",
                )
            )
    return issues


def _parse_sae_model_shape(sae_model: str) -> tuple[int | None, int | None]:
    match = re.search(r"-k(?P<k>\d+)-codebook(?P<codebook>\d+)", sae_model)
    if not match:
        return None, None
    return int(match.group("k")), int(match.group("codebook"))


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
