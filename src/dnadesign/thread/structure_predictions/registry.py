"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_predictions/registry.py

Parquet registry for model-predicted structures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.thread.structure_predictions.hashes import file_sha256_uri
from dnadesign.thread.structure_predictions.models import StructurePredictionArtifacts, StructurePredictionIssue

STRUCTURE_PREDICTION_REGISTRY_FILE_NAME = "structure_prediction_registry.parquet"
STRUCTURE_PREDICTION_REGISTRY_SCHEMA_ID = "thread.structure_predictions.registry"
_ALLOWED_STATUSES = {"accepted", "errored"}

_REGISTRY_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("sequence_hash", pa.string()),
        ("prediction_id", pa.string()),
        ("prediction_set_id", pa.string()),
        ("backend_kind", pa.string()),
        ("model_family", pa.string()),
        ("model_name", pa.string()),
        ("model_version", pa.string()),
        ("runtime_or_endpoint", pa.string()),
        ("parameters_hash", pa.string()),
        ("request_hash", pa.string()),
        ("source_request_hash", pa.string()),
        ("raw_response_hash", pa.string()),
        ("structure_hash", pa.string()),
        ("structure_source_uri", pa.string()),
        ("local_structure_path", pa.string()),
        ("plddt", pa.float64()),
        ("ptm", pa.float64()),
        ("pae_summary_hash", pa.string()),
        ("status", pa.string()),
        ("failure_reason", pa.string()),
    ]
)


def write_structure_prediction_registry(
    *,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    request_hash: str,
) -> StructurePredictionArtifacts:
    """Write a compact structure-prediction registry."""

    output_root.mkdir(parents=True, exist_ok=True)
    registry_path = output_root / STRUCTURE_PREDICTION_REGISTRY_FILE_NAME
    metadata = {
        b"schema_id": STRUCTURE_PREDICTION_REGISTRY_SCHEMA_ID.encode("utf-8"),
        b"schema_version": b"1",
        b"status": b"materialized",
        b"request_hash": request_hash.encode("utf-8"),
    }
    table = pa.Table.from_pylist(list(rows), schema=_REGISTRY_SCHEMA)
    pq.write_table(table.replace_schema_metadata(metadata), registry_path)
    return StructurePredictionArtifacts(registry_path=registry_path)


def validate_structure_prediction_registry(
    *,
    registry_path: Path,
    request_hash: str,
) -> list[StructurePredictionIssue]:
    """Validate a structure-prediction registry without study-specific policy."""

    if not registry_path.exists():
        return [
            StructurePredictionIssue(
                check_id="thread.structure_predictions.registry_missing",
                message="Structure-prediction registry is missing",
                path=str(registry_path),
            )
        ]
    table = pq.read_table(registry_path)
    issues: list[StructurePredictionIssue] = []
    metadata = table.schema.metadata or {}
    if metadata.get(b"schema_id") != STRUCTURE_PREDICTION_REGISTRY_SCHEMA_ID.encode("utf-8"):
        issues.append(
            StructurePredictionIssue(
                check_id="thread.structure_predictions.schema_mismatch",
                message=f"Registry must declare schema id {STRUCTURE_PREDICTION_REGISTRY_SCHEMA_ID}",
                path=str(registry_path),
            )
        )
    if metadata.get(b"request_hash") != request_hash.encode("utf-8"):
        issues.append(
            StructurePredictionIssue(
                check_id="thread.structure_predictions.request_hash_mismatch",
                message="Registry metadata request_hash must match the current request",
                path=str(registry_path),
            )
        )
    missing_columns = sorted(set(_REGISTRY_SCHEMA.names) - set(table.column_names))
    if missing_columns:
        issues.append(
            StructurePredictionIssue(
                check_id="thread.structure_predictions.missing_columns",
                message=f"Registry is missing required columns: {missing_columns}",
                path=str(registry_path),
            )
        )
        return issues
    if issues:
        return issues

    seen_prediction_ids: set[str] = set()
    for index, row in enumerate(table.to_pylist()):
        row_path = f"{registry_path}:row[{index}]"
        status = str(row.get("status", ""))
        if status not in _ALLOWED_STATUSES:
            issues.append(
                StructurePredictionIssue(
                    check_id="thread.structure_predictions.invalid_status",
                    message=f"Registry status must be one of {sorted(_ALLOWED_STATUSES)}",
                    path=row_path,
                )
            )
        prediction_id = str(row.get("prediction_id", ""))
        if not prediction_id.strip():
            issues.append(
                StructurePredictionIssue(
                    check_id="thread.structure_predictions.missing_prediction_id",
                    message="Every structure-prediction row must carry prediction_id",
                    path=row_path,
                )
            )
        elif prediction_id in seen_prediction_ids:
            issues.append(
                StructurePredictionIssue(
                    check_id="thread.structure_predictions.duplicate_prediction_id",
                    message=f"Duplicate prediction_id: {prediction_id}",
                    path=row_path,
                )
            )
        seen_prediction_ids.add(prediction_id)
        if str(row.get("request_hash", "")) != request_hash:
            issues.append(
                StructurePredictionIssue(
                    check_id="thread.structure_predictions.row_request_hash_mismatch",
                    message="Row request_hash must match registry metadata request_hash",
                    path=row_path,
                )
            )
        if status == "accepted":
            issues.extend(_validate_accepted_row(row, row_path=row_path))
        elif not str(row.get("failure_reason", "")).strip():
            issues.append(
                StructurePredictionIssue(
                    check_id="thread.structure_predictions.missing_failure_reason",
                    message="Errored structure-prediction rows must carry failure_reason",
                    path=row_path,
                )
            )
    return issues


def _validate_accepted_row(row: Mapping[str, Any], *, row_path: str) -> list[StructurePredictionIssue]:
    issues: list[StructurePredictionIssue] = []
    for field in (
        "candidate_id",
        "sequence_hash",
        "prediction_set_id",
        "backend_kind",
        "model_family",
        "model_name",
        "model_version",
        "runtime_or_endpoint",
        "parameters_hash",
        "raw_response_hash",
        "structure_hash",
    ):
        if not str(row.get(field, "")).strip():
            issues.append(
                StructurePredictionIssue(
                    check_id=f"thread.structure_predictions.missing_{field}",
                    message=f"Accepted structure-prediction rows must carry {field}",
                    path=row_path,
                )
            )
    local_path = str(row.get("local_structure_path", ""))
    if local_path:
        path = Path(local_path)
        if not path.exists():
            issues.append(
                StructurePredictionIssue(
                    check_id="thread.structure_predictions.local_structure_missing",
                    message="local_structure_path must exist when declared",
                    path=row_path,
                )
            )
        elif file_sha256_uri(path) != str(row.get("structure_hash", "")):
            issues.append(
                StructurePredictionIssue(
                    check_id="thread.structure_predictions.structure_hash_mismatch",
                    message="local_structure_path hash must match structure_hash",
                    path=row_path,
                )
            )
    return issues
