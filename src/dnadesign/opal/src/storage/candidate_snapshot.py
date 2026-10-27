"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/candidate_snapshot.py

Digest and schema contract for a campaign candidate/X Parquet snapshot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pyarrow.parquet as pq

from ..core.utils import OpalError, file_sha256

CANDIDATE_SNAPSHOT_FIELDS = {"path", "sha256", "row_count", "columns", "schema_sha256"}


@dataclass(frozen=True)
class VerifiedCandidateSnapshot:
    path: Path
    sha256: str
    row_count: int
    columns: tuple[str, ...]
    schema_sha256: str


def candidate_snapshot_record(path: Path, *, relative_path: str = "records.parquet") -> dict[str, object]:
    """Describe one complete candidate/X table for a promotion manifest."""

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "path": relative_path,
        "sha256": file_sha256(path),
        "row_count": int(parquet.metadata.num_rows),
        "columns": schema.names,
        "schema_sha256": sha256(schema.serialize().to_pybytes()).hexdigest(),
    }


def verify_candidate_snapshot(
    raw: object,
    *,
    root: Path,
    expected_path: str,
    id_column: str,
    x_column: str | None,
) -> VerifiedCandidateSnapshot:
    """Verify file, digest, schema, and configured candidate/X columns."""

    if not isinstance(raw, dict) or set(raw) != CANDIDATE_SNAPSHOT_FIELDS:
        raise OpalError(
            f"Observed-label promotion candidate_artifact fields must be exactly {sorted(CANDIDATE_SNAPSHOT_FIELDS)}."
        )
    expected_relative, expected_resolved = _resolve_relative(root, expected_path, field="candidate_path")
    artifact_relative, artifact_path = _resolve_relative(
        root,
        _required_string(raw, "path"),
        field="candidate_artifact.path",
    )
    if artifact_relative != expected_relative or artifact_path != expected_resolved:
        raise OpalError("Observed-label promotion candidate_artifact.path does not match the configured snapshot.")

    expected_sha256 = _required_sha256(raw, "sha256")
    expected_schema_sha256 = _required_sha256(raw, "schema_sha256")
    expected_row_count = raw.get("row_count")
    if isinstance(expected_row_count, bool) or not isinstance(expected_row_count, int) or expected_row_count < 1:
        raise OpalError("Observed-label promotion candidate_artifact.row_count must be a positive integer.")
    expected_columns = _columns(raw.get("columns"))
    required_columns = {str(id_column).strip()}
    if x_column is not None:
        required_columns.add(str(x_column).strip())
    if "" in required_columns or not required_columns.issubset(expected_columns):
        raise OpalError(
            "Observed-label promotion candidate_artifact.columns do not contain the configured candidate/X columns."
        )
    if not artifact_path.is_file():
        raise OpalError(f"Observed-label promotion candidate artifact not found: {artifact_path}")
    try:
        actual_sha256 = file_sha256(artifact_path)
        parquet = pq.ParquetFile(artifact_path)
        actual_row_count = int(parquet.metadata.num_rows)
        actual_columns = tuple(parquet.schema_arrow.names)
        actual_schema_sha256 = sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest()
    except Exception as exc:
        raise OpalError(f"Failed to inspect observed-label candidate artifact {artifact_path}: {exc}") from exc
    if actual_sha256 != expected_sha256:
        raise OpalError(
            "Observed-label promotion candidate artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}."
        )
    if actual_row_count != expected_row_count:
        raise OpalError(
            "Observed-label promotion candidate_artifact.row_count mismatch: "
            f"expected {expected_row_count}, found {actual_row_count}."
        )
    if actual_columns != expected_columns or actual_schema_sha256 != expected_schema_sha256:
        raise OpalError("Observed-label promotion candidate artifact schema identity mismatch.")
    return VerifiedCandidateSnapshot(
        path=artifact_path,
        sha256=actual_sha256,
        row_count=actual_row_count,
        columns=actual_columns,
        schema_sha256=actual_schema_sha256,
    )


def _resolve_relative(root: Path, value: str | Path, *, field: str) -> tuple[str, Path]:
    raw = str(value).strip()
    posix = PurePosixPath(raw)
    if not raw or "\\" in raw or posix.is_absolute() or PureWindowsPath(raw).is_absolute() or ".." in posix.parts:
        raise OpalError(f"Observed-label promotion {field} must remain within the USR dataset root.")
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(*posix.parts)).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise OpalError(f"Observed-label promotion {field} must remain within the USR dataset root.")
    return posix.as_posix(), resolved


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OpalError(f"Observed-label promotion candidate_artifact.{field} must be a non-empty string.")
    return value.strip()


def _required_sha256(payload: dict[str, Any], field: str) -> str:
    value = _required_string(payload, field)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise OpalError(f"Observed-label promotion candidate_artifact.{field} must be a lowercase SHA-256 digest.")
    return value


def _columns(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise OpalError("Observed-label promotion candidate_artifact.columns must be unique non-empty strings.")
    return tuple(value)


__all__ = [
    "CANDIDATE_SNAPSHOT_FIELDS",
    "VerifiedCandidateSnapshot",
    "candidate_snapshot_record",
    "verify_candidate_snapshot",
]
