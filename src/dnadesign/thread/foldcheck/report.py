"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/foldcheck/report.py

Generic fold-check report writer and validator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.thread.foldcheck.models import FoldCheckIssue

FOLDCHECK_REPORT_SCHEMA_ID = "thread.foldcheck_report"
_REQUIRED_COLUMNS = {
    "candidate_id",
    "runtime_kind",
    "runtime_version",
    "input_sequence_hash",
    "reference_structure_id",
    "wt_baseline_artifact_id",
    "runtime_parameters_hash",
    "threshold_id",
    "threshold_values",
    "plddt",
    "pae_summary",
    "backbone_rmsd_to_reference",
    "protected_contact_retention",
    "status",
    "rejection_reason",
    "missing_metric_reason",
}
_ALLOWED_STATUSES = {"accepted", "rejected", "errored"}


def write_foldcheck_report(path: Path, rows: Sequence[Mapping[str, Any]], *, request_hash: str) -> None:
    """Write normalized fold-check rows to Parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    metadata = {
        b"schema_id": FOLDCHECK_REPORT_SCHEMA_ID.encode("utf-8"),
        b"schema_version": b"1",
        b"status": b"materialized",
        b"request_hash": request_hash.encode("utf-8"),
    }
    pq.write_table(table.replace_schema_metadata(metadata), path)


def validate_foldcheck_report(
    path: Path,
    *,
    request_hash: str,
    expected_candidate_ids: set[str] | None = None,
    wt_candidate_id: str = "wild_type",
) -> list[FoldCheckIssue]:
    """Validate a normalized fold-check report without study-specific biology."""

    table = pq.read_table(path)
    missing_columns = sorted(_REQUIRED_COLUMNS - set(table.column_names))
    if missing_columns:
        return [
            FoldCheckIssue(
                check_id="thread.foldcheck_report.missing_columns",
                message=f"Fold-check report is missing required columns: {missing_columns}",
                path=str(path),
            )
        ]

    issues: list[FoldCheckIssue] = []
    metadata = table.schema.metadata or {}
    if metadata.get(b"schema_id") != FOLDCHECK_REPORT_SCHEMA_ID.encode("utf-8"):
        issues.append(
            FoldCheckIssue(
                check_id="thread.foldcheck_report.metadata_mismatch",
                message="Fold-check report must declare the generic fold-check schema id",
                path=str(path),
            )
        )
    if metadata.get(b"request_hash") != request_hash.encode("utf-8"):
        issues.append(
            FoldCheckIssue(
                check_id="thread.foldcheck_report.request_hash_mismatch",
                message=f"Fold-check report metadata must carry request hash {request_hash}",
                path=str(path),
            )
        )

    rows = table.to_pylist()
    if not rows:
        issues.append(
            FoldCheckIssue(
                check_id="thread.foldcheck_report.empty",
                message="Fold-check report must contain at least a WT baseline row",
                path=str(path),
            )
        )
        return issues

    candidate_ids = {str(row["candidate_id"]) for row in rows}
    if wt_candidate_id not in candidate_ids:
        issues.append(
            FoldCheckIssue(
                check_id="thread.foldcheck_report.missing_wt_baseline",
                message=f"Fold-check report must include WT baseline candidate id {wt_candidate_id!r}",
                path=str(path),
            )
        )
    if expected_candidate_ids is not None:
        missing = sorted(expected_candidate_ids - candidate_ids)
        if missing:
            issues.append(
                FoldCheckIssue(
                    check_id="thread.foldcheck_report.missing_candidates",
                    message=f"Fold-check report is missing expected candidate ids: {missing}",
                    path=str(path),
                )
            )

    for index, row in enumerate(rows):
        row_path = f"{path}:row[{index}]"
        status = str(row["status"])
        if status not in _ALLOWED_STATUSES:
            issues.append(
                FoldCheckIssue(
                    check_id="thread.foldcheck_report.invalid_status",
                    message=f"Fold-check row status must be one of {sorted(_ALLOWED_STATUSES)}",
                    path=row_path,
                )
            )
        if not _is_hash_uri(row["input_sequence_hash"]):
            issues.append(
                FoldCheckIssue(
                    check_id="thread.foldcheck_report.invalid_sequence_hash",
                    message="Fold-check rows must carry a sha256 input_sequence_hash",
                    path=row_path,
                )
            )
        if status == "accepted":
            _validate_accepted_row(row, row_path, issues)
        elif not str(row.get("rejection_reason") or row.get("missing_metric_reason") or "").strip():
            issues.append(
                FoldCheckIssue(
                    check_id="thread.foldcheck_report.missing_failure_reason",
                    message="Rejected or errored fold-check rows must carry a failure reason",
                    path=row_path,
                )
            )
    return issues


def _validate_accepted_row(row: Mapping[str, Any], row_path: str, issues: list[FoldCheckIssue]) -> None:
    required_text_fields = (
        "runtime_kind",
        "runtime_version",
        "reference_structure_id",
        "wt_baseline_artifact_id",
        "threshold_id",
    )
    for field in required_text_fields:
        if not isinstance(row.get(field), str) or not str(row[field]).strip():
            issues.append(
                FoldCheckIssue(
                    check_id="thread.foldcheck_report.accepted_missing_required_text",
                    message=f"Accepted fold-check rows must declare {field}",
                    path=f"{row_path}:{field}",
                )
            )
    if not _is_hash_uri(row.get("runtime_parameters_hash")):
        issues.append(
            FoldCheckIssue(
                check_id="thread.foldcheck_report.accepted_missing_runtime_hash",
                message="Accepted fold-check rows must carry a runtime_parameters_hash",
                path=row_path,
            )
        )
    if not isinstance(row.get("threshold_values"), Mapping) or not row["threshold_values"]:
        issues.append(
            FoldCheckIssue(
                check_id="thread.foldcheck_report.accepted_missing_threshold_values",
                message="Accepted fold-check rows must carry threshold_values",
                path=row_path,
            )
        )
    if row.get("plddt") is None or row.get("backbone_rmsd_to_reference") is None:
        issues.append(
            FoldCheckIssue(
                check_id="thread.foldcheck_report.accepted_missing_metrics",
                message="Accepted fold-check rows must carry plddt and backbone RMSD metrics",
                path=row_path,
            )
        )


def _is_hash_uri(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.strip().removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest.lower())
