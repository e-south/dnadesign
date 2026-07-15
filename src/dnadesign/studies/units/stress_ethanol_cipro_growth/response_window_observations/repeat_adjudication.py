"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/repeat_adjudication.py

Typed evidence requirements for repeated-experiment decisions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from .contracts import REPEAT_CLASSIFICATIONS, REPEAT_STATUSES, ResponseWindowAggregationError

_CLASSIFICATIONS_BY_STATUS = {
    "comparable": {"assay_context_comparable", "corrected_technical_error"},
    "excluded_noncomparable": {"noncomparable_assay_context", "plausible_biological_variation"},
    "remeasure_required": {"remeasurement_required"},
}


def validate_repeat_adjudications(frame: pd.DataFrame, *, evidence_root: Path | None = None) -> None:
    """Reject unsupported status text and optionally verify referenced evidence."""

    invalid_statuses = sorted(set(frame["status"].astype(str)) - REPEAT_STATUSES)
    if invalid_statuses:
        raise ResponseWindowAggregationError(f"repeat decisions contain unsupported statuses: {invalid_statuses}")
    invalid = sorted(set(frame["classification"].astype(str)) - REPEAT_CLASSIFICATIONS)
    if invalid:
        raise ResponseWindowAggregationError(f"repeat decisions contain unsupported classifications: {invalid}")
    for row in frame.itertuples(index=False):
        _validate_row(row, evidence_root=evidence_root)


def _validate_row(row: object, *, evidence_root: Path | None) -> None:
    status = str(row.status)
    classification = str(row.classification)
    evidence_fields = (row.evidence_artifact, row.evidence_sha256, row.adjudicated_by, row.adjudicated_at)
    if status == "review_required":
        if classification != "unresolved" or any(not _missing(value) for value in evidence_fields):
            raise ResponseWindowAggregationError(
                f"{row.candidate_id}: review_required must remain unresolved without adjudication evidence."
            )
        return
    if classification not in _CLASSIFICATIONS_BY_STATUS[status]:
        raise ResponseWindowAggregationError(
            f"{row.candidate_id}: repeat status {status!r} disagrees with classification {classification!r}."
        )
    if any(_missing(value) for value in evidence_fields):
        raise ResponseWindowAggregationError(f"{row.candidate_id}: final repeat decision requires typed evidence.")
    digest = str(row.evidence_sha256)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ResponseWindowAggregationError(f"{row.candidate_id}: repeat evidence digest is invalid.")
    try:
        timestamp = datetime.fromisoformat(str(row.adjudicated_at))
    except ValueError as exc:
        raise ResponseWindowAggregationError(f"{row.candidate_id}: repeat adjudication timestamp is invalid.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ResponseWindowAggregationError(
            f"{row.candidate_id}: repeat adjudication timestamp must be timezone-aware."
        )
    if evidence_root is not None:
        path = _evidence_path(evidence_root, str(row.evidence_artifact), candidate_id=str(row.candidate_id))
        if _sha256(path) != digest:
            raise ResponseWindowAggregationError(f"{row.candidate_id}: repeat evidence artifact digest mismatch.")


def _evidence_path(root: Path, value: str, *, candidate_id: str) -> Path:
    relative = PurePosixPath(value)
    if not value or "\\" in value or relative.is_absolute() or ".." in relative.parts:
        raise ResponseWindowAggregationError(f"{candidate_id}: repeat evidence artifact path is not confined.")
    resolved_root = Path(root).resolve()
    path = (resolved_root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise ResponseWindowAggregationError(f"{candidate_id}: repeat evidence artifact is missing or unconfined.")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _missing(value: object) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) or not str(value).strip()


__all__ = ["validate_repeat_adjudications"]
