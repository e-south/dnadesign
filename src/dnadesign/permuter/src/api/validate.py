"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/api/validate.py

Dataset validation contracts for materialized Permuter outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dnadesign.permuter.src.api.contracts import DatasetRef, ValidationReport
from dnadesign.permuter.src.contracts.metrics import observed_metric_ids, reject_legacy_metric_columns
from dnadesign.permuter.src.core.paths import normalize_data_path
from dnadesign.permuter.src.core.storage import read_parquet

_CORE = ("id", "bio_type", "sequence", "alphabet", "length", "source", "created_at")


def validate_dataset(data: DatasetRef | str | Path, *, strict: bool = False) -> ValidationReport:
    """Validate a materialized Permuter dataset without invoking the CLI."""

    records = data.records_path if isinstance(data, DatasetRef) else normalize_data_path(Path(data))
    df = read_parquet(records)
    _validate_frame(df, strict=strict)
    return ValidationReport(
        ok=True,
        records_path=records,
        row_count=len(df),
        strict=strict,
        metric_ids=tuple(observed_metric_ids(df.columns)),
    )


def _validate_frame(df, *, strict: bool) -> None:
    missing = [column for column in _CORE if column not in df.columns]
    if missing:
        raise ValueError(f"USR core columns missing: {missing}")
    recomputed = df.apply(lambda row: _sha1(str(row["bio_type"]), str(row["sequence"])), axis=1)
    bad = (recomputed != df["id"]).sum()
    if bad:
        raise ValueError(f"{bad} row(s) have incorrect id for (bio_type|sequence)")
    if strict:
        reject_legacy_metric_columns(df, context="validate")
        if "permuter__variant_id" in df.columns:
            raise ValueError("permuter__variant_id is not supported; use permuter__var_id")
    for column in df.columns:
        if column in _CORE:
            continue
        if "__" not in str(column) and strict:
            raise ValueError(f"Non-namespaced derived column in strict mode: {column}")
    required = (
        "permuter__scope",
        "permuter__ref",
        "permuter__protocol",
        "permuter__var_id",
    )
    miss = [column for column in required if column not in df.columns]
    if miss and strict:
        raise ValueError(f"Missing required permuter columns: {miss}")


def _sha1(bio_type: str, sequence: str) -> str:
    return hashlib.sha1(f"{bio_type}|{sequence}".encode("utf-8")).hexdigest()
