"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/x_contracts.py

Validation helpers for OPAL feature-vector columns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.utils import OpalError


@dataclass(frozen=True)
class XSeriesValidation:
    row_count: int
    x_dim: int


def validate_x_series(
    series: pd.Series,
    *,
    x_column: str,
    id_series: pd.Series | None = None,
) -> XSeriesValidation:
    """Validate that an OPAL X column is a complete fixed-length finite vector series."""

    expected_dim: int | None = None
    row_count = 0
    ids = id_series.tolist() if id_series is not None else [None] * len(series)

    for row_index, (value, row_id) in enumerate(zip(series.tolist(), ids, strict=True)):
        sample_id = _sample_id(row_id, row_index=row_index)
        if _is_missing_cell(value):
            raise OpalError(f"X column '{x_column}' is null for id {sample_id}.")

        vector = _coerce_x_vector(value, x_column=x_column, sample_id=sample_id)
        if not vector:
            raise OpalError(f"X column '{x_column}' has an empty vector for id {sample_id}.")

        current_dim = len(vector)
        if expected_dim is None:
            expected_dim = current_dim
        elif current_dim != expected_dim:
            raise OpalError(
                f"X column '{x_column}' must be fixed-length; first non-null vector has length "
                f"{expected_dim}, but id {sample_id} has length {current_dim}."
            )
        row_count += 1

    if row_count == 0 or expected_dim is None:
        raise OpalError(f"X column '{x_column}' has no rows to validate.")
    return XSeriesValidation(row_count=row_count, x_dim=expected_dim)


def validate_x_parquet_column(
    records_path: str | Path,
    *,
    x_column: str,
    id_column: str = "id",
    batch_size: int = 64,
) -> XSeriesValidation:
    """Validate an X column directly from Parquet without materializing all records."""

    parquet_path = Path(records_path)
    if not parquet_path.exists():
        raise OpalError(f"records.parquet not found: {parquet_path}")
    if int(batch_size) <= 0:
        raise OpalError("X validation batch_size must be positive.")
    try:
        parquet = pq.ParquetFile(parquet_path)
    except Exception as exc:
        raise OpalError(f"Failed to read records.parquet schema: {parquet_path}: {exc}") from exc

    schema = parquet.schema_arrow
    missing = [column for column in (id_column, x_column) if column not in schema.names]
    if missing:
        raise OpalError(f"records.parquet missing required column(s): {missing}")

    field = schema.field(x_column)
    if not pa.types.is_fixed_size_list(field.type):
        raise OpalError(f"X column '{x_column}' must be stored as a Parquet fixed_size_list; found {field.type}.")
    fixed_dim = int(field.type.list_size)
    expected_dim: int | None = fixed_dim
    if expected_dim == 0:
        raise OpalError(f"X column '{x_column}' has an empty fixed-size-list vector type.")

    row_count = 0
    try:
        batches = parquet.iter_batches(columns=[id_column, x_column], batch_size=int(batch_size))
        for batch in batches:
            ids = batch.column(id_column).to_pylist()
            values = batch.column(x_column)
            _validate_fixed_size_x_batch(
                values,
                ids,
                x_column=x_column,
                expected_dim=fixed_dim,
                row_offset=row_count,
            )
            row_count += len(values)
    except OpalError:
        raise
    except Exception as exc:
        if "Expected all lists to be of size" in str(exc):
            raise OpalError(
                f"X column '{x_column}' has null or ragged fixed-size-list rows; "
                f"expected every row to contain exactly {fixed_dim} values."
            ) from exc
        raise OpalError(f"Failed to validate X column '{x_column}' in {parquet_path}: {exc}") from exc

    if row_count == 0 or expected_dim is None:
        raise OpalError(f"X column '{x_column}' has no rows to validate.")
    return XSeriesValidation(row_count=int(row_count), x_dim=int(expected_dim))


def _validate_fixed_size_x_batch(
    values: pa.Array,
    ids: Sequence[object],
    *,
    x_column: str,
    expected_dim: int,
    row_offset: int,
) -> None:
    if values.null_count:
        local_index = _first_true(values.is_null().to_pylist())
        sample_id = _sample_id(ids[local_index], row_index=row_offset + local_index)
        raise OpalError(f"X column '{x_column}' is null for id {sample_id}.")
    child = values.values
    if child.null_count:
        local_index = _first_true(child.is_null().to_pylist()) // int(expected_dim)
        sample_id = _sample_id(ids[local_index], row_index=row_offset + local_index)
        raise OpalError(f"X column '{x_column}' contains null vector values for id {sample_id}.")
    if pa.types.is_floating(child.type):
        array = child.to_numpy(zero_copy_only=False)
        bad = np.flatnonzero(~np.isfinite(array))
        if len(bad):
            local_index = int(bad[0]) // int(expected_dim)
            sample_id = _sample_id(ids[local_index], row_index=row_offset + local_index)
            raise OpalError(f"X column '{x_column}' contains non-finite values for id {sample_id}.")
    elif not pa.types.is_integer(child.type):
        for local_index, value in enumerate(values.to_pylist()):
            sample_id = _sample_id(ids[local_index], row_index=row_offset + local_index)
            vector = _coerce_x_vector(value, x_column=x_column, sample_id=sample_id)
            if len(vector) != int(expected_dim):
                raise OpalError(
                    f"X column '{x_column}' must be fixed-length; first non-null vector has length "
                    f"{expected_dim}, but id {sample_id} has length {len(vector)}."
                )


def _first_true(values: Sequence[bool]) -> int:
    for index, value in enumerate(values):
        if value:
            return int(index)
    raise OpalError("internal X validation error: expected at least one invalid value.")


def _sample_id(value: object, *, row_index: int) -> str:
    if value is None or _is_missing_cell(value):
        return f"row {row_index}"
    return repr(str(value))


def _is_missing_cell(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        marker = pd.isna(value)
    except Exception:
        return False
    if isinstance(marker, (bool, np.bool_)):
        return bool(marker)
    return False


def _coerce_x_vector(value: object, *, x_column: str, sample_id: str) -> list[float]:
    value = _unwrap_arrow_like(value)
    if isinstance(value, str):
        return _coerce_x_string(value, x_column=x_column, sample_id=sample_id)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return _coerce_sequence(value, x_column=x_column, sample_id=sample_id)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _coerce_sequence(list(value), x_column=x_column, sample_id=sample_id)
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise OpalError(f"X column '{x_column}' contains non-finite values for id {sample_id}.")
        return [numeric]
    raise OpalError(
        f"X column '{x_column}' must contain numeric vectors or JSON arrays; "
        f"id {sample_id} has value type {type(value).__name__}."
    )


def _unwrap_arrow_like(value: object) -> object:
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        return as_py()
    to_pylist = getattr(value, "to_pylist", None)
    if callable(to_pylist):
        return to_pylist()
    return value


def _coerce_x_string(value: str, *, x_column: str, sample_id: str) -> list[float]:
    stripped = value.strip()
    if not stripped:
        raise OpalError(f"X column '{x_column}' has an empty string for id {sample_id}.")
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise OpalError(f"X column '{x_column}' has malformed JSON array for id {sample_id}: {exc}") from exc
        if not isinstance(parsed, list):
            raise OpalError(f"X column '{x_column}' JSON value must be an array for id {sample_id}.")
        return _coerce_sequence(parsed, x_column=x_column, sample_id=sample_id)
    try:
        numeric = float(stripped)
    except ValueError as exc:
        raise OpalError(
            f"X column '{x_column}' string values must be JSON arrays or numeric scalars; id {sample_id} is invalid."
        ) from exc
    if not math.isfinite(numeric):
        raise OpalError(f"X column '{x_column}' contains non-finite values for id {sample_id}.")
    return [numeric]


def _coerce_sequence(value: object, *, x_column: str, sample_id: str) -> list[float]:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except Exception as exc:
        raise OpalError(f"X column '{x_column}' contains non-numeric vector values for id {sample_id}.") from exc
    if arr.size == 0:
        return []
    if not np.all(np.isfinite(arr)):
        raise OpalError(f"X column '{x_column}' contains non-finite values for id {sample_id}.")
    return [float(v) for v in arr.tolist()]
