"""Source-surface contract checks for the DenseGen axis probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .constants import CANDIDATE_RECORDS, X_COLUMN
from .paths import _resolve_repo_path

EXPECTED_X_DIM = 8192


def validate_candidate_x_surface(repo_root: Path, *, expected_rows: int) -> dict[str, Any]:
    """Validate the OPAL candidate table X schema without scanning vector payloads."""

    records_path = _resolve_repo_path(repo_root, CANDIDATE_RECORDS)
    if not records_path.exists():
        raise ValueError(f"candidate records missing: {records_path}")
    try:
        parquet = pq.ParquetFile(records_path)
    except Exception as exc:
        raise ValueError(f"failed to read candidate records schema: {records_path}: {exc}") from exc

    schema = parquet.schema_arrow
    missing = [column for column in ("id", X_COLUMN) if column not in schema.names]
    if missing:
        raise ValueError(f"candidate records missing OPAL X contract column(s): {missing}")

    field = schema.field(X_COLUMN)
    if not pa.types.is_fixed_size_list(field.type):
        raise ValueError(f"OPAL X column {X_COLUMN!r} must be fixed_size_list, found {field.type}")
    if not pa.types.is_float32(field.type.value_type):
        raise ValueError(f"OPAL X column {X_COLUMN!r} values must be float32, found {field.type.value_type}")
    x_dim = int(field.type.list_size)
    if x_dim != EXPECTED_X_DIM:
        raise ValueError(f"OPAL X column {X_COLUMN!r} has dimension {x_dim}; expected {EXPECTED_X_DIM}")

    row_count = int(parquet.metadata.num_rows)
    if row_count != int(expected_rows):
        raise ValueError(
            f"candidate records row count {row_count} does not match source oracle row count {int(expected_rows)}"
        )

    return {
        "records_path": str(records_path),
        "x_column": X_COLUMN,
        "row_count": row_count,
        "x_dim": x_dim,
        "x_value_type": str(field.type.value_type),
        "validation_level": "parquet_schema_and_row_count",
    }
