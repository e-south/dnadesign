"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/io/parquet_source.py

Thin Parquet row reader that validates required columns and yields python dict rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..core import SchemaError


def _ensure_pyarrow():
    try:
        import pyarrow.parquet as pq

        return pq
    except Exception as exc:
        raise SchemaError("Reading Parquet requires pyarrow to be installed and importable.") from exc


def iter_parquet_rows(path: str | Path, columns: Sequence[str], batch_size: int = 4096) -> Iterable[dict]:
    pq = _ensure_pyarrow()
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"Parquet input does not exist: {p}")

    pf = pq.ParquetFile(p)
    schema_names = set(pf.schema_arrow.names)
    required = [str(c) for c in columns]
    missing = [c for c in required if c not in schema_names]
    if missing:
        raise SchemaError(f"Missing required Parquet columns: {missing}. Present: {sorted(schema_names)}")

    for batch in pf.iter_batches(columns=required, batch_size=int(batch_size)):
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                raise SchemaError("Parquet row conversion failed: expected dict rows")
            yield row
