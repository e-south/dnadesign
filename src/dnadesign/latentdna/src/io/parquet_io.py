"""
Parquet IO helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_table(path: Path, *, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(path, columns=columns)


def read_schema(path: Path):
    return pq.read_schema(path)


def read_row_count(path: Path) -> int:
    metadata = pq.ParquetFile(path).metadata
    return 0 if metadata is None else metadata.num_rows


def write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
