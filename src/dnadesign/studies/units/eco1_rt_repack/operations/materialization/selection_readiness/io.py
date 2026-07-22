"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/io.py

Input/output helpers for Eco1 panel-selection materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_rows(path: Path, *, required: bool = True) -> list[dict[str, object]]:
    """Read parquet rows or return an empty list for optional missing inputs."""

    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    return pq.read_table(path).to_pylist()


def write_rows(path: Path, rows: list[dict[str, object]], *, schema_id: str) -> None:
    """Write rows to parquet with lightweight schema metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        b"schema_id": schema_id.encode(),
        b"schema_version": b"1",
        b"status": b"materialized",
    }
    table = pa.Table.from_pylist(rows)
    pq.write_table(table.replace_schema_metadata(metadata), path)
