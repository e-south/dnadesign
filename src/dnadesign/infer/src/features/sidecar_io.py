"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/sidecar_io.py

Filesystem primitives for Infer feature sidecar commits.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.contracts.feature_payloads import payload_lock

SIDECAR_LOCK_RELATIVE_PATH = "_derived/infer/.sidecar.lock"


@contextmanager
def sidecar_dataset_lock(*, dataset_root: str | Path, dataset_id: str) -> Iterator[None]:
    lock_path = Path(dataset_root) / dataset_id / SIDECAR_LOCK_RELATIVE_PATH
    with payload_lock(lock_path, exclusive=True):
        yield


def atomic_parquet_temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_table_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = atomic_parquet_temp_path(path)
    try:
        temp_path.unlink(missing_ok=True)
        pq.write_table(table, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_table_with_schema(path: Path, *, schema: pa.Schema) -> pa.Table:
    table = pq.read_table(path)
    columns = {}
    for field in schema:
        if field.name in table.column_names:
            columns[field.name] = table.column(field.name).cast(field.type)
        else:
            columns[field.name] = pa.nulls(table.num_rows, type=field.type)
    return pa.table(columns, schema=schema)
