"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/storage/parquet.py

Parquet IO primitives used by USR dataset and maintenance operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts import SequencesError

SNAPSHOT_DIR_NAME: str = "_snapshots"
SNAPSHOT_KEEP_N: int = 5
PARQUET_COMPRESSION: str = "zstd"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _snapshot_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def _write_parquet(
    table: pa.Table,
    out_path: Path,
    *,
    compression: Optional[str] = None,
    use_dictionary: bool = True,
    write_statistics: bool = True,
) -> None:
    pq.write_table(
        table,
        out_path,
        compression=compression,
        use_dictionary=use_dictionary,
        write_statistics=write_statistics,
    )


def _prune_snapshots(snapshot_dir: Path, keep_n: int) -> None:
    snaps = sorted(snapshot_dir.glob("records-*.parquet"))
    if keep_n <= 0:
        failures = []
        for path in snaps:
            try:
                path.unlink()
            except OSError as exc:
                failures.append((path, exc))
        if failures:
            first_path, first_err = failures[0]
            raise SequencesError(
                f"Failed to prune {len(failures)} snapshot(s); first error for {first_path}: {first_err}"
            ) from first_err
        return
    if len(snaps) <= keep_n:
        return
    failures = []
    for path in snaps[:-keep_n]:
        try:
            path.unlink()
        except OSError as exc:
            failures.append((path, exc))
    if failures:
        first_path, first_err = failures[0]
        raise SequencesError(
            f"Failed to prune {len(failures)} snapshot(s); first error for {first_path}: {first_err}"
        ) from first_err


def write_parquet_atomic(
    table: pa.Table,
    target: Path,
    snapshot_dir: Path,
    *,
    compression: Optional[str] = None,
    preserve_metadata_from: Optional[pa.Table] = None,
) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if preserve_metadata_from is not None:
        base_md = preserve_metadata_from.schema.metadata
        if base_md and not table.schema.metadata:
            table = table.replace_schema_metadata(base_md)

    codec = PARQUET_COMPRESSION if compression is None else compression

    if SNAPSHOT_KEEP_N > 0:
        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snapshot_dir / f"records-{_snapshot_stamp()}.parquet"
            _write_parquet(table, snap_path, compression=codec)
            _prune_snapshots(snapshot_dir, SNAPSHOT_KEEP_N)
        except OSError as exc:
            raise SequencesError(f"Snapshot write/prune failed for {target}: {exc}") from exc

    tmp = target.with_suffix(".tmp.parquet")
    _write_parquet(table, tmp, compression=codec)
    os.replace(tmp, target)


def commit_parquet_atomic_file(
    tmp_path: Path,
    target: Path,
    snapshot_dir: Optional[Path],
) -> None:
    """
    Atomically replace target with an existing parquet file and snapshot it.
    """
    tmp_path = Path(tmp_path)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if snapshot_dir and SNAPSHOT_KEEP_N > 0:
        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snapshot_dir / f"records-{_snapshot_stamp()}.parquet"
            shutil.copy2(tmp_path, snap_path)
            _prune_snapshots(snapshot_dir, SNAPSHOT_KEEP_N)
        except OSError as exc:
            raise SequencesError(f"Snapshot write/prune failed for {target}: {exc}") from exc

    os.replace(tmp_path, target)


def snapshot_parquet_file(source: Path, snapshot_dir: Path) -> None:
    source = Path(source)
    if SNAPSHOT_KEEP_N <= 0:
        return
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snapshot_dir / f"records-{_snapshot_stamp()}.parquet"
        shutil.copy2(source, snap_path)
        _prune_snapshots(snapshot_dir, SNAPSHOT_KEEP_N)
    except OSError as exc:
        raise SequencesError(f"Snapshot write/prune failed for {source}: {exc}") from exc


def write_parquet_atomic_batches(
    batches,
    schema: pa.Schema,
    target: Path,
    snapshot_dir: Optional[Path],
    *,
    compression: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    codec = PARQUET_COMPRESSION if compression is None else compression
    if metadata:
        schema = schema.with_metadata(metadata)

    tmp = target.with_suffix(".tmp.parquet")
    wrote = False
    writer = pq.ParquetWriter(
        tmp,
        schema,
        compression=codec,
        use_dictionary=True,
        write_statistics=True,
    )
    try:
        for batch in batches:
            writer.write_batch(batch)
            wrote = True
        if not wrote:
            empty_arrays = [pa.array([], type=field.type) for field in schema]
            empty_batch = pa.RecordBatch.from_arrays(empty_arrays, schema=schema)
            writer.write_batch(empty_batch)
    finally:
        writer.close()

    if snapshot_dir and SNAPSHOT_KEEP_N > 0:
        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snapshot_dir / f"records-{_snapshot_stamp()}.parquet"
            shutil.copy2(tmp, snap_path)
            _prune_snapshots(snapshot_dir, SNAPSHOT_KEEP_N)
        except OSError as exc:
            raise SequencesError(f"Snapshot write/prune failed for {target}: {exc}") from exc

    os.replace(tmp, target)


def read_parquet(path: Path, columns=None) -> pa.Table:
    return pq.read_table(path, columns=columns)


def iter_parquet_batches(path: Path, columns=None, batch_size: int = 65536):
    parquet_file = pq.ParquetFile(path)
    return parquet_file.iter_batches(batch_size=int(batch_size), columns=columns)


def read_parquet_head(path: Path, n: int, columns=None) -> pa.Table:
    parquet_file = pq.ParquetFile(path)
    if n <= 0:
        return pa.Table.from_batches([], schema=parquet_file.schema_arrow)
    batches = parquet_file.iter_batches(batch_size=int(n), columns=columns)
    try:
        batch = next(batches)
    except StopIteration:
        return pa.Table.from_batches([], schema=parquet_file.schema_arrow)
    table = pa.Table.from_batches([batch])
    if table.num_rows > n:
        table = table.slice(0, n)
    return table


__all__ = [
    "PARQUET_COMPRESSION",
    "SNAPSHOT_DIR_NAME",
    "SNAPSHOT_KEEP_N",
    "commit_parquet_atomic_file",
    "iter_parquet_batches",
    "now_utc",
    "read_parquet",
    "read_parquet_head",
    "snapshot_parquet_file",
    "write_parquet_atomic",
    "write_parquet_atomic_batches",
]
