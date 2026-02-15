"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_dedupe.py

Row deduplication operation extracted from Dataset methods.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq

from .errors import SchemaError, SequencesError
from .maintenance import require_maintenance
from .storage.locking import dataset_write_lock
from .storage.parquet import PARQUET_COMPRESSION, iter_parquet_batches, now_utc, write_parquet_atomic_batches


def dedupe_dataset(
    *,
    dataset: Any,
    key: str,
    keep: str,
    batch_size: int = 65536,
    dry_run: bool = False,
):
    """
    Deduplicate rows by key, keeping either the first or last occurrence.
    """
    dataset._require_exists()
    ctx = require_maintenance("dedupe")
    if batch_size < 1:
        raise SequencesError("batch_size must be >= 1.")
    key = str(key or "").strip()
    keep = str(keep or "").strip()
    if key not in {"id", "sequence", "sequence_norm", "sequence_ci"}:
        raise SchemaError(f"Unsupported dedupe key '{key}'.")
    if keep not in {"keep-first", "keep-last"}:
        raise SchemaError(f"Unsupported dedupe keep policy '{keep}'.")

    key_cols = ["id"] if key == "id" else ["sequence"]
    if key == "sequence_ci":
        key_cols = ["sequence", "alphabet"]

    def _chunked(items: List[str], size: int = 900) -> Iterable[List[str]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    def _fetch_existing(conn: sqlite3.Connection, keys: List[str]) -> set[str]:
        existing: set[str] = set()
        if not keys:
            return existing
        for chunk in _chunked(keys):
            placeholders = ",".join("?" for _ in chunk)
            cur = conn.execute(f"SELECT k FROM seen WHERE k IN ({placeholders})", chunk)
            existing.update(row[0] for row in cur.fetchall())
        return existing

    def _fetch_last_indices(conn: sqlite3.Connection, keys: List[str]) -> Dict[str, int]:
        last: Dict[str, int] = {}
        if not keys:
            return last
        for chunk in _chunked(keys):
            placeholders = ",".join("?" for _ in chunk)
            cur = conn.execute(f"SELECT k, idx FROM last WHERE k IN ({placeholders})", chunk)
            for key_value, idx in cur.fetchall():
                last[str(key_value)] = int(idx)
        return last

    def _filter_batch(batch: pa.RecordBatch, keep_mask: List[bool]) -> Iterable[pa.RecordBatch]:
        if not keep_mask or not any(keep_mask):
            return []
        tbl = pa.Table.from_batches([batch])
        filtered = tbl.filter(pa.array(keep_mask))
        return filtered.to_batches()

    with dataset_write_lock(dataset.dir):
        dataset._auto_freeze_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "dedupe.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE counts (k TEXT PRIMARY KEY, cnt INTEGER NOT NULL)")
                if keep == "keep-last":
                    conn.execute("CREATE TABLE last (k TEXT PRIMARY KEY, idx INTEGER NOT NULL)")

                row_idx = 0
                for batch in iter_parquet_batches(dataset.records_path, columns=key_cols, batch_size=int(batch_size)):
                    keys = dataset._key_list_from_batch(batch, key)
                    conn.executemany(
                        "INSERT INTO counts (k, cnt) VALUES (?, 1) ON CONFLICT(k) DO UPDATE SET cnt = cnt + 1",
                        [(k,) for k in keys],
                    )
                    if keep == "keep-last":
                        conn.executemany(
                            "INSERT INTO last (k, idx) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET idx = excluded.idx",
                            [(k, row_idx + i) for i, k in enumerate(keys)],
                        )
                    row_idx += len(keys)

                groups = conn.execute("SELECT COUNT(*) FROM counts WHERE cnt > 1").fetchone()[0]
                dropped = conn.execute("SELECT COALESCE(SUM(cnt - 1), 0) FROM counts WHERE cnt > 1").fetchone()[0]
                from .dataset import DedupeStats

                stats = DedupeStats(
                    rows_total=int(row_idx),
                    rows_dropped=int(dropped),
                    groups=int(groups),
                    key=key,
                    keep=keep,
                )

                if dry_run or stats.rows_dropped == 0:
                    return stats

                pf = pq.ParquetFile(str(dataset.records_path))
                schema = pf.schema_arrow

                def _iter_keep_first():
                    conn.execute("CREATE TABLE seen (k TEXT PRIMARY KEY)")
                    for batch in iter_parquet_batches(dataset.records_path, columns=None, batch_size=int(batch_size)):
                        keys = dataset._key_list_from_batch(batch, key)
                        existing = _fetch_existing(conn, keys)
                        batch_seen: set[str] = set()
                        keep_mask: List[bool] = []
                        new_keys: List[tuple[str]] = []
                        for key_value in keys:
                            if key_value in existing or key_value in batch_seen:
                                keep_mask.append(False)
                                continue
                            keep_mask.append(True)
                            batch_seen.add(key_value)
                            new_keys.append((key_value,))
                        if new_keys:
                            conn.executemany("INSERT OR IGNORE INTO seen(k) VALUES (?)", new_keys)
                        for out_batch in _filter_batch(batch, keep_mask):
                            yield out_batch

                def _iter_keep_last():
                    row_at = 0
                    for batch in iter_parquet_batches(dataset.records_path, columns=None, batch_size=int(batch_size)):
                        keys = dataset._key_list_from_batch(batch, key)
                        last_idx = _fetch_last_indices(conn, keys)
                        keep_mask = [(last_idx.get(key_value) == row_at + i) for i, key_value in enumerate(keys)]
                        for out_batch in _filter_batch(batch, keep_mask):
                            yield out_batch
                        row_at += len(keys)

                batches = _iter_keep_first() if keep == "keep-first" else _iter_keep_last()
                metadata = dataset._base_metadata(created_at=now_utc())
                write_parquet_atomic_batches(
                    batches,
                    schema,
                    dataset.records_path,
                    dataset.snapshot_dir,
                    compression=PARQUET_COMPRESSION,
                    metadata=metadata,
                )
                dataset._record_event(
                    "dedupe",
                    args={
                        "key": key,
                        "keep": keep,
                        "rows_total": stats.rows_total,
                        "rows_dropped": stats.rows_dropped,
                        "groups": stats.groups,
                        "maintenance_reason": ctx.reason,
                    },
                    maintenance={"reason": ctx.reason},
                    actor=ctx.actor,
                )
                return stats
            finally:
                conn.close()
