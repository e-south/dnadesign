"""
DenseGen output ID index (SQLite-backed) for fast dedup + alignment checks.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable, List

from .base import AlignmentDigest

INDEX_FILENAME = "_densegen_ids.sqlite"
INDEX_VERSION = "1"
_HEX_WIDTH = 32  # 16 bytes -> 32 hex chars
_MAX_SQL_VARS = 900  # conservative chunk size for SQLite IN queries


def _hash_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "big")


def _xor_hash(current: int, ids: Iterable[str]) -> int:
    out = int(current)
    for value in ids:
        out ^= _hash_id(value)
    return out


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class IdIndex:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS ids (id TEXT PRIMARY KEY)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._xor_hash = int(self._get_meta("xor_hash", "0" * _HEX_WIDTH), 16)
        self._count = int(self._get_meta("id_count", "0"))
        self._get_meta("version", INDEX_VERSION)

    def _get_meta(self, key: str, default: str) -> str:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is not None:
            return str(row[0])
        self._conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, default))
        self._conn.commit()
        return default

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def _flush_meta(self) -> None:
        self._set_meta("xor_hash", f"{self._xor_hash:0{_HEX_WIDTH}x}")
        self._set_meta("id_count", str(self._count))
        self._conn.commit()

    def alignment_digest(self) -> AlignmentDigest:
        return AlignmentDigest(self._count, f"{self._xor_hash:0{_HEX_WIDTH}x}")

    def contains(self, seq_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM ids WHERE id=? LIMIT 1", (seq_id,)).fetchone()
        return row is not None

    def contains_many(self, ids: Iterable[str]) -> set[str]:
        ids_list = [str(i) for i in ids if str(i)]
        if not ids_list:
            return set()
        if self._count <= 0:
            return set()
        return self._existing_in_batch(ids_list)

    def existing_ids(self) -> set[str]:
        rows = self._conn.execute("SELECT id FROM ids").fetchall()
        return {str(r[0]) for r in rows}

    def _existing_in_batch(self, ids: List[str]) -> set[str]:
        existing: set[str] = set()
        for chunk in _chunked(ids, _MAX_SQL_VARS):
            q = "SELECT id FROM ids WHERE id IN (%s)" % ",".join("?" for _ in chunk)
            rows = self._conn.execute(q, chunk).fetchall()
            existing.update(str(r[0]) for r in rows)
        return existing

    def add(self, ids: Iterable[str]) -> int:
        ids_list = [str(i) for i in ids if str(i)]
        if not ids_list:
            return 0
        existing = self._existing_in_batch(ids_list) if self._count else set()
        new_ids = [i for i in ids_list if i not in existing]
        if not new_ids:
            return 0
        self._conn.executemany("INSERT OR IGNORE INTO ids (id) VALUES (?)", [(i,) for i in new_ids])
        self._xor_hash = _xor_hash(self._xor_hash, new_ids)
        self._count += len(new_ids)
        self._flush_meta()
        return len(new_ids)

    def bootstrap_from_parquet(self, dataset_path: Path) -> None:
        if self._count > 0:
            return
        try:
            import pyarrow.dataset as ds
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(f"Parquet support is not available: {e}") from e
        dataset = ds.dataset(dataset_path, format="parquet")
        scanner = ds.Scanner.from_dataset(dataset, columns=["id"], batch_size=4096)
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            ids = [str(x) for x in batch.column(0).to_pylist() if x is not None]
            if ids:
                self.add(ids)


def compute_alignment_digest_from_ids(ids: Iterable[str]) -> AlignmentDigest:
    xor_val = 0
    count = 0
    for value in ids:
        if value is None:
            continue
        xor_val ^= _hash_id(str(value))
        count += 1
    return AlignmentDigest(count, f"{xor_val:0{_HEX_WIDTH}x}")
