"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/io.py

Arrow I/O + snapshot retention + small filesystem helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

SNAPSHOT_DIR_NAME: str = "_snapshots"
SNAPSHOT_KEEP_N: int = 5
PARQUET_COMPRESSION: str = "zstd"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
        for p in snaps:
            try: p.unlink()
            except Exception: pass
        return
    if len(snaps) <= keep_n:
        return
    for p in snaps[:-keep_n]:
        try: p.unlink()
        except Exception: pass


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
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            snap_path = snapshot_dir / f"records-{ts}.parquet"
            _write_parquet(table, snap_path, compression=codec)
            _prune_snapshots(snapshot_dir, SNAPSHOT_KEEP_N)
        except Exception:
            pass

    tmp = target.with_suffix(".tmp.parquet")
    _write_parquet(table, tmp, compression=codec)
    os.replace(tmp, target)


def read_parquet(path: Path, columns=None) -> pa.Table:
    return pq.read_table(path, columns=columns)


def append_event(event_path: Path, payload: dict) -> None:
    event_path = Path(event_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("ts", now_utc())
    try:
        with event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass
