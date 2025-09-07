"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/diff.py

Local parquet/file stats + compact diff summary formatting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pyarrow.parquet as pq

from .remote import RemoteDatasetStat


@dataclass
class FileStat:
    exists: bool
    size: Optional[int]
    sha256: Optional[str]
    rows: Optional[int]
    cols: Optional[int]
    mtime: Optional[str]  # epoch seconds as string (for uniformity)


@dataclass
class SnapshotStat:
    count: int
    newest_ts: Optional[str]  # "YYYYMMDDThhmmss"
    newer_than_local: int     # computed relative to other side


@dataclass
class DiffSummary:
    dataset: str
    primary_local: FileStat
    primary_remote: FileStat
    meta_local_mtime: Optional[str]
    meta_remote_mtime: Optional[str]
    events_local_lines: int
    events_remote_lines: int
    snapshots: SnapshotStat
    has_change: bool
    changes: dict


_PAT_TS = re.compile(r"records-(\d{8}T\d{6})\.parquet$")


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parquet_stats(path: Path) -> FileStat:
    p = Path(path)
    if not p.exists():
        return FileStat(False, None, None, None, None, None)
    mtime = str(int(p.stat().st_mtime))
    size = int(p.stat().st_size)
    sha = _sha256_file(p)
    pf = pq.ParquetFile(str(p))
    meta = pf.metadata
    return FileStat(True, size, sha, meta.num_rows, meta.num_columns, mtime)


def file_mtime(path: Path) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    return str(int(p.stat().st_mtime))


def events_tail_count(path: Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    # Efficient enough for logs that are typically small
    with p.open("rb") as f:
        return sum(1 for _ in f)


def _snap_ts_from_name(name: str) -> Optional[str]:
    m = _PAT_TS.search(name)
    return m.group(1) if m else None


def snapshots_stat(dir_path: Path, remote_names: List[str]) -> SnapshotStat:
    # Local inventory
    local_dir = Path(dir_path)
    local_names = []
    if local_dir.exists():
        for n in local_dir.iterdir():
            if n.is_file() and _PAT_TS.match(n.name):
                local_names.append(n.name)

    def newest(names: List[str]) -> Optional[str]:
        ts = [_snap_ts_from_name(n) for n in names]
        ts = [t for t in ts if t]
        if not ts:
            return None
        return sorted(ts)[-1]

    newest_local = newest(local_names)
    newest_remote = newest(remote_names)

    newer_than_local = 0
    if newest_local:
        for n in remote_names:
            t = _snap_ts_from_name(n)
            if t and t > newest_local:
                newer_than_local += 1
    else:
        newer_than_local = len(remote_names)

    return SnapshotStat(
        count=len(remote_names),
        newest_ts=newest_remote,
        newer_than_local=newer_than_local,
    )


def compute_diff(dataset_dir: Path, remote: RemoteDatasetStat, dataset_name: str) -> DiffSummary:
    dataset_dir = Path(dataset_dir)
    primary_local = parquet_stats(dataset_dir / "records.parquet")
    meta_local = file_mtime(dataset_dir / "meta.yaml")
    events_local = events_tail_count(dataset_dir / ".events.log")

    # Convert remote stat to FileStat for unified view
    primary_remote = FileStat(
        exists=remote.primary.exists,
        size=remote.primary.size,
        sha256=remote.primary.sha256,
        rows=remote.primary.rows,
        cols=remote.primary.cols,
        mtime=remote.primary.mtime,
    )

    snaps = snapshots_stat(dataset_dir / "_snapshots", remote.snapshot_names)

    changes = {
        "primary_sha_diff": (
            (primary_local.sha256 != primary_remote.sha256)
            if (primary_local.exists and primary_remote.exists and primary_remote.sha256)
            else (primary_local.size != primary_remote.size)
            if (primary_local.exists and primary_remote.exists)
            else (primary_local.exists != primary_remote.exists)
        ),
        "meta_mtime_diff": (meta_local != remote.meta_mtime),
        "events_new_remote_lines": max(0, remote.events_lines - events_local),
        "snapshots_remote_newer": snaps.newer_than_local,
    }

    has_change = bool(
        changes["primary_sha_diff"]
        or changes["meta_mtime_diff"]
        or changes["events_new_remote_lines"] > 0
        or changes["snapshots_remote_newer"] > 0
    )

    return DiffSummary(
        dataset=dataset_name,
        primary_local=primary_local,
        primary_remote=primary_remote,
        meta_local_mtime=meta_local,
        meta_remote_mtime=remote.meta_mtime,
        events_local_lines=events_local,
        events_remote_lines=remote.events_lines,
        snapshots=snaps,
        has_change=has_change,
        changes=changes,
    )
