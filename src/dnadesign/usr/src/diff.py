"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/diff.py

Local/remote file diffing and verification helpers for USR sync.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pyarrow.parquet as pq

from .errors import VerificationError
from .remote import RemoteDatasetStat

if TYPE_CHECKING:  # for static checkers only; avoids runtime coupling
    from .remote import RemotePrimaryStat  # noqa: F401


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
    newest_ts: Optional[str]  # "YYYYMMDDThhmmss[ffffff]"
    newer_than_local: int  # computed relative to other side


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
    derived_local_files: List[str]
    derived_remote_files: List[str]
    aux_local_files: List[str]
    aux_remote_files: List[str]
    has_change: bool
    changes: dict
    verify_mode: str
    verify_notes: List[str]


_PAT_TS = re.compile(r"records-(\d{8}T\d{6,})\.parquet$")


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve_verify_mode(requested: str, remote: object) -> tuple[str, List[str]]:
    req = str(requested or "").strip().lower()
    if req not in {"auto", "hash", "size", "parquet"}:
        raise VerificationError(f"Unsupported verify mode '{requested}'.")

    exists = bool(getattr(remote, "exists", True))
    if not exists:
        if req == "auto":
            return "hash", []
        return req, []

    has_sha = bool(getattr(remote, "sha256", None))
    has_size = getattr(remote, "size", None) is not None
    has_parquet = (getattr(remote, "rows", None) is not None) and (getattr(remote, "cols", None) is not None)

    if req == "hash":
        if not has_sha:
            raise VerificationError("verify=hash requires remote sha256 (sha256sum/shasum).")
        return "hash", []
    if req == "size":
        if not has_size:
            raise VerificationError("verify=size requires remote file size.")
        return "size", []
    if req == "parquet":
        if not has_parquet:
            raise VerificationError("verify=parquet requires remote parquet row/col stats.")
        return "parquet", []

    if has_sha:
        return "hash", []
    if has_size:
        return "size", ["Falling back from hash to size because remote sha256 is unavailable."]
    if has_parquet:
        return "parquet", ["Falling back from hash/size to parquet because remote size is unavailable."]
    raise VerificationError("No available verification method for remote file.")


def _primary_diff(local: FileStat, remote: FileStat, verify_mode: str) -> bool:
    if not (local.exists and remote.exists):
        return local.exists != remote.exists
    if verify_mode == "hash":
        if not local.sha256 or not remote.sha256:
            raise VerificationError("verify=hash requires sha256 on both sides.")
        return local.sha256 != remote.sha256
    if verify_mode == "size":
        if local.size is None or remote.size is None:
            raise VerificationError("verify=size requires size on both sides.")
        return local.size != remote.size
    if verify_mode == "parquet":
        if local.rows is None or local.cols is None or remote.rows is None or remote.cols is None:
            raise VerificationError("verify=parquet requires row/col stats on both sides.")
        return (local.rows != remote.rows) or (local.cols != remote.cols)
    raise VerificationError(f"Unsupported verify mode '{verify_mode}'.")


def verify_primary_match(local: FileStat, remote: FileStat, verify_mode: str, *, context: str) -> None:
    if not (local.exists and remote.exists):
        raise VerificationError(f"{context}: missing file on one side (local={local.exists}, remote={remote.exists}).")
    if verify_mode == "hash":
        if not local.sha256 or not remote.sha256:
            raise VerificationError(f"{context}: verify=hash requires sha256 on both sides.")
        if local.sha256 != remote.sha256:
            raise VerificationError(f"{context}: SHA mismatch local={local.sha256} remote={remote.sha256}")
        return
    if verify_mode == "size":
        if local.size is None or remote.size is None:
            raise VerificationError(f"{context}: verify=size requires size on both sides.")
        if local.size != remote.size:
            raise VerificationError(f"{context}: size mismatch local={local.size} remote={remote.size}")
        return
    if verify_mode == "parquet":
        if local.rows is None or local.cols is None or remote.rows is None or remote.cols is None:
            raise VerificationError(f"{context}: verify=parquet requires row/col stats on both sides.")
        if local.rows != remote.rows:
            raise VerificationError(f"{context}: row mismatch local={local.rows} remote={remote.rows}")
        if local.cols != remote.cols:
            raise VerificationError(f"{context}: col mismatch local={local.cols} remote={remote.cols}")
        return
    raise VerificationError(f"{context}: unsupported verify mode '{verify_mode}'.")


def parquet_stats(path: Path, *, include_sha: bool = True, include_parquet: bool = True) -> FileStat:
    p = Path(path)
    if not p.exists():
        return FileStat(False, None, None, None, None, None)
    try:
        stat = p.stat()
    except FileNotFoundError:
        return FileStat(False, None, None, None, None, None)
    except OSError as exc:
        raise VerificationError(f"Failed to stat local file: {p}") from exc
    mtime = str(int(stat.st_mtime))
    size = int(stat.st_size)
    if include_sha:
        try:
            sha = _sha256_file(p)
        except FileNotFoundError:
            return FileStat(False, None, None, None, None, None)
        except OSError as exc:
            raise VerificationError(f"Failed to hash local file: {p}") from exc
    else:
        sha = None
    rows = cols = None
    if include_parquet:
        try:
            pf = pq.ParquetFile(str(p))
            meta = pf.metadata
            rows, cols = meta.num_rows, meta.num_columns
        except FileNotFoundError:
            return FileStat(False, None, None, None, None, None)
        except Exception as exc:
            raise VerificationError(f"Failed to read local parquet file: {p}") from exc
    return FileStat(True, size, sha, rows, cols, mtime)


def file_stats(path: Path, *, include_sha: bool = True, include_parquet: bool = True) -> FileStat:
    p = Path(path)
    if not p.exists():
        return FileStat(False, None, None, None, None, None)
    try:
        stat = p.stat()
    except FileNotFoundError:
        return FileStat(False, None, None, None, None, None)
    except OSError as exc:
        raise VerificationError(f"Failed to stat local file: {p}") from exc
    mtime = str(int(stat.st_mtime))
    size = int(stat.st_size)
    if include_sha:
        try:
            sha = _sha256_file(p)
        except FileNotFoundError:
            return FileStat(False, None, None, None, None, None)
        except OSError as exc:
            raise VerificationError(f"Failed to hash local file: {p}") from exc
    else:
        sha = None
    rows = cols = None
    if include_parquet and p.suffix.lower() == ".parquet":
        try:
            pf = pq.ParquetFile(str(p))
            rows, cols = pf.metadata.num_rows, pf.metadata.num_columns
        except FileNotFoundError:
            return FileStat(False, None, None, None, None, None)
        except Exception as exc:
            raise VerificationError(f"Failed to read local parquet file: {p}") from exc
    return FileStat(True, size, sha, rows, cols, mtime)


def compute_file_diff(
    local_file: Path,
    remote_primary: RemoteDatasetStat | RemotePrimaryStat,
    display: str,
    *,
    verify_mode: str,
    verify_notes: List[str],
) -> DiffSummary:
    # accept either a RemoteDatasetStat.primary or a RemotePrimaryStat
    if isinstance(remote_primary, RemoteDatasetStat):
        rp = remote_primary.primary
    else:
        rp = remote_primary
    local = file_stats(
        local_file,
        include_sha=verify_mode == "hash",
        include_parquet=verify_mode == "parquet",
    )
    remote = FileStat(
        exists=rp.exists,
        size=rp.size,
        sha256=rp.sha256,
        rows=rp.rows,
        cols=rp.cols,
        mtime=rp.mtime,
    )
    # Only primary file diff; meta/events/snapshots are N/A for file mode
    changes = {
        "primary_sha_diff": _primary_diff(local, remote, verify_mode),
        "meta_mtime_diff": False,
        "events_new_remote_lines": 0,
        "snapshots_remote_newer": 0,
        "derived_files_diff": False,
    }
    has_change = bool(changes["primary_sha_diff"])
    return DiffSummary(
        dataset=display,
        primary_local=local,
        primary_remote=remote,
        meta_local_mtime=None,
        meta_remote_mtime=None,
        events_local_lines=0,
        events_remote_lines=0,
        snapshots=SnapshotStat(count=0, newest_ts=None, newer_than_local=0),
        derived_local_files=[],
        derived_remote_files=[],
        aux_local_files=[],
        aux_remote_files=[],
        has_change=has_change,
        changes=changes,
        verify_mode=verify_mode,
        verify_notes=verify_notes,
    )


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
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
    except OSError as exc:
        raise VerificationError(f"Failed to read local events log: {p}") from exc


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


def _derived_file_inventory(derived_dir: Path) -> List[str]:
    derived_dir = Path(derived_dir)
    if not derived_dir.exists():
        return []
    files: list[str] = []
    for item in sorted(derived_dir.rglob("*")):
        if not item.is_file():
            continue
        files.append(item.relative_to(derived_dir).as_posix())
    return files


def _aux_file_inventory(dataset_dir: Path) -> List[str]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return []
    files: list[str] = []
    for item in sorted(dataset_dir.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(dataset_dir)
        rel_text = rel.as_posix()
        if rel_text in {"records.parquet", "meta.md", ".events.log", ".usr.lock"}:
            continue
        if rel.parts and rel.parts[0] in {"_snapshots", "_derived"}:
            continue
        files.append(rel_text)
    return files


def compute_diff(
    dataset_dir: Path,
    remote: RemoteDatasetStat,
    dataset_name: str,
    *,
    verify_mode: str,
    verify_notes: List[str],
) -> DiffSummary:
    dataset_dir = Path(dataset_dir)
    primary_local = parquet_stats(
        dataset_dir / "records.parquet",
        include_sha=verify_mode == "hash",
        include_parquet=verify_mode == "parquet",
    )
    meta_local = file_mtime(dataset_dir / "meta.md")
    events_local = events_tail_count(dataset_dir / ".events.log")
    local_snapshot_names: list[str] = []
    local_snapshot_dir = dataset_dir / "_snapshots"
    if local_snapshot_dir.exists():
        for item in local_snapshot_dir.iterdir():
            if item.is_file() and _PAT_TS.match(item.name):
                local_snapshot_names.append(item.name)
    local_snapshot_names = sorted(local_snapshot_names)
    remote_snapshot_names = sorted(remote.snapshot_names)
    local_derived_files = _derived_file_inventory(dataset_dir / "_derived")
    remote_derived_files = sorted(remote.derived_files)
    local_aux_files = _aux_file_inventory(dataset_dir)
    remote_aux_files = sorted(remote.aux_files)

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
        "primary_sha_diff": _primary_diff(primary_local, primary_remote, verify_mode),
        "meta_mtime_diff": (meta_local != remote.meta_mtime),
        "events_new_remote_lines": max(0, remote.events_lines - events_local),
        "snapshots_name_diff": (local_snapshot_names != remote_snapshot_names),
        "snapshots_remote_newer": snaps.newer_than_local,
        "derived_files_diff": (local_derived_files != remote_derived_files),
        "aux_files_diff": (local_aux_files != remote_aux_files),
    }

    has_change = bool(
        changes["primary_sha_diff"]
        or changes["meta_mtime_diff"]
        or changes["events_new_remote_lines"] > 0
        or changes["snapshots_name_diff"]
        or changes["snapshots_remote_newer"] > 0
        or changes["derived_files_diff"]
        or changes["aux_files_diff"]
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
        derived_local_files=local_derived_files,
        derived_remote_files=remote_derived_files,
        aux_local_files=local_aux_files,
        aux_remote_files=remote_aux_files,
        has_change=has_change,
        changes=changes,
        verify_mode=verify_mode,
        verify_notes=verify_notes,
    )
