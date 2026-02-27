"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/sync_sidecars.py

Sidecar inventory and strict-fidelity verification helpers for USR sync.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .diff import events_tail_count, file_mtime
from .errors import VerificationError
from .remote import RemoteDatasetStat


@dataclass(frozen=True)
class SidecarState:
    meta_mtime: str | None
    events_lines: int
    snapshot_names: tuple[str, ...]
    derived_files: tuple[str, ...]
    derived_hashes: tuple[tuple[str, str], ...]
    aux_files: tuple[str, ...]


_SNAPSHOT_RE = re.compile(r"^records-\d{8}T\d{6,}\.parquet$")


def ensure_sidecar_verify_compatible(
    *, verify_sidecars: bool, verify_derived_hashes: bool, primary_only: bool, skip_snapshots: bool
) -> None:
    if verify_derived_hashes and not verify_sidecars:
        raise VerificationError("--verify-derived-hashes requires sidecar verification.")
    if not verify_sidecars:
        return
    if primary_only or skip_snapshots:
        raise VerificationError(
            "--verify-sidecars requires full dataset transfer (no --primary-only/--skip-snapshots)."
        )


def _snapshot_names_from_dir(snapshot_dir: Path) -> tuple[str, ...]:
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return ()
    return tuple(
        sorted([item.name for item in snapshot_dir.iterdir() if item.is_file() and _SNAPSHOT_RE.match(item.name)])
    )


def _derived_names_from_dir(derived_dir: Path) -> tuple[str, ...]:
    derived_dir = Path(derived_dir)
    if not derived_dir.exists():
        return ()
    files: list[str] = []
    for item in sorted(derived_dir.rglob("*")):
        if not item.is_file():
            continue
        files.append(item.relative_to(derived_dir).as_posix())
    return tuple(files)


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _derived_hashes_from_dir(derived_dir: Path, derived_files: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for rel in derived_files:
        entries.append((rel, _sha256_file(Path(derived_dir) / rel)))
    return tuple(entries)


def local_sidecar_state(dataset_dir: Path, *, include_derived_hashes: bool = False) -> SidecarState:
    dataset_dir = Path(dataset_dir)
    derived_files = _derived_names_from_dir(dataset_dir / "_derived")
    aux_files: list[str] = []
    if dataset_dir.exists():
        for item in sorted(dataset_dir.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(dataset_dir)
            rel_text = rel.as_posix()
            if rel_text in {"records.parquet", "meta.md", ".events.log", ".usr.lock"}:
                continue
            if rel.parts and rel.parts[0] in {"_snapshots", "_derived"}:
                continue
            aux_files.append(rel_text)
    return SidecarState(
        meta_mtime=file_mtime(dataset_dir / "meta.md"),
        events_lines=events_tail_count(dataset_dir / ".events.log"),
        snapshot_names=_snapshot_names_from_dir(dataset_dir / "_snapshots"),
        derived_files=derived_files,
        derived_hashes=(
            _derived_hashes_from_dir(dataset_dir / "_derived", derived_files) if include_derived_hashes else ()
        ),
        aux_files=tuple(aux_files),
    )


def remote_sidecar_state(remote_stat: RemoteDatasetStat, *, include_derived_hashes: bool = False) -> SidecarState:
    return SidecarState(
        meta_mtime=remote_stat.meta_mtime,
        events_lines=int(remote_stat.events_lines),
        snapshot_names=tuple(sorted(remote_stat.snapshot_names)),
        derived_files=tuple(sorted(remote_stat.derived_files)),
        derived_hashes=(
            tuple(sorted((str(k), str(v)) for k, v in dict(remote_stat.derived_hashes).items()))
            if include_derived_hashes
            else ()
        ),
        aux_files=tuple(sorted(remote_stat.aux_files)),
    )


def verify_sidecar_state_match(local: SidecarState, remote: SidecarState, *, context: str) -> None:
    mismatches: list[str] = []
    if local.meta_mtime != remote.meta_mtime:
        mismatches.append(f"meta.md mtime local={local.meta_mtime or '-'} remote={remote.meta_mtime or '-'}")
    if local.events_lines != remote.events_lines:
        mismatches.append(f".events.log lines local={local.events_lines} remote={remote.events_lines}")
    if local.snapshot_names != remote.snapshot_names:
        mismatches.append(f"_snapshots names local={list(local.snapshot_names)} remote={list(remote.snapshot_names)}")
    if local.derived_files != remote.derived_files:
        mismatches.append(f"_derived files local={list(local.derived_files)} remote={list(remote.derived_files)}")
    if local.derived_hashes != remote.derived_hashes:
        mismatches.append(f"_derived hashes local={list(local.derived_hashes)} remote={list(remote.derived_hashes)}")
    if local.aux_files != remote.aux_files:
        mismatches.append(f"auxiliary files local={list(local.aux_files)} remote={list(remote.aux_files)}")
    if mismatches:
        raise VerificationError(f"{context}: sidecar mismatch; " + "; ".join(mismatches))
