"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/sync_transfer.py

Staging and atomic promotion helpers for USR dataset sync transfers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .errors import VerificationError


def make_pull_staging_dir(root: Path, dataset: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset.replace("/", "__")
    return Path(tempfile.mkdtemp(prefix=f".usr-pull-{safe_dataset}-", dir=str(root)))


def copy_file_atomic(src: Path, dst: Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.usr-sync-", dir=str(dst.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def collect_staged_entries(staged: Path, *, skip_snapshots: bool) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    for src_path in sorted(Path(staged).rglob("*")):
        rel = src_path.relative_to(staged)
        if not rel.parts:
            continue
        rel_text = rel.as_posix()
        if rel_text in {"records.parquet", ".usr.lock"}:
            continue
        if skip_snapshots and rel.parts[0] == "_snapshots":
            continue
        if src_path.is_symlink():
            raise VerificationError(f"Staged pull payload contains symlink entry: {rel_text}")
        if src_path.is_dir() or src_path.is_file():
            entries.append((src_path, rel))
            continue
        raise VerificationError(f"Staged pull payload contains unsupported entry type: {rel_text}")
    return entries


def promote_staged_pull(staged: Path, dest: Path, *, primary_only: bool, skip_snapshots: bool) -> None:
    staged = Path(staged)
    dest = Path(dest)
    staged_primary = staged / "records.parquet"
    if not staged_primary.exists():
        raise VerificationError(f"Staged pull payload missing records.parquet: {staged_primary}")
    if staged_primary.is_symlink():
        raise VerificationError(f"Staged pull payload contains symlink entry: {staged_primary.name}")
    if not staged_primary.is_file():
        raise VerificationError(f"Staged pull payload contains unsupported records entry: {staged_primary.name}")

    staged_entries = collect_staged_entries(staged, skip_snapshots=skip_snapshots)

    dest.mkdir(parents=True, exist_ok=True)
    copy_file_atomic(staged_primary, dest / "records.parquet")
    if primary_only:
        return

    kept_paths: set[str] = {"records.parquet"}
    for src_path, rel in staged_entries:
        rel_text = rel.as_posix()
        kept_paths.add(rel_text)
        dst_path = dest / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        copy_file_atomic(src_path, dst_path)

    keep_with_parents: set[str] = {".usr.lock"}
    for rel_text in kept_paths:
        keep_with_parents.add(rel_text)
        parent = Path(rel_text).parent
        while str(parent) != ".":
            keep_with_parents.add(parent.as_posix())
            parent = parent.parent

    for local_path in sorted(dest.rglob("*"), key=lambda p: (len(p.parts), p.as_posix()), reverse=True):
        rel = local_path.relative_to(dest)
        rel_text = rel.as_posix()
        if rel_text in keep_with_parents:
            continue
        if skip_snapshots and rel.parts and rel.parts[0] == "_snapshots":
            continue
        if local_path.is_file() or local_path.is_symlink():
            local_path.unlink()
            continue
        try:
            local_path.rmdir()
        except OSError:
            pass
