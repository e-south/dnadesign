"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/staging.py

Staging helpers for payload-centric YIU bundle publication.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path


def _unique_sibling_path(target: Path, *, label: str) -> Path:
    parent = target.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        candidate = parent / f".{target.name}.{label}.{uuid.uuid4().hex}"
        if not candidate.exists():
            return candidate


def create_bundle_staging_dir(bundle_dir: Path) -> Path:
    staging_dir = _unique_sibling_path(bundle_dir, label="staging")
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def remove_managed_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def promote_staged_bundle(*, staged_bundle_dir: Path, bundle_dir: Path, force_overwrite: bool) -> None:
    if not staged_bundle_dir.exists():
        raise FileNotFoundError(f"staged YIU bundle directory not found: {staged_bundle_dir}")
    if bundle_dir.exists() and not force_overwrite:
        raise ValueError(f"YIU bundle directory already exists: {bundle_dir}. Use --force-overwrite to replace it.")
    if bundle_dir.exists() and not bundle_dir.is_dir():
        raise ValueError(f"YIU bundle path is not a directory: {bundle_dir}")

    backup_dir: Path | None = None
    if bundle_dir.exists():
        backup_dir = _unique_sibling_path(bundle_dir, label="backup")
        bundle_dir.rename(backup_dir)
    try:
        staged_bundle_dir.rename(bundle_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not bundle_dir.exists():
            backup_dir.rename(bundle_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir, ignore_errors=True)


__all__ = [
    "create_bundle_staging_dir",
    "promote_staged_bundle",
    "remove_managed_path",
]
