"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/io/artifact_dirs.py

Helpers for staging and swapping artifact directories safely.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from ..contracts.errors import ArtifactConflictError


def stage_artifact_dir(parent_dir: Path, artifact_id: str) -> Path:
    output_root = parent_dir.parent
    staging_root = output_root / "runs" / "_staging" / parent_dir.name
    staging_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{artifact_id}_", dir=staging_root))


def commit_staged_artifact_dirs(pairs: list[tuple[Path, Path]], *, force: bool) -> None:
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        if force:
            for _, final_dir in pairs:
                if final_dir.exists():
                    backup_dir = final_dir.parent / f".{final_dir.name}_backup_{uuid.uuid4().hex}"
                    final_dir.rename(backup_dir)
                    backups[final_dir] = backup_dir

        for staging_dir, final_dir in pairs:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                raise FileExistsError(final_dir)
            try:
                staging_dir.rename(final_dir)
            except OSError as exc:
                if final_dir.exists():
                    raise ArtifactConflictError(
                        "artifact materialization raced with another run: "
                        f"{final_dir}; rerun the deliverable or serialize concurrent runs"
                    ) from exc
                raise
            committed.append(final_dir)
    except Exception:
        for final_dir in reversed(committed):
            shutil.rmtree(final_dir, ignore_errors=True)
        for final_dir, backup_dir in backups.items():
            if backup_dir.exists() and not final_dir.exists():
                backup_dir.rename(final_dir)
        raise
    else:
        for backup_dir in backups.values():
            shutil.rmtree(backup_dir, ignore_errors=True)
