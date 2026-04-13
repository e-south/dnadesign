"""
Helpers for staging and swapping artifact directories safely.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path


def stage_artifact_dir(parent_dir: Path, artifact_id: str) -> Path:
    parent_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{artifact_id}_", dir=parent_dir))


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
            if final_dir.exists():
                raise FileExistsError(final_dir)
            staging_dir.rename(final_dir)
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
