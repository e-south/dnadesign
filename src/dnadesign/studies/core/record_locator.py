"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/record_locator.py

Active-study selection helpers for the flat checked-in study registry.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.catalog import discover_repo_root

from .registry import StudyIndexEntry, load_study_index


@dataclass(frozen=True)
class ActiveStudySelection:
    repo_root: Path
    index_path: Path
    active_study_id: str
    family: str
    study_root: Path
    entry: StudyIndexEntry


def discover_active_study_selection(
    *,
    repo_root: Path | None,
    status_kind: str,
) -> ActiveStudySelection:
    resolved_repo_root = repo_root.expanduser().resolve() if repo_root is not None else discover_repo_root(Path.cwd())
    if resolved_repo_root is None:
        raise ValueError(
            f"status kind '{status_kind}' requires --study-dir or a dnadesign repository checkout "
            "with docs/studies/index.yaml"
        )
    study_index = load_study_index(resolved_repo_root)
    entry = study_index.study_index.get(study_index.active_study_id)
    if entry is None:
        raise ValueError(
            f"active_study_id '{study_index.active_study_id}' is not declared under studies in {study_index.index_path}"
        )
    return ActiveStudySelection(
        repo_root=study_index.repo_root,
        index_path=study_index.index_path,
        active_study_id=study_index.active_study_id,
        family=entry.family,
        study_root=entry.record_root,
        entry=entry,
    )


__all__ = ["ActiveStudySelection", "discover_active_study_selection"]
