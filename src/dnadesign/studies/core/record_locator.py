"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/record_locator.py

Active-study selection helpers for the flat checked-in study registry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.ops.catalog import discover_repo_root

from .record_loader import load_study_ops_contract
from .registry import StudyIndexEntry, load_study_index


@dataclass(frozen=True)
class ActiveStudySelection:
    repo_root: Path
    index_path: Path
    active_study_id: str
    study_root: Path
    entry: StudyIndexEntry


def discover_active_study_selection(
    *,
    repo_root: Path | None,
    status_kind: str,
) -> ActiveStudySelection:
    resolved_repo_root = _resolve_repo_root(repo_root=repo_root, status_kind=status_kind)
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
        study_root=entry.record_root,
        entry=entry,
    )


def discover_study_selection_for_status_kind(
    *,
    repo_root: Path | None,
    status_kind: str,
) -> ActiveStudySelection:
    requested_status_kind = str(status_kind or "").strip()
    if not requested_status_kind:
        raise ValueError("status kind is required to select a checked-in study record")
    resolved_repo_root = _resolve_repo_root(repo_root=repo_root, status_kind=requested_status_kind)
    study_index = load_study_index(resolved_repo_root)

    matches: list[StudyIndexEntry] = []
    for entry in study_index.studies:
        contract = load_study_ops_contract(entry.record_root)
        if requested_status_kind in {contract.status_kind, contract.preflight_kind}:
            matches.append(entry)

    if not matches:
        raise ValueError(
            f"status kind '{requested_status_kind}' is not declared by any checked-in study "
            f"ops_surfaces.status_kind or ops_surfaces.preflight_kind in {study_index.index_path}"
        )
    if len(matches) > 1:
        study_ids = ", ".join(entry.study_id for entry in matches)
        raise ValueError(
            f"status kind '{requested_status_kind}' is declared by multiple checked-in study records "
            f"in {study_index.index_path}: {study_ids}"
        )

    entry = matches[0]
    return ActiveStudySelection(
        repo_root=study_index.repo_root,
        index_path=study_index.index_path,
        active_study_id=study_index.active_study_id,
        study_root=entry.record_root,
        entry=entry,
    )


def _resolve_repo_root(*, repo_root: Path | None, status_kind: str) -> Path:
    resolved_repo_root = repo_root.expanduser().resolve() if repo_root is not None else discover_repo_root(Path.cwd())
    if resolved_repo_root is None:
        raise ValueError(
            f"status kind '{status_kind}' requires --study-dir or a dnadesign repository checkout "
            "with docs/studies/index.yaml"
        )
    return resolved_repo_root


__all__ = [
    "ActiveStudySelection",
    "discover_active_study_selection",
    "discover_study_selection_for_status_kind",
]
