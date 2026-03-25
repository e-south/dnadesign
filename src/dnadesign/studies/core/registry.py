"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/registry.py

Generic checked-in study registry discovery for family-owned study packages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.ops.catalog import discover_repo_root
from dnadesign.ops.status.path_ref import resolve_path_ref


@dataclass(frozen=True)
class ActiveStudySelection:
    repo_root: Path
    index_path: Path
    active_study: str
    study_root: Path


def discover_active_study_selection(
    *,
    repo_root: Path | None,
    family_id: str,
    status_kind: str,
) -> ActiveStudySelection:
    resolved_repo_root = repo_root.expanduser().resolve() if repo_root is not None else discover_repo_root(Path.cwd())
    if resolved_repo_root is None:
        raise ValueError(
            f"status kind '{status_kind}' requires --study-dir or a dnadesign repository checkout "
            f"with docs/studies/{family_id}/index.yaml"
        )

    index_path = resolved_repo_root / "docs" / "studies" / family_id / "index.yaml"
    if not index_path.exists():
        raise ValueError(f"{family_id} study registry not found: {index_path}")

    payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{family_id} study registry must be a mapping: {index_path}")
    active_study = _required_text(payload.get("active_study"), label="active_study", source=index_path)
    studies_payload = payload.get("studies") or []
    if not isinstance(studies_payload, list):
        raise ValueError(f"{family_id} study registry must define a 'studies' list: {index_path}")

    matches = [
        entry
        for entry in studies_payload
        if isinstance(entry, dict) and _string_or_none(entry.get("study_id")) == active_study
    ]
    if not matches:
        raise ValueError(f"active_study '{active_study}' is not declared under 'studies' in {index_path}")
    if len(matches) > 1:
        raise ValueError(f"active_study '{active_study}' is declared more than once in {index_path}")

    raw_path = _required_text(matches[0].get("path"), label="study path", source=index_path)
    study_root = resolve_path_ref(
        raw_path,
        repo_root=resolved_repo_root,
        default_base="repo",
        label="study path",
    )
    return ActiveStudySelection(
        repo_root=resolved_repo_root,
        index_path=index_path,
        active_study=active_study,
        study_root=study_root,
    )


def _required_text(value: object, *, label: str, source: Path) -> str:
    text = _string_or_none(value)
    if text is None:
        raise ValueError(f"{label} is required in {source}")
    return text


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = ["ActiveStudySelection", "discover_active_study_selection"]
