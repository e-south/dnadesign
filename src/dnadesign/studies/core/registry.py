"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/registry.py

Checked-in study index loading for flat study-first record roots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dnadesign.ops.catalog import discover_repo_root
from dnadesign.ops.status import resolve_path_ref


@dataclass(frozen=True)
class StudyIndexEntry:
    study_id: str
    record_root: Path
    title: str | None = None
    raw_payload: dict[str, object] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class StudyIndex:
    repo_root: Path
    index_path: Path
    active_study_id: str
    studies: tuple[StudyIndexEntry, ...]

    @property
    def study_index(self) -> dict[str, StudyIndexEntry]:
        return {entry.study_id: entry for entry in self.studies}


def load_study_index(repo_root: Path | None) -> StudyIndex:
    resolved_repo_root = repo_root.expanduser().resolve() if repo_root is not None else discover_repo_root(Path.cwd())
    if resolved_repo_root is None:
        raise ValueError("checked-in study index requires a dnadesign repository checkout with docs/studies/index.yaml")

    index_path = resolved_repo_root / "docs" / "studies" / "index.yaml"
    if not index_path.exists():
        raise ValueError(f"checked-in study index not found: {index_path}")

    payload = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"checked-in study index must be a mapping: {index_path}")
    version = int(payload.get("version") or 0)
    if version != 1:
        raise ValueError(f"unsupported checked-in study index version {version}: {index_path}")

    active_study_id = _required_text(payload.get("active_study_id"), label="active_study_id", source=index_path)
    studies_payload = payload.get("studies") or []
    if not isinstance(studies_payload, list) or not studies_payload:
        raise ValueError(f"checked-in study index must define a non-empty studies list: {index_path}")

    studies: list[StudyIndexEntry] = []
    seen_study_ids: set[str] = set()
    for index, item in enumerate(studies_payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"study index entry {index} must be a mapping: {index_path}")
        if "family" in item:
            raise ValueError(
                "checked-in study index entries must not define legacy family; "
                f"use the study record's explicit ops_surfaces instead: {index_path}"
            )
        study_id = _required_text(item.get("study_id"), label="study_id", source=index_path)
        if study_id in seen_study_ids:
            raise ValueError(f"checked-in study index must not duplicate study_id {study_id!r}: {index_path}")
        seen_study_ids.add(study_id)
        raw_record_root = _required_text(
            item.get("record_root"), label=f"studies.{study_id}.record_root", source=index_path
        )
        record_root = resolve_path_ref(
            raw_record_root,
            repo_root=resolved_repo_root,
            default_base="repo",
            label=f"studies.{study_id}.record_root",
        )
        studies.append(
            StudyIndexEntry(
                study_id=study_id,
                title=_string_or_none(item.get("title")),
                record_root=record_root,
                raw_payload=dict(item),
            )
        )

    if active_study_id not in seen_study_ids:
        raise ValueError(f"active_study_id '{active_study_id}' is not declared under studies in {index_path}")

    return StudyIndex(
        repo_root=resolved_repo_root,
        index_path=index_path,
        active_study_id=active_study_id,
        studies=tuple(studies),
    )


def _required_text(value: object, *, label: str, source: Path) -> str:
    text = _string_or_none(value)
    if text is None:
        raise ValueError(f"{label} is required in {source}")
    return text


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


__all__ = ["StudyIndex", "StudyIndexEntry", "load_study_index"]
