"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/workspaces/scaffold.py

Workspace template hydration and scaffolding for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from os.path import relpath
from pathlib import Path
from typing import Any

import yaml

from ..contracts.errors import WorkspaceValidationError
from .paths import builtin_templates_dir, resolve_repo_path


def _read_study_datasets(study_dir: Path) -> dict[str, dict[str, Any]]:
    required_files = [
        "record/campaign.yaml",
        "record/datasets.yaml",
        "record/status.md",
        "operations/ops.study.yaml",
    ]
    missing = [name for name in required_files if not (study_dir / name).exists()]
    if missing:
        raise WorkspaceValidationError(
            f"study record is missing required files in {study_dir}: {', '.join(sorted(missing))}"
        )
    datasets_path = study_dir / "record" / "datasets.yaml"
    payload = yaml.safe_load(datasets_path.read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise WorkspaceValidationError(f"study datasets registry is empty: {datasets_path}")
    by_role: dict[str, dict[str, Any]] = {}
    for entry in datasets:
        if not isinstance(entry, dict):
            raise WorkspaceValidationError(f"study dataset entries must be mappings: {datasets_path}")
        role = entry.get("role")
        if not isinstance(role, str) or not role:
            raise WorkspaceValidationError(f"study dataset entry is missing role: {entry!r}")
        by_role[role] = entry
    return by_role


def _hydrate_template_from_study(payload: dict[str, Any], *, study_dir: Path, workspace_dir: Path) -> None:
    datasets = _read_study_datasets(study_dir)
    source_role_map = {
        "merged_anchor_insert": "merged_anchor_source",
        "full_context_1kb": "construct_context",
    }
    for source_id, role in source_role_map.items():
        source_payload = payload.get("sources", {}).get(source_id)
        if not isinstance(source_payload, dict):
            continue
        study_entry = datasets.get(role)
        if study_entry is None:
            raise WorkspaceValidationError(f"study record is missing dataset role {role!r} for source {source_id!r}")
        usr_root = study_entry.get("usr_root")
        dataset_id = study_entry.get("dataset")
        if not isinstance(usr_root, str) or not usr_root:
            raise WorkspaceValidationError(f"study dataset role {role!r} is missing usr_root")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise WorkspaceValidationError(f"study dataset role {role!r} is missing dataset")
        resolved_usr_root = resolve_repo_path(usr_root)
        source_payload["kind"] = "usr"
        source_payload.pop("path", None)
        source_payload["root"] = Path(relpath(resolved_usr_root, workspace_dir)).as_posix()
        source_payload["dataset"] = dataset_id

    payload["study_binding"] = {
        "study_id": study_dir.name,
        "record_root": study_dir.resolve().as_posix(),
        "deliverable_docs_root": _deliverable_docs_root_for_study(study_dir).as_posix(),
    }


def _deliverable_docs_root_for_study(study_dir: Path) -> Path:
    analysis_root = study_dir / "analysis"
    return analysis_root.resolve() if analysis_root.is_dir() else study_dir.resolve()


def scaffold_workspace(*, workspace_dir: Path, template: str, from_study_dir: str | Path | None = None) -> Path:
    template_dir = builtin_templates_dir() / template
    if not template_dir.is_dir():
        raise WorkspaceValidationError(f"unknown workspace template: {template}")
    if workspace_dir.exists():
        raise WorkspaceValidationError(f"workspace already exists: {workspace_dir}")
    workspace_dir.mkdir(parents=True, exist_ok=False)
    try:
        for source in template_dir.rglob("*"):
            relative = source.relative_to(template_dir)
            target = workspace_dir / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        config_path = workspace_dir / "config.yaml"
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["workspace"]["id"] = workspace_dir.name
        if from_study_dir is not None:
            _hydrate_template_from_study(
                payload,
                study_dir=resolve_repo_path(from_study_dir),
                workspace_dir=workspace_dir,
            )
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        (workspace_dir / "outputs" / "runs").mkdir(parents=True, exist_ok=True)
        return workspace_dir
    except Exception:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise
