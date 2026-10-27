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
from pathlib import Path

import yaml

from ..contracts.errors import WorkspaceValidationError
from .paths import builtin_templates_dir


def scaffold_workspace(*, workspace_dir: Path, template: str) -> Path:
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
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        (workspace_dir / "outputs" / "runs").mkdir(parents=True, exist_ok=True)
        return workspace_dir
    except Exception:
        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise
