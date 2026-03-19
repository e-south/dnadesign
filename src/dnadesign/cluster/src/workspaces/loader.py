"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/workspaces/loader.py

Thin workspace config loading facade for cluster.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .contracts import WorkspaceConfig
from .paths import builtin_workspaces_dir, resolve_workspace_dir
from .schema import load_workspace_payload


def load_workspace_config(workspace: str | Path) -> WorkspaceConfig:
    workspace_dir = resolve_workspace_dir(workspace)
    payload = load_workspace_payload(workspace_dir / "config.yaml")
    builtin_root = builtin_workspaces_dir().resolve()
    source = "builtin" if workspace_dir.resolve().is_relative_to(builtin_root) else "local"
    return WorkspaceConfig(
        workspace_dir=workspace_dir,
        source=source,
        schema_version=payload.schema_version,
        input=payload.input,
        fit=payload.fit,
        umap=payload.umap,
        analyze=payload.analyze,
        fit_plot=payload.fit_plot,
        umap_plot=payload.umap_plot,
        analyze_plot=payload.analyze_plot,
    )


__all__ = ["load_workspace_config"]
