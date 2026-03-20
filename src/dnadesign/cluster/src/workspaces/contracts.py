"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/workspaces/contracts.py

Typed workspace contracts for cluster.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    workspace_dir: Path
    source: str
    schema_version: int
    input: dict[str, Any]
    fit: dict[str, Any]
    umap: dict[str, Any]
    analyze: dict[str, Any]
    fit_plot: dict[str, Any]
    umap_plot: dict[str, Any]
    analyze_plot: dict[str, Any]

    @property
    def workspace_id(self) -> str:
        return self.workspace_dir.name

    @property
    def config_path(self) -> Path:
        return self.workspace_dir / "config.yaml"

    @property
    def results_root(self) -> Path:
        if self.source == "builtin":
            return (Path.cwd() / "workspaces" / self.workspace_id / "outputs" / "cluster").resolve()
        return self.workspace_dir / "outputs" / "cluster"

    def section_params(self, section: str) -> dict[str, Any]:
        if section == "fit":
            base = dict(self.input)
            base.update(self.fit)
            return base
        if section == "umap":
            base = dict(self.input)
            base.update(self.umap)
            return base
        if section == "analyze":
            base = dict(self.input)
            base.update(self.analyze)
            return base
        raise KeyError(f"Unsupported workspace section '{section}'.")

    def section_plot(self, section: str) -> dict[str, Any]:
        if section == "fit":
            return dict(self.fit_plot)
        if section == "umap":
            return dict(self.umap_plot)
        if section == "analyze":
            return dict(self.analyze_plot)
        raise KeyError(f"Unsupported workspace section '{section}'.")
