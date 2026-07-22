"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/workspaces/loader.py

Workspace loading and validation for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..contracts.errors import WorkspaceValidationError
from ..contracts.notebook import NotebookConfig
from ..contracts.workspace import (
    AlignmentConfig,
    CohortConfig,
    DeliverableConfig,
    ExportConfig,
    LandmarkConfig,
    RecipeConfig,
    ScalarConfig,
    SourceBackedViewConfig,
    WorkspaceConfig,
)
from ..io.json_io import read_json
from .paths import resolve_workspace_path
from .validation import validate_workspace_config


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    workspace_dir: Path
    config_path: Path
    config: WorkspaceConfig
    _output_root: Path

    @property
    def workspace_id(self) -> str:
        return self.config.workspace.id

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def analysis_dtype(self) -> str:
        return self.config.defaults.analysis_dtype

    def require_source(self, source_id: str):
        if source_id not in self.config.sources:
            raise WorkspaceValidationError(f"unknown source: {source_id}")
        return self.config.sources[source_id]

    def require_view(self, view_id: str):
        if view_id not in self.config.views:
            raise WorkspaceValidationError(f"unknown view: {view_id}")
        return self.config.views[view_id]

    def require_source_view(self, view_id: str) -> SourceBackedViewConfig:
        view = self.require_view(view_id)
        if not isinstance(view, SourceBackedViewConfig):
            raise WorkspaceValidationError(f"view {view_id} is not a source-backed view")
        return view

    def require_alignment(self, alignment_id: str) -> AlignmentConfig:
        if alignment_id not in self.config.alignments:
            raise WorkspaceValidationError(f"unknown alignment: {alignment_id}")
        return self.config.alignments[alignment_id]

    def require_scalar(self, scalar_id: str) -> ScalarConfig:
        if scalar_id not in self.config.scalars:
            raise WorkspaceValidationError(f"unknown scalar: {scalar_id}")
        return self.config.scalars[scalar_id]

    def require_landmark(self, landmark_id: str) -> LandmarkConfig:
        if landmark_id not in self.config.landmarks:
            raise WorkspaceValidationError(f"unknown landmark: {landmark_id}")
        return self.config.landmarks[landmark_id]

    def require_plot(self, plot_id: str):
        if plot_id not in self.config.plots:
            raise WorkspaceValidationError(f"unknown plot recipe: {plot_id}")
        return self.config.plots[plot_id]

    def require_cohort(self, cohort_id: str) -> CohortConfig:
        if cohort_id not in self.config.cohorts:
            raise WorkspaceValidationError(f"unknown cohort: {cohort_id}")
        return self.config.cohorts[cohort_id]

    def require_export(self, export_id: str) -> ExportConfig:
        if export_id not in self.config.exports:
            raise WorkspaceValidationError(f"unknown export: {export_id}")
        return self.config.exports[export_id]

    def require_notebook(self, notebook_id: str) -> NotebookConfig:
        if notebook_id not in self.config.notebooks:
            raise WorkspaceValidationError(f"unknown notebook: {notebook_id}")
        return self.config.notebooks[notebook_id]

    def require_recipe(self, recipe_id: str) -> RecipeConfig:
        if recipe_id not in self.config.recipes:
            raise WorkspaceValidationError(f"unknown recipe: {recipe_id}")
        return self.config.recipes[recipe_id]

    def require_deliverable(self, deliverable_id: str) -> DeliverableConfig:
        if deliverable_id not in self.config.deliverables:
            raise WorkspaceValidationError(f"unknown deliverable: {deliverable_id}")
        return self.config.deliverables[deliverable_id]

    def require_source_backed_reference_source(self, ref_id: str):
        if ref_id in self.config.sources:
            return self.require_source(ref_id)
        view = self.require_source_view(ref_id)
        return self.require_source(view.source)

    def read_manifest(self, path: Path) -> dict[str, Any]:
        return read_json(path)


def load_workspace_config(workspace: str | Path, *, validate_plot_semantics: bool = False) -> WorkspaceContext:
    workspace_dir = resolve_workspace_path(workspace)
    config_path = workspace_dir / "config.yaml"
    if not config_path.exists():
        raise WorkspaceValidationError(f"workspace config.yaml not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = WorkspaceConfig.model_validate(payload)
    validate_workspace_config(config)
    context = WorkspaceContext(
        workspace_dir=workspace_dir,
        config_path=config_path,
        config=config,
        _output_root=_resolve_output_root(workspace_dir=workspace_dir, config=config),
    )
    if validate_plot_semantics:
        from .plot_semantics import validate_plot_semantics_sidecars

        validate_plot_semantics_sidecars(context)
    return context


def _resolve_output_root(*, workspace_dir: Path, config: WorkspaceConfig) -> Path:
    candidate = Path(config.workspace.output_root)
    if not candidate.is_absolute():
        candidate = workspace_dir / candidate
    resolved = candidate.resolve()
    workspace_root = workspace_dir.resolve()
    if workspace_root not in resolved.parents and resolved != workspace_root:
        raise WorkspaceValidationError(f"workspace output root must stay inside the workspace: {resolved}")
    required = (workspace_root / "outputs").resolve()
    if resolved != required:
        raise WorkspaceValidationError(f"workspace output_root must resolve to {required}; got {resolved}")
    return resolved
