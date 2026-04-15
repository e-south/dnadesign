"""
Workspace loading and validation for latentdna.
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
from .paths import has_legacy_output_entries, legacy_output_root, resolve_workspace_path
from .validation import validate_workspace_config


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    workspace_dir: Path
    config_path: Path
    config: WorkspaceConfig

    @property
    def workspace_id(self) -> str:
        return self.config.workspace.id

    @property
    def output_root(self) -> Path:
        candidate = Path(self.config.workspace.output_root)
        if not candidate.is_absolute():
            candidate = self.workspace_dir / candidate
        resolved = candidate.resolve()
        if self.workspace_dir.resolve() not in resolved.parents and resolved != self.workspace_dir.resolve():
            raise WorkspaceValidationError(f"workspace output root must stay inside the workspace: {resolved}")
        return resolved

    @property
    def legacy_output_root(self) -> Path:
        return legacy_output_root(self.workspace_dir)

    @property
    def analysis_dtype(self) -> str:
        return self.config.defaults.analysis_dtype

    def assert_no_legacy_outputs(self) -> None:
        legacy_output_root = self.legacy_output_root
        if self.output_root == legacy_output_root:
            raise WorkspaceValidationError(
                f"legacy output_root is not supported for latentdna workspaces: {legacy_output_root}; use ./outputs"
            )
        if has_legacy_output_entries(legacy_output_root):
            raise WorkspaceValidationError(
                "legacy output tree is not supported for latentdna workspaces; remove "
                f"{legacy_output_root} before running latentdna commands"
            )

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


def load_workspace_config(workspace: str | Path, *, allow_legacy_outputs: bool = False) -> WorkspaceContext:
    workspace_dir = resolve_workspace_path(workspace)
    config_path = workspace_dir / "config.yaml"
    if not config_path.exists():
        raise WorkspaceValidationError(f"workspace config.yaml not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = WorkspaceConfig.model_validate(payload)
    validate_workspace_config(config)
    context = WorkspaceContext(workspace_dir=workspace_dir, config_path=config_path, config=config)
    if not allow_legacy_outputs:
        context.assert_no_legacy_outputs()
    return context
