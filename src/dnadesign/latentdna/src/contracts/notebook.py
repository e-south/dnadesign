"""Notebook scaffold and control-plane contracts for latentdna."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictNotebookModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceNotebookConfig(StrictNotebookModel):
    kind: Literal["workspace"]
    title: str
    description: str | None = None
    default_deliverable: str


class WorkspaceNotebookGeometry(StrictNotebookModel):
    view_id: str
    label: str
    model: str
    family: str
    context: str
    role: str | None = None
    materialized: bool
    projection_ids: list[str]
    coordinate_space_id: str | None = None
    rows: int | None = None
    dims: int | None = None


class WorkspaceNotebookTableRef(StrictNotebookModel):
    kind: Literal["scalar", "distance", "cluster"]
    artifact_id: str
    relative_path: str
    columns: list[str]
    manifest_path: str | None = None


class WorkspaceNotebookLayoutPreset(StrictNotebookModel):
    id: str
    label: str
    mode: Literal["single_view", "model_pair", "fixed_grid"]
    description: str
    view_ids: list[str] = Field(default_factory=list)
    panel_titles: list[str] = Field(default_factory=list)
    view_order: list[str] = Field(default_factory=list)


class WorkspaceNotebookComparisonBasis(StrictNotebookModel):
    id: str
    kind: Literal["alignment"]
    alignment_id: str
    left_view: str
    right_view: str
    support: str
    left_aggregation: str | None = None
    right_aggregation: str | None = None
    label: str


class WorkspaceNotebookCompareMetrics(StrictNotebookModel):
    sample_rows: int
    distance_pair_limit: int
    knn_k: int


class WorkspaceNotebookSwitchboardControls(StrictNotebookModel):
    default_model: str
    default_family: str
    default_context: str
    default_layout: str
    default_compare_left: str | None = None
    default_compare_right: str | None = None
    geometries: list[WorkspaceNotebookGeometry]
    preferred_hues: list[str]
    joinable_tables: list[WorkspaceNotebookTableRef]
    layout_presets: list[WorkspaceNotebookLayoutPreset]
    comparison_bases: list[WorkspaceNotebookComparisonBasis]
    reference_labels: list[str]
    compare_metrics: WorkspaceNotebookCompareMetrics


class WorkspaceNotebookContextAudit(StrictNotebookModel):
    artifact_id: str
    status: str
    decision: str
    rule: dict[str, object]
    error: str | None = None
    rows: int | None = None
    table_path: str | None = None
    metrics: dict[str, object] | None = None


class WorkspaceNotebookControls(StrictNotebookModel):
    schema_version: Literal["latentdna.workspace_notebook_controls.v1"]
    workspace_id: str
    notebook_id: str
    generated_at: str
    geometry_switchboard: WorkspaceNotebookSwitchboardControls
    context_audit: WorkspaceNotebookContextAudit


NotebookConfig = WorkspaceNotebookConfig
