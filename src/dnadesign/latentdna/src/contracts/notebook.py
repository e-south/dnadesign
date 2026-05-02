"""Notebook scaffold and control-plane contracts for latentdna."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .candidate_inventory import CandidateInventoryRow

HueKind = Literal["categorical", "binary", "continuous", "ordinal"]
NotebookSurface = Literal["plots", "geometry_browser"]


class StrictNotebookModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceNotebookConfig(StrictNotebookModel):
    kind: Literal["workspace"]
    title: str
    description: str | None = None
    default_deliverable: str
    default_surface: NotebookSurface = "plots"
    candidate_sets: list[str] = Field(default_factory=list)
    default_candidate_set: str | None = None
    default_reference_set: str | None = None
    ordered_plots: list[str] = Field(default_factory=list)
    context_audit_scalar_ids: list[str] = Field(default_factory=list)
    geometry_order: list[str] = Field(default_factory=list)
    candidate_grid_views: list[str] = Field(default_factory=list)
    candidate_grid_panel_titles: list[str] = Field(default_factory=list)
    preferred_hues: list[str] = Field(default_factory=list)
    preferred_hue_kinds: dict[str, HueKind] = Field(default_factory=dict)
    default_layout: str | None = None
    default_compare_views: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_geometry_controls(self) -> "WorkspaceNotebookConfig":
        if self.candidate_grid_panel_titles and not self.candidate_grid_views:
            raise ValueError("candidate_grid_panel_titles require candidate_grid_views")
        if self.candidate_grid_panel_titles and len(self.candidate_grid_panel_titles) != len(self.candidate_grid_views):
            raise ValueError("candidate_grid_panel_titles must match candidate_grid_views length")
        if self.default_compare_views and len(self.default_compare_views) != 2:
            raise ValueError("default_compare_views must declare exactly two view ids when provided")
        if self.default_candidate_set is not None and self.candidate_sets:
            if self.default_candidate_set not in self.candidate_sets:
                raise ValueError("default_candidate_set must be declared in candidate_sets when candidate_sets are set")
        return self


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
    view_ids: list[str] = Field(default_factory=list)


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


class WorkspaceNotebookReferenceSet(StrictNotebookModel):
    reference_set_id: str
    label: str | None = None
    match_column: str
    label_column: str | None = None
    label_mode: Literal["label_and_highlight", "highlight_only"]
    explicit_ids: list[str] = Field(default_factory=list)
    selector_columns: list[str] = Field(default_factory=list)


class WorkspaceNotebookCompareMetrics(StrictNotebookModel):
    sample_rows: int
    distance_pair_limit: int
    knn_k: int


class WorkspaceNotebookCandidateView(StrictNotebookModel):
    view_id: str
    label: str
    panel_title: str
    status: str
    role: str | None = None
    model: str | None = None
    family: str | None = None
    scope: str | None = None
    coordinate_space_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    materialized: bool
    rows: int | None = None
    dims: int | None = None


class WorkspaceNotebookCandidateSet(StrictNotebookModel):
    candidate_set_id: str
    label: str
    description: str | None = None
    view_ids: list[str] = Field(default_factory=list)
    available_view_ids: list[str] = Field(default_factory=list)
    panel_titles: list[str] = Field(default_factory=list)
    views: list[WorkspaceNotebookCandidateView] = Field(default_factory=list)


class WorkspaceNotebookGeometryControls(StrictNotebookModel):
    default_model: str
    default_family: str
    default_context: str
    default_layout: str
    default_reference_set: str = ""
    default_compare_left: str | None = None
    default_compare_right: str | None = None
    geometries: list[WorkspaceNotebookGeometry]
    preferred_hues: list[str]
    row_metadata_hues: list[str] = Field(default_factory=list)
    hue_kinds: dict[str, HueKind] = Field(default_factory=dict)
    joinable_tables: list[WorkspaceNotebookTableRef]
    layout_presets: list[WorkspaceNotebookLayoutPreset]
    comparison_bases: list[WorkspaceNotebookComparisonBasis]
    reference_labels: list[str]
    reference_sets: list[WorkspaceNotebookReferenceSet] = Field(default_factory=list)
    candidate_sets: list[WorkspaceNotebookCandidateSet] = Field(default_factory=list)
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


class WorkspaceNotebookRuntimePaths(StrictNotebookModel):
    workspace_relative_path: str
    output_relative_path: str
    catalog_relative_path: str
    health_relative_path: str


class WorkspaceNotebookPlotEntry(StrictNotebookModel):
    plot_id: str
    deliverable_id: str
    deliverable_title: str
    visibility_tier: Literal["primary", "appendix", "debug", "hidden"]
    status: str | None = None
    stale: bool = False


class WorkspaceNotebookPlotControls(StrictNotebookModel):
    default_surface: NotebookSurface
    ordered_plot_ids: list[str]
    plots: list[WorkspaceNotebookPlotEntry]


class WorkspaceNotebookControls(StrictNotebookModel):
    schema_version: Literal["latentdna.workspace_notebook_controls.v4"]
    workspace_id: str
    notebook_id: str
    generated_at: str
    runtime_paths: WorkspaceNotebookRuntimePaths
    candidate_inventory: list[CandidateInventoryRow] = Field(default_factory=list)
    plot_controls: WorkspaceNotebookPlotControls
    geometry_controls: WorkspaceNotebookGeometryControls
    context_audit: WorkspaceNotebookContextAudit


NotebookConfig = WorkspaceNotebookConfig
