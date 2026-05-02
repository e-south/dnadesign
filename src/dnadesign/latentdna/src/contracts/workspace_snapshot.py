"""
Workspace snapshot contracts for study-facing status publication.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictWorkspaceSnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WorkspaceSnapshotSource(StrictWorkspaceSnapshotModel):
    kind: str
    path: str
    dataset_id: str | None = None
    row_count: int
    columns: list[str] = Field(default_factory=list)
    vector_columns: list[str] = Field(default_factory=list)


class WorkspaceSnapshotDeliverable(StrictWorkspaceSnapshotModel):
    title: str
    status: str
    freshness: str
    acceptance_checks: list[dict[str, object]] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    docs_refs: list[dict[str, str]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorkspaceSnapshotExport(StrictWorkspaceSnapshotModel):
    status: str
    artifact_path: str
    manifest_path: str
    warnings: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSnapshotBrowser(StrictWorkspaceSnapshotModel):
    default_geometry_ids: list[str] = Field(default_factory=list)
    preferred_hues: list[str] = Field(default_factory=list)
    candidate_sets: dict[str, dict[str, object]] = Field(default_factory=dict)


class WorkspaceSnapshot(StrictWorkspaceSnapshotModel):
    schema_version: Literal["latentdna.workspace_snapshot.v1"]
    workspace_id: str
    output_root: str
    sources: dict[str, WorkspaceSnapshotSource]
    model_families: list[str] = Field(default_factory=list)
    canonical_views: list[str] = Field(default_factory=list)
    deliverables: dict[str, WorkspaceSnapshotDeliverable] = Field(default_factory=dict)
    exports: dict[str, WorkspaceSnapshotExport] = Field(default_factory=dict)
    browser: WorkspaceSnapshotBrowser
    decision_ladder: list[str] = Field(default_factory=list)
    last_updated_at: str
