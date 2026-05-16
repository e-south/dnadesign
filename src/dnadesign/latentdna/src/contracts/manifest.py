"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/contracts/manifest.py

Artifact manifest contracts for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactInput(BaseModel):
    kind: str
    id: str
    digest: str
    path: str | None = None


class ArtifactOutput(BaseModel):
    path: str
    media_type: str


class ArtifactManifest(BaseModel):
    schema_version: Literal["latentdna.manifest.v1"] = "latentdna.manifest.v1"
    artifact_kind: str
    artifact_id: str
    workspace_id: str
    created_at: str
    tool_version: str
    git_commit: str | None = None
    run_id: str | None = None
    command: str
    status: Literal["ok", "attention", "missing", "error"] = "ok"
    inputs: list[ArtifactInput] = Field(default_factory=list)
    input_digests: dict[str, str] = Field(default_factory=dict)
    freshness_basis: dict[str, Any] = Field(default_factory=dict)
    source_provenance: list[dict[str, Any]] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    outputs: list[ArtifactOutput] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    semantics: dict[str, Any] | None = None
