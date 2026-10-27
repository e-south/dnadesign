"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/contracts/deliverable.py

Deliverable status contracts for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ARTIFACT_REFERENCE_CATEGORIES: dict[str, str] = {
    "agreements": "agreement_set",
    "alignments": "alignment_set",
    "clusters": "cluster_set",
    "distances": "distance_set",
    "enrichments": "enrichment_set",
    "exports": "export_bundle",
    "neighbors": "neighbor_set",
    "notebooks": "notebook",
    "plots": "plot",
    "projections": "projection",
    "reducers": "reducer",
    "reduced_views": "reduced_view",
    "samples": "sample_set",
    "scalars": "scalar_table",
    "views": "view",
}

CONFIG_REFERENCE_CATEGORIES: dict[str, str] = {
    "cohorts": "cohort",
    "landmarks": "landmark",
    "recipes": "recipe",
    "sources": "source",
}

SUPPORTED_DELIVERABLE_REFERENCE_CATEGORIES: frozenset[str] = frozenset(
    set(ARTIFACT_REFERENCE_CATEGORIES) | set(CONFIG_REFERENCE_CATEGORIES)
)

SINGULAR_REFERENCE_NAMES: dict[str, str] = {
    **ARTIFACT_REFERENCE_CATEGORIES,
    **CONFIG_REFERENCE_CATEGORIES,
}


class DeliverableEntryStatus(BaseModel):
    name: str
    status: Literal["ok", "attention", "missing", "error"]
    reason: str | None = None
    path: str | None = None


class DeliverableStatusResult(BaseModel):
    schema_version: Literal["latentdna.deliverable_status.v1"] = "latentdna.deliverable_status.v1"
    deliverable_id: str
    title: str
    section: str
    question: str
    summary: str
    status: Literal["ok", "attention", "missing", "error"]
    checks: list[DeliverableEntryStatus] = Field(default_factory=list)
    outputs: list[DeliverableEntryStatus] = Field(default_factory=list)
    docs_refs: list[dict[str, str]] = Field(default_factory=list)
    acceptance_checks: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
