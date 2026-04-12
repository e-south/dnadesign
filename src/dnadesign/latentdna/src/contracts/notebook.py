"""
Notebook scaffold contracts for latentdna.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SUPPORTED_NOTEBOOK_ARTIFACT_KINDS: frozenset[str] = frozenset(
    {
        "agreement_set",
        "alignment_set",
        "cluster_set",
        "distance_set",
        "enrichment_set",
        "export_bundle",
        "neighbor_set",
        "plot",
        "projection",
        "reducer",
        "reduced_view",
        "sample_set",
        "scalar_table",
        "view",
    }
)


class NotebookArtifactReference(BaseModel):
    kind: str
    id: str
    alias: str | None = None


class NotebookConfig(BaseModel):
    kind: Literal["artifact_review"]
    title: str
    description: str | None = None
    artifacts: list[NotebookArtifactReference] = Field(min_length=1)
