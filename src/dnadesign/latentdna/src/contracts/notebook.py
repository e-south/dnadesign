"""
Notebook scaffold contracts for latentdna.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class StrictNotebookModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NotebookArtifactReference(StrictNotebookModel):
    kind: str
    id: str
    alias: str | None = None


class ArtifactReviewNotebookConfig(StrictNotebookModel):
    kind: Literal["artifact_review"]
    title: str
    description: str | None = None
    artifacts: list[NotebookArtifactReference] = Field(min_length=1)


class WorkspaceBrowserNotebookConfig(StrictNotebookModel):
    kind: Literal["workspace_browser"]
    title: str
    description: str | None = None
    default_deliverable: str


NotebookConfig = Annotated[
    ArtifactReviewNotebookConfig | WorkspaceBrowserNotebookConfig,
    Field(discriminator="kind"),
]
