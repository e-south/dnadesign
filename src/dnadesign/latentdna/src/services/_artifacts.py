"""
Internal artifact path helpers for latentdna services.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.deliverable import ARTIFACT_REFERENCE_CATEGORIES
from ..contracts.errors import ContractViolationError
from ..workspaces.loader import WorkspaceContext

ARTIFACT_KIND_DIRS: dict[str, str] = {
    "agreement_set": "agreements",
    "alignment_set": "alignments",
    "cluster_set": "clusters",
    "distance_set": "distances",
    "enrichment_set": "enrichments",
    "export_bundle": "exports",
    "neighbor_set": "neighbors",
    "notebook": "notebooks",
    "plot": "plots",
    "projection": "projections",
    "reducer": "reducers",
    "reduced_view": "reduced_views",
    "sample_set": "samples",
    "scalar_table": "scalars",
    "snapshot": "snapshots",
    "view": "views",
}


def artifact_kind_for_category(category: str) -> str:
    if category not in ARTIFACT_REFERENCE_CATEGORIES:
        raise ContractViolationError(f"unsupported artifact reference category: {category}")
    return ARTIFACT_REFERENCE_CATEGORIES[category]


def artifact_dir(context: WorkspaceContext, *, artifact_kind: str, artifact_id: str) -> Path:
    if artifact_kind not in ARTIFACT_KIND_DIRS:
        raise ContractViolationError(f"unsupported artifact kind: {artifact_kind}")
    return context.output_root / ARTIFACT_KIND_DIRS[artifact_kind] / artifact_id


def artifact_manifest_path(context: WorkspaceContext, *, artifact_kind: str, artifact_id: str) -> Path:
    return artifact_dir(context, artifact_kind=artifact_kind, artifact_id=artifact_id) / "manifest.json"


def artifact_exists(context: WorkspaceContext, *, artifact_kind: str, artifact_id: str) -> bool:
    return artifact_manifest_path(context, artifact_kind=artifact_kind, artifact_id=artifact_id).is_file()
