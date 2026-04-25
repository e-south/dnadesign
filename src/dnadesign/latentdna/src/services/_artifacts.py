"""
Internal artifact path helpers for latentdna services.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

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

_MANAGED_ARTIFACT_CONFIG_FIELDS: dict[str, str] = {
    "alignment_set": "alignments",
    "export_bundle": "exports",
    "notebook": "notebooks",
    "scalar_table": "scalars",
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


def configured_artifact_ids(context: WorkspaceContext, *, artifact_kind: str) -> set[str]:
    config_field = _MANAGED_ARTIFACT_CONFIG_FIELDS.get(artifact_kind)
    if config_field is None:
        return set()
    configured = getattr(context.config, config_field, {})
    if not isinstance(configured, dict):
        return set()
    return set(configured)


def prune_retired_artifact_dirs(
    context: WorkspaceContext,
    *,
    artifact_kind: str,
) -> list[str]:
    if artifact_kind not in _MANAGED_ARTIFACT_CONFIG_FIELDS:
        return []
    configured_ids = configured_artifact_ids(context, artifact_kind=artifact_kind)
    root = context.output_root / ARTIFACT_KIND_DIRS[artifact_kind]
    if not root.is_dir():
        return []
    removed: list[str] = []
    for candidate in sorted(path for path in root.iterdir() if path.is_dir()):
        if candidate.name in configured_ids:
            continue
        if not (candidate / "manifest.json").is_file():
            continue
        shutil.rmtree(candidate)
        removed.append(candidate.as_posix())
    return removed


def prune_retired_managed_artifacts(
    context: WorkspaceContext,
    *,
    artifact_kinds: Iterable[str] | None = None,
) -> list[str]:
    kinds = list(artifact_kinds or sorted(_MANAGED_ARTIFACT_CONFIG_FIELDS))
    removed: list[str] = []
    for artifact_kind in kinds:
        removed.extend(prune_retired_artifact_dirs(context, artifact_kind=artifact_kind))
    return removed
