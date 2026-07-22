"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/services/_artifact_inputs.py

Internal helpers for recording managed artifact dependencies in manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.manifest import ArtifactInput
from ..io.hashing import sha256_file
from ..workspaces.loader import WorkspaceContext
from ._artifacts import artifact_manifest_path

_MANAGED_INPUT_ARTIFACT_KIND_MAP: dict[str, str] = {
    "agreement_set": "agreement_set",
    "alignment_set": "alignment_set",
    "cluster_set": "cluster_set",
    "distance_set": "distance_set",
    "enrichment_set": "enrichment_set",
    "export_bundle": "export_bundle",
    "neighbor_rows": "neighbor_set",
    "neighbor_set": "neighbor_set",
    "notebook": "notebook",
    "plot": "plot",
    "projection": "projection",
    "reducer": "reducer",
    "reduced_view": "reduced_view",
    "sample_set": "sample_set",
    "scalar_table": "scalar_table",
    "snapshot": "snapshot",
    "view": "view",
    "view_matrix": "view",
    "view_rows": "view",
}


def artifact_kind_for_input_dependency(kind: str) -> str | None:
    return _MANAGED_INPUT_ARTIFACT_KIND_MAP.get(kind)


def artifact_input_from_path(kind: str, artifact_id: str, path: Path) -> ArtifactInput:
    return ArtifactInput(
        kind=kind,
        id=artifact_id,
        digest=sha256_file(path),
        path=path.as_posix(),
    )


def artifact_input_from_manifest(
    kind: str,
    artifact_id: str,
    *,
    digest_path: Path,
    recorded_path: Path | None = None,
) -> ArtifactInput:
    target_path = recorded_path or digest_path
    return ArtifactInput(
        kind=kind,
        id=artifact_id,
        digest=sha256_file(digest_path),
        path=target_path.as_posix(),
    )


def dependency_artifact_input(
    context: WorkspaceContext,
    *,
    kind: str,
    artifact_id: str,
    path: Path,
) -> ArtifactInput:
    artifact_kind = artifact_kind_for_input_dependency(kind)
    if artifact_kind is None:
        return artifact_input_from_path(kind, artifact_id, path)
    manifest_path = artifact_manifest_path(context, artifact_kind=artifact_kind, artifact_id=artifact_id)
    return artifact_input_from_manifest(kind, artifact_id, digest_path=manifest_path)
