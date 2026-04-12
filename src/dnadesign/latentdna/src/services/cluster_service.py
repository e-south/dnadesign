"""
Cluster services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..clusters.fit import fit_cluster_artifact
from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..version import __version__
from ..views.scopes import scope_input_digest_path
from ..workspaces.loader import load_workspace_config


def fit_cluster(
    workspace: str | Path,
    cluster_id: str,
    *,
    view_id: str,
    n_clusters: int,
    seed: int | None,
    max_iter: int,
    sample_id: str | None,
    alignment_id: str | None,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    cluster_dir = context.output_root / "clusters" / cluster_id
    if cluster_dir.exists() and not force:
        raise ArtifactConflictError(f"cluster artifact already exists: {cluster_dir}")
    if force and cluster_dir.exists():
        import shutil

        shutil.rmtree(cluster_dir)

    seed_value = seed if seed is not None else context.config.defaults.random_seed
    artifact_dir, rows, scope_kind, scope_id, iterations, converged, cluster_sizes = fit_cluster_artifact(
        context,
        cluster_id=cluster_id,
        view_id=view_id,
        n_clusters=n_clusters,
        seed=seed_value,
        max_iter=max_iter,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    scope_kind_input, scope_id_input, scope_digest_path = scope_input_digest_path(
        context,
        view_id=view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    manifest = ArtifactManifest(
        artifact_kind="cluster_set",
        artifact_id=cluster_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="cluster fit",
        inputs=[
            ArtifactInput(
                kind="view_matrix",
                id=view_id,
                digest=sha256_file(context.output_root / "views" / view_id / "matrix.npy"),
            ),
            ArtifactInput(kind=scope_kind_input, id=scope_id_input, digest=sha256_file(scope_digest_path)),
        ],
        params={
            "method": "kmeans",
            "view_id": view_id,
            "scope_kind": scope_kind,
            "scope_id": scope_id,
            "n_clusters": n_clusters,
            "seed": seed_value,
            "max_iter": max_iter,
            "iterations": iterations,
            "converged": converged,
        },
        outputs=[
            ArtifactOutput(path="assignments.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="summary.json", media_type="application/json"),
        ],
        stats={"rows": rows, "n_clusters": n_clusters},
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="cluster fit",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="cluster_set",
        artifact_id=cluster_id,
        outputs=[artifact_dir.as_posix()],
        inputs={"view": view_id, "sample": sample_id, "alignment": alignment_id},
        metrics={
            "rows": rows,
            "n_clusters": n_clusters,
            "iterations": iterations,
            "converged": converged,
            "cluster_sizes": cluster_sizes,
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="cluster_fit",
        artifact_id=cluster_id,
    )
    return result
