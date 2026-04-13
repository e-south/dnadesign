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
    method: str,
    n_clusters: int | None,
    seed: int | None,
    max_iter: int,
    sample_id: str | None,
    alignment_id: str | None,
    neighbor_set_id: str | None,
    metric: str | None,
    k: int,
    resolution: float,
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
    metric_value = metric or context.config.defaults.metric
    artifact_dir, summary = fit_cluster_artifact(
        context,
        cluster_id=cluster_id,
        view_id=view_id,
        method=method,
        n_clusters=n_clusters,
        seed=seed_value,
        max_iter=max_iter,
        sample_id=sample_id,
        alignment_id=alignment_id,
        neighbor_set_id=neighbor_set_id,
        metric=metric_value,
        k=k,
        resolution=resolution,
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
            "method": summary["method"],
            "view_id": view_id,
            "scope_kind": summary["scope_kind"],
            "scope_id": summary["scope_id"],
            "n_clusters": n_clusters,
            "seed": seed_value,
            "max_iter": max_iter,
            "metric": metric_value,
            "k": summary.get("k"),
            "resolution": summary.get("resolution"),
            "neighbor_set_id": neighbor_set_id,
            "iterations": summary.get("iterations"),
            "converged": summary.get("converged"),
        },
        outputs=[
            ArtifactOutput(path="assignments.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="cluster_sizes.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="medoids.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="nearest_landmarks.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="summary.json", media_type="application/json"),
        ],
        stats={"rows": summary["rows"], "n_clusters": len(summary["cluster_sizes"])},
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="cluster fit",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="cluster_set",
        artifact_id=cluster_id,
        outputs=[artifact_dir.as_posix()],
        inputs={
            "view": view_id,
            "sample": sample_id,
            "alignment": alignment_id,
            "neighbor_set": neighbor_set_id,
            "method": method,
        },
        metrics={
            "rows": summary["rows"],
            "n_clusters": len(summary["cluster_sizes"]),
            "iterations": summary.get("iterations"),
            "converged": summary.get("converged"),
            "cluster_sizes": summary["cluster_sizes"],
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="cluster_fit",
        artifact_id=cluster_id,
    )
    return result
