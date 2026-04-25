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
from ..views.scopes import matrix_input_digest_path, scope_input_digest_path
from ..workspaces.loader import load_workspace_config
from .memory_service import apply_memory_preflight, evaluate_cluster_preflight


def fit_cluster(
    workspace: str | Path,
    cluster_id: str,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
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
    allow_memory_overage: bool = False,
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
    preflight = evaluate_cluster_preflight(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
        method=method,
        n_clusters=n_clusters,
        k=k,
        sample_id=sample_id,
        alignment_id=alignment_id,
        neighbor_set_id=neighbor_set_id,
    )
    status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)
    artifact_dir, summary = fit_cluster_artifact(
        context,
        cluster_id=cluster_id,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
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
    matrix_input_kind, matrix_input_id, matrix_input_path = matrix_input_digest_path(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
    )
    scope_kind_input, scope_id_input, scope_digest_path = scope_input_digest_path(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
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
        status=status,
        inputs=[
            ArtifactInput(
                kind=matrix_input_kind,
                id=matrix_input_id,
                digest=sha256_file(matrix_input_path),
                path=matrix_input_path.as_posix(),
            ),
            ArtifactInput(
                kind=scope_kind_input,
                id=scope_id_input,
                digest=sha256_file(scope_digest_path),
                path=scope_digest_path.as_posix(),
            ),
        ],
        params={
            "method": summary["method"],
            "view_id": view_id,
            "reduced_view_id": reduced_view_id,
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
            "memory_preflight": preflight.as_payload(),
        },
        outputs=[
            ArtifactOutput(path="assignments.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="cluster_sizes.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="medoids.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="nearest_landmarks.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="summary.json", media_type="application/json"),
        ],
        stats={"rows": summary["rows"], "n_clusters": len(summary["cluster_sizes"])},
        warnings=warnings,
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="cluster fit",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="cluster_set",
        artifact_id=cluster_id,
        outputs=[artifact_dir.as_posix()],
        inputs={
            "view": view_id,
            "reduced_view": reduced_view_id,
            "sample": sample_id,
            "alignment": alignment_id,
            "neighbor_set": neighbor_set_id,
            "method": method,
        },
        warnings=warnings,
        metrics={
            "rows": summary["rows"],
            "n_clusters": len(summary["cluster_sizes"]),
            "iterations": summary.get("iterations"),
            "converged": summary.get("converged"),
            "cluster_sizes": summary["cluster_sizes"],
            "memory_preflight": preflight.as_payload(),
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="cluster_fit",
        artifact_id=cluster_id,
    )
    return result
