"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/services/neighbors_service.py

Neighbor services for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.manifest_io import write_manifest
from ..neighbors.fit import fit_neighbor_artifact
from ..runs.recorder import record_audit
from ..version import __version__
from ..views.scopes import matrix_input_digest_path, scope_input_digest_path
from ..workspaces.loader import load_workspace_config
from ._artifact_inputs import dependency_artifact_input
from .memory_service import (
    apply_memory_preflight,
    approximate_backend_warning,
    evaluate_neighbors_preflight,
    merge_attention_status,
)


def fit_neighbors(
    workspace: str | Path,
    neighbor_id: str,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
    k: int,
    metric: str | None,
    backend: str | None,
    sample_id: str | None,
    alignment_id: str | None,
    seed: int | None,
    allow_memory_overage: bool = False,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    neighbor_dir = context.output_root / "neighbors" / neighbor_id
    if neighbor_dir.exists() and not force:
        raise ArtifactConflictError(f"neighbor artifact already exists: {neighbor_dir}")
    if force and neighbor_dir.exists():
        import shutil

        shutil.rmtree(neighbor_dir)

    metric_value = metric or context.config.defaults.metric
    backend_value = backend or context.config.defaults.neighbor_backend
    seed_value = seed if seed is not None else context.config.defaults.random_seed
    preflight = evaluate_neighbors_preflight(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
        k=k,
        backend=backend_value,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    preflight_status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)
    artifact_dir, rows, resolved_backend, approximate, scope_kind, scope_id = fit_neighbor_artifact(
        context,
        neighbor_id=neighbor_id,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
        k=k,
        metric=metric_value,
        backend=backend_value,
        seed=seed_value,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )
    degraded_status, degraded_warnings = approximate_backend_warning(
        requested_backend=backend_value,
        resolved_backend=resolved_backend,
    )
    warnings = [*warnings, *degraded_warnings]
    status = merge_attention_status([preflight_status, degraded_status], warnings)
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
        artifact_kind="neighbor_set",
        artifact_id=neighbor_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="neighbors fit",
        status=status,
        inputs=[
            dependency_artifact_input(
                context,
                kind=matrix_input_kind,
                artifact_id=matrix_input_id,
                path=matrix_input_path,
            ),
            dependency_artifact_input(
                context,
                kind=scope_kind_input,
                artifact_id=scope_id_input,
                path=scope_digest_path,
            ),
        ],
        params={
            "view_id": view_id,
            "reduced_view_id": reduced_view_id,
            "k": k,
            "metric": metric_value,
            "backend": resolved_backend,
            "requested_backend": backend_value,
            "approximate": approximate,
            "seed": seed_value,
            "scope_kind": scope_kind,
            "scope_id": scope_id,
            "memory_preflight": preflight.as_payload(),
        },
        outputs=[
            ArtifactOutput(path="indices.npy", media_type="application/x-npy"),
            ArtifactOutput(path="distances.npy", media_type="application/x-npy"),
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
        ],
        stats={"rows": rows, "k": k},
        warnings=warnings,
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="neighbors fit",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="neighbor_set",
        artifact_id=neighbor_id,
        outputs=[artifact_dir.as_posix()],
        inputs={"view": view_id, "reduced_view": reduced_view_id, "sample": sample_id, "alignment": alignment_id},
        warnings=warnings,
        metrics={
            "rows": rows,
            "k": k,
            "backend": resolved_backend,
            "approximate": approximate,
            "memory_preflight": preflight.as_payload(),
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="neighbors_fit",
        artifact_id=neighbor_id,
    )
    return result
