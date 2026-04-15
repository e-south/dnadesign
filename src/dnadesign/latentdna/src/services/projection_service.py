"""
Projection services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.artifact_dirs import commit_staged_artifact_dirs, stage_artifact_dir
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..projections.fit import fit_projection_artifact
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config
from .memory_service import apply_memory_preflight, evaluate_projection_preflight


def fit_projection(
    workspace: str | Path,
    view_id: str,
    *,
    projection_id: str,
    sample_id: str,
    metric: str | None,
    seed: int,
    allow_memory_overage: bool = False,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    projection_dir = context.output_root / "projections" / projection_id
    if projection_dir.exists() and not force:
        raise ArtifactConflictError(f"projection artifact already exists: {projection_dir}")
    staging_dir = stage_artifact_dir(context.output_root / "projections", projection_id)

    metric_value = metric or context.config.defaults.metric
    preflight = evaluate_projection_preflight(context, view_id=view_id, sample_id=sample_id)
    status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)
    try:
        artifact_dir, rows = fit_projection_artifact(
            context,
            view_id=view_id,
            projection_id=projection_id,
            sample_id=sample_id,
            metric=metric_value,
            seed=seed,
            artifact_dir=staging_dir,
        )
        assert artifact_dir == staging_dir
    except Exception:
        import shutil

        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    manifest = ArtifactManifest(
        artifact_kind="projection",
        artifact_id=projection_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="projection fit",
        status=status,
        inputs=[
            ArtifactInput(
                kind="view_matrix",
                id=view_id,
                digest=sha256_file(context.output_root / "views" / view_id / "matrix.npy"),
            ),
            ArtifactInput(
                kind="sample_set",
                id=sample_id,
                digest=sha256_file(context.output_root / "samples" / sample_id / "rows.parquet"),
            ),
        ],
        params={
            "method": "umap",
            "metric": metric_value,
            "random_seed": seed,
            "dimensionality": 2,
            "memory_preflight": preflight.as_payload(),
        },
        outputs=[ArtifactOutput(path="coords.parquet", media_type="application/x-parquet")],
        stats={"rows": rows, "dims": 2},
        warnings=warnings,
    )
    write_manifest(staging_dir / "manifest.json", manifest.model_dump(mode="json"))
    commit_staged_artifact_dirs([(staging_dir, projection_dir)], force=force)
    result = CommandResult(
        command="projection fit",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="projection",
        artifact_id=projection_id,
        outputs=[projection_dir.as_posix()],
        inputs={"view": view_id, "sample": sample_id},
        warnings=warnings,
        metrics={"rows": rows, "memory_preflight": preflight.as_payload()},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="projection_fit",
        artifact_id=projection_id,
    )
    return result
