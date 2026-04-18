"""
Projection services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.artifact_dirs import commit_staged_artifact_dirs, stage_artifact_dir
from ..io.manifest_io import write_manifest
from ..projections.fit import _fit_projection_artifact
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config
from ._artifact_inputs import dependency_artifact_input
from .memory_service import apply_memory_preflight, evaluate_projection_preflight
from .operation_lock_service import acquire_workspace_operation_lock


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
    with acquire_workspace_operation_lock(
        context.output_root,
        operation="projection_fit",
        owner_id=projection_id,
    ):
        projection_dir = context.output_root / "projections" / projection_id
        if projection_dir.exists() and not force:
            raise ArtifactConflictError(f"projection artifact already exists: {projection_dir}")

        metric_value = metric or context.config.defaults.metric
        preflight = evaluate_projection_preflight(context, view_id=view_id, sample_id=sample_id)
        status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)
        sample_manifest = context.read_manifest(context.output_root / "samples" / sample_id / "manifest.json")
        sample_params = sample_manifest.get("params", {}) if isinstance(sample_manifest.get("params"), dict) else {}
        population_rows = int(
            context.read_manifest(context.output_root / "views" / view_id / "manifest.json")["stats"]["rows"]
        )
        staging_dir = stage_artifact_dir(context.output_root / "projections", projection_id)
        try:
            artifact_dir, rows = _fit_projection_artifact(
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
                dependency_artifact_input(
                    context,
                    kind="view_matrix",
                    artifact_id=view_id,
                    path=context.output_root / "views" / view_id / "matrix.npy",
                ),
                dependency_artifact_input(
                    context,
                    kind="sample_set",
                    artifact_id=sample_id,
                    path=context.output_root / "samples" / sample_id / "rows.parquet",
                ),
            ],
            params={
                "method": "umap",
                "metric": metric_value,
                "random_seed": seed,
                "dimensionality": 2,
                "sampling_strategy": sample_params.get("strategy", "unknown"),
                "projection_role": ("primary" if rows == population_rows else "appendix"),
                "default_rank": 0 if rows == population_rows else 100,
                "memory_preflight": preflight.as_payload(),
            },
            outputs=[ArtifactOutput(path="coords.parquet", media_type="application/x-parquet")],
            stats={
                "rows": rows,
                "dims": 2,
                "projected_rows": rows,
                "population_rows": population_rows,
                "is_full_population": rows == population_rows,
            },
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
