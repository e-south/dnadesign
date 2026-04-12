"""
Distance services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..distances.score import score_distance_artifact
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config


def score_distance(
    workspace: str | Path,
    distance_id: str,
    *,
    view_id: str,
    landmark_ids: list[str],
    metric: str | None,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    distance_dir = context.output_root / "distances" / distance_id
    if distance_dir.exists() and not force:
        raise ArtifactConflictError(f"distance artifact already exists: {distance_dir}")
    if force and distance_dir.exists():
        import shutil

        shutil.rmtree(distance_dir)

    metric_value = metric or context.config.defaults.metric
    artifact_dir, rows, columns, representation_modes, member_counts = score_distance_artifact(
        context,
        distance_id=distance_id,
        view_id=view_id,
        landmark_ids=landmark_ids,
        metric=metric_value,
    )
    manifest = ArtifactManifest(
        artifact_kind="distance_set",
        artifact_id=distance_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="distance score",
        inputs=[
            ArtifactInput(
                kind="view_matrix",
                id=view_id,
                digest=sha256_file(context.output_root / "views" / view_id / "matrix.npy"),
            )
        ],
        params={
            "view_id": view_id,
            "landmark_ids": landmark_ids,
            "metric": metric_value,
            "representation_modes": representation_modes,
            "member_counts": member_counts,
        },
        outputs=[ArtifactOutput(path="table.parquet", media_type="application/x-parquet")],
        stats={"rows": rows, "columns": len(columns)},
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="distance score",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="distance_set",
        artifact_id=distance_id,
        outputs=[artifact_dir.as_posix()],
        inputs={"view": view_id, "landmarks": landmark_ids},
        metrics={"rows": rows, "columns": len(columns)},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="distance_score",
        artifact_id=distance_id,
    )
    return result
