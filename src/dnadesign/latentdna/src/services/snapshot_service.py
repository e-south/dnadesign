"""
Snapshot services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..snapshots.build import build_snapshot_artifact
from ..sources.resolver import resolve_source, source_digest
from ..version import __version__
from ..workspaces.loader import load_workspace_config


def build_snapshot(workspace: str | Path, snapshot_id: str, *, source_id: str, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    snapshot_dir = context.output_root / "snapshots" / snapshot_id
    if snapshot_dir.exists() and not force:
        raise ArtifactConflictError(f"snapshot artifact already exists: {snapshot_dir}")
    if force and snapshot_dir.exists():
        import shutil

        shutil.rmtree(snapshot_dir)

    artifact_dir, source_path, rows, row_columns, metadata_columns = build_snapshot_artifact(
        context,
        snapshot_id=snapshot_id,
        source_id=source_id,
    )
    source = context.require_source(source_id)
    resolved_source = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
    source_input_digest, source_provenance, input_digests = source_digest(
        resolved_source,
        columns=[*row_columns, *metadata_columns],
    )
    manifest = ArtifactManifest(
        artifact_kind="snapshot",
        artifact_id=snapshot_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="snapshot build",
        inputs=[ArtifactInput(kind="source", id=source_id, digest=source_input_digest)],
        input_digests=input_digests,
        freshness_basis={"kind": "source_provenance", "known": True},
        source_provenance=source_provenance,
        params={
            "source": source_id,
            "row_columns": row_columns,
            "metadata_columns": metadata_columns,
        },
        outputs=[
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="metadata.parquet", media_type="application/x-parquet"),
        ],
        stats={"rows": rows},
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="snapshot build",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="snapshot",
        artifact_id=snapshot_id,
        outputs=[artifact_dir.as_posix()],
        inputs={"snapshot": snapshot_id, "source": source_id},
        input_digests=input_digests,
        metrics={"rows": rows},
        freshness_known=True,
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="snapshot_build",
        artifact_id=snapshot_id,
    )
    return result
