"""
Alignment services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..alignments.builder import build_alignment_artifact
from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config


def build_alignment(workspace: str | Path, alignment_id: str, *, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    alignment = context.require_alignment(alignment_id)
    alignment_dir = context.output_root / "alignments" / alignment_id
    if alignment_dir.exists() and not force:
        raise ArtifactConflictError(f"alignment artifact already exists: {alignment_dir}")
    if force and alignment_dir.exists():
        import shutil

        shutil.rmtree(alignment_dir)

    (
        artifact_dir,
        matched_rows,
        left_unmatched,
        right_unmatched,
        left_input_path,
        right_input_path,
        key_columns,
    ) = build_alignment_artifact(context, alignment_id=alignment_id)

    manifest = ArtifactManifest(
        artifact_kind="alignment_set",
        artifact_id=alignment_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="alignment build",
        inputs=[
            ArtifactInput(
                kind="alignment_input",
                id=alignment.left,
                digest=sha256_file(left_input_path),
                path=left_input_path.as_posix(),
            ),
            ArtifactInput(
                kind="alignment_input",
                id=alignment.right,
                digest=sha256_file(right_input_path),
                path=right_input_path.as_posix(),
            ),
        ],
        params={
            "left": alignment.left,
            "right": alignment.right,
            "key_basis": alignment.on,
            "key_columns": key_columns,
            "support": alignment.support,
            "left_aggregation": alignment.left_aggregation,
            "right_aggregation": alignment.right_aggregation,
        },
        outputs=[
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
            ArtifactOutput(path="mapping.parquet", media_type="application/x-parquet"),
        ],
        stats={
            "matched_rows": matched_rows,
            "left_unmatched_rows": left_unmatched,
            "right_unmatched_rows": right_unmatched,
        },
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="alignment build",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="alignment_set",
        artifact_id=alignment_id,
        outputs=[artifact_dir.as_posix()],
        inputs={"left": alignment.left, "right": alignment.right},
        metrics={"rows": matched_rows},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="alignment_build",
        artifact_id=alignment_id,
    )
    return result
