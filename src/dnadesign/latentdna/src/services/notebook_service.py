"""
Notebook scaffold services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError, MissingArtifactError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..notebooks.scaffold import render_artifact_review_notebook
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config
from ._artifacts import artifact_dir, artifact_exists, artifact_manifest_path


def generate_notebook(workspace: str | Path, notebook_id: str, *, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    notebook = context.require_notebook(notebook_id)
    notebook_dir = context.output_root / "notebooks" / notebook_id
    if notebook_dir.exists() and not force:
        raise ArtifactConflictError(f"notebook artifact already exists: {notebook_dir}")
    if force and notebook_dir.exists():
        import shutil

        shutil.rmtree(notebook_dir)

    resolved_artifacts: list[dict[str, str]] = []
    inputs: list[ArtifactInput] = []
    for artifact in notebook.artifacts:
        if not artifact_exists(context, artifact_kind=artifact.kind, artifact_id=artifact.id):
            raise MissingArtifactError(f"artifact is missing for notebook generation: {artifact.kind}:{artifact.id}")
        resolved_dir = artifact_dir(context, artifact_kind=artifact.kind, artifact_id=artifact.id)
        manifest_path = artifact_manifest_path(context, artifact_kind=artifact.kind, artifact_id=artifact.id)
        resolved_artifacts.append(
            {
                "alias": artifact.alias or artifact.id,
                "kind": artifact.kind,
                "id": artifact.id,
                "path": resolved_dir.relative_to(context.workspace_dir).as_posix(),
            }
        )
        inputs.append(
            ArtifactInput(
                kind=artifact.kind,
                id=artifact.id,
                digest=sha256_file(manifest_path),
            )
        )

    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / "notebook.py"
    notebook_path.write_text(
        render_artifact_review_notebook(
            workspace_id=context.workspace_id,
            notebook_id=notebook_id,
            title=notebook.title,
            description=notebook.description,
            artifacts=resolved_artifacts,
        ),
        encoding="utf-8",
    )
    manifest = ArtifactManifest(
        artifact_kind="notebook",
        artifact_id=notebook_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="notebook generate",
        inputs=inputs,
        params={
            "kind": notebook.kind,
            "runtime": "marimo",
            "title": notebook.title,
            "artifacts": resolved_artifacts,
        },
        outputs=[ArtifactOutput(path="notebook.py", media_type="text/x-python")],
        stats={"artifacts": len(resolved_artifacts)},
    )
    write_manifest(notebook_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="notebook generate",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="notebook",
        artifact_id=notebook_id,
        outputs=[notebook_dir.as_posix()],
        inputs={"notebook": notebook_id},
        metrics={"artifacts": len(resolved_artifacts)},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="notebook_generate",
        artifact_id=notebook_id,
    )
    return result
