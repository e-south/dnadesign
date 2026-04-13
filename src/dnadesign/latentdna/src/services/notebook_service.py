"""
Notebook scaffold services for latentdna.
"""

from __future__ import annotations

import runpy
from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError, MissingArtifactError, WorkspaceValidationError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.notebook import ArtifactReviewNotebookConfig, WorkspaceBrowserNotebookConfig
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.json_io import read_json, write_json
from ..io.manifest_io import write_manifest
from ..notebooks.scaffold import render_artifact_review_notebook, render_workspace_browser_notebook
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config
from ._artifacts import artifact_dir, artifact_exists, artifact_manifest_path


def _browser_script_path(context, notebook_id: str) -> Path:
    return context.output_root / "notebooks" / f"{notebook_id}.py"


def _default_deliverable_plot_inputs(context, default_deliverable: str) -> tuple[list[ArtifactInput], list[str]]:
    deliverable = context.require_deliverable(default_deliverable)
    plot_ids = list(deliverable.outputs.get("plots", []))
    inputs: list[ArtifactInput] = []
    for plot_id in plot_ids:
        manifest_path = artifact_manifest_path(context, artifact_kind="plot", artifact_id=plot_id)
        if not manifest_path.exists():
            raise MissingArtifactError(
                f"default deliverable plot is missing for notebook generation: {default_deliverable}:{plot_id}"
            )
        inputs.append(ArtifactInput(kind="plot", id=plot_id, digest=sha256_file(manifest_path)))
    return inputs, plot_ids


def generate_notebook(workspace: str | Path, notebook_id: str, *, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    notebook = context.require_notebook(notebook_id)
    notebook_dir = context.output_root / "notebooks" / notebook_id
    browser_script_path = _browser_script_path(context, notebook_id)
    if (notebook_dir.exists() or browser_script_path.exists()) and not force:
        raise ArtifactConflictError(f"notebook artifact already exists: {notebook_dir}")
    if force and notebook_dir.exists():
        import shutil

        shutil.rmtree(notebook_dir)
    if force and browser_script_path.exists():
        browser_script_path.unlink()

    if isinstance(notebook, WorkspaceBrowserNotebookConfig):
        inputs, plot_ids = _default_deliverable_plot_inputs(context, notebook.default_deliverable)
        notebook_dir.mkdir(parents=True, exist_ok=True)
        browser_script_path.parent.mkdir(parents=True, exist_ok=True)
        browser_script_path.write_text(
            render_workspace_browser_notebook(
                workspace_id=context.workspace_id,
                notebook_id=notebook_id,
                title=notebook.title,
                description=notebook.description,
                default_deliverable=notebook.default_deliverable,
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
                "default_deliverable": notebook.default_deliverable,
            },
            outputs=[ArtifactOutput(path="../" + browser_script_path.name, media_type="text/x-python")],
            stats={"plots": len(plot_ids)},
        )
        write_manifest(notebook_dir / "manifest.json", manifest.model_dump(mode="json"))
        result = CommandResult(
            command="notebook generate",
            workspace_id=context.workspace_id,
            status="ok",
            artifact_kind="notebook",
            artifact_id=notebook_id,
            outputs=[browser_script_path.as_posix(), notebook_dir.as_posix()],
            inputs={"notebook": notebook_id, "default_deliverable": notebook.default_deliverable},
            metrics={"plots": len(plot_ids)},
        )
        record_audit(
            context.output_root / "logs" / "audit",
            payload=result.model_dump(mode="json"),
            command="notebook_generate",
            artifact_id=notebook_id,
        )
        return result

    assert isinstance(notebook, ArtifactReviewNotebookConfig)
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


def smoke_workspace_browser(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    browser_notebooks = [
        (notebook_id, notebook)
        for notebook_id, notebook in context.config.notebooks.items()
        if isinstance(notebook, WorkspaceBrowserNotebookConfig)
    ]
    if not browser_notebooks:
        raise WorkspaceValidationError("workspace does not declare a workspace_browser notebook")

    notebook_id, notebook = browser_notebooks[0]
    notebook_path = _browser_script_path(context, notebook_id)
    plots_index_path = context.output_root / "plots" / "index.json"
    health_path = context.output_root / "notebooks" / "health.json"

    checks = {
        "notebook_exists": notebook_path.is_file(),
        "imports_resolve": False,
        "plot_catalog_loads": False,
        "default_deliverable_ready": False,
        "static_links_resolve": False,
    }
    warnings: list[str] = []

    if checks["notebook_exists"]:
        try:
            runpy.run_path(notebook_path.as_posix(), init_globals={"__name__": "__latentdna_smoke__"})
            checks["imports_resolve"] = True
        except Exception as exc:  # pragma: no cover - surfaced in health payload
            warnings.append(f"imports_resolve failed: {exc}")

    plot_index: dict[str, object] = {}
    if plots_index_path.is_file():
        try:
            plot_index = read_json(plots_index_path)
            checks["plot_catalog_loads"] = isinstance(plot_index, dict) and isinstance(plot_index.get("plots"), list)
        except Exception as exc:  # pragma: no cover - surfaced in health payload
            warnings.append(f"plot_catalog_loads failed: {exc}")

    deliverable = context.require_deliverable(notebook.default_deliverable)
    default_plot_ids = list(deliverable.outputs.get("plots", []))
    checks["default_deliverable_ready"] = bool(default_plot_ids) and all(
        artifact_exists(context, artifact_kind="plot", artifact_id=plot_id) for plot_id in default_plot_ids
    )

    if checks["plot_catalog_loads"]:
        plot_rows = plot_index.get("plots", [])
        output_paths: list[Path] = []
        for row in plot_rows if isinstance(plot_rows, list) else []:
            if not isinstance(row, dict):
                continue
            for path_text in row.get("output_paths", []):
                output_paths.append(context.output_root / str(path_text))
        checks["static_links_resolve"] = bool(output_paths) and all(path.is_file() for path in output_paths)

    status = "ok" if all(checks.values()) else "error"
    payload = {
        "workspace_id": context.workspace_id,
        "notebook_id": notebook_id,
        "status": status,
        "checks": checks,
        "warnings": warnings,
    }
    write_json(health_path, payload)
    return payload
