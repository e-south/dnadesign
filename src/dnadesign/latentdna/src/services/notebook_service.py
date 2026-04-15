"""
Workspace notebook scaffold services for latentdna.
"""

from __future__ import annotations

import runpy
from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError, WorkspaceValidationError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.notebook import WorkspaceNotebookConfig, WorkspaceNotebookControls
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.json_io import read_json, write_json
from ..io.manifest_io import write_manifest
from ..notebooks.scaffold import render_workspace_notebook
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import load_workspace_config
from ._artifacts import artifact_dir, artifact_exists, artifact_manifest_path
from .notebook_controls_service import build_workspace_notebook_controls_payload


def _workspace_notebook_dir(context, notebook_id: str) -> Path:
    return artifact_dir(context, artifact_kind="notebook", artifact_id=notebook_id)


def _workspace_notebook_path(context, notebook_id: str) -> Path:
    return _workspace_notebook_dir(context, notebook_id) / "notebook.py"


def _default_deliverable_plot_inputs(
    context, default_deliverable: str
) -> tuple[list[ArtifactInput], list[str], list[str]]:
    deliverable = context.require_deliverable(default_deliverable)
    plot_ids = list(deliverable.outputs.get("plots", []))
    inputs: list[ArtifactInput] = []
    missing_plot_ids: list[str] = []
    for plot_id in plot_ids:
        manifest_path = artifact_manifest_path(context, artifact_kind="plot", artifact_id=plot_id)
        if not manifest_path.exists():
            missing_plot_ids.append(plot_id)
            continue
        inputs.append(ArtifactInput(kind="plot", id=plot_id, digest=sha256_file(manifest_path)))
    return inputs, plot_ids, missing_plot_ids


def _load_catalog_payload(context) -> dict[str, object]:
    catalog_path = context.output_root / "catalog.json"
    if catalog_path.is_file():
        return read_json(catalog_path)
    from .catalog_service import workspace_catalog

    return workspace_catalog(context.workspace_dir)


def generate_notebook(workspace: str | Path, notebook_id: str, *, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    notebook = context.require_notebook(notebook_id)
    assert isinstance(notebook, WorkspaceNotebookConfig)

    notebook_dir = _workspace_notebook_dir(context, notebook_id)
    notebook_path = _workspace_notebook_path(context, notebook_id)
    controls_path = notebook_dir / "controls.json"
    if notebook_dir.exists() and not force:
        raise ArtifactConflictError(f"notebook artifact already exists: {notebook_dir}")
    if force and notebook_dir.exists():
        import shutil

        shutil.rmtree(notebook_dir)

    inputs, plot_ids, missing_plot_ids = _default_deliverable_plot_inputs(context, notebook.default_deliverable)
    catalog_payload = _load_catalog_payload(context)
    controls_payload = build_workspace_notebook_controls_payload(context, notebook_id=notebook_id)
    status = "attention" if missing_plot_ids else "ok"
    warnings = (
        [
            "default deliverable is not fully materialized; "
            "notebook generated with an explicit degraded main-plot state: " + ", ".join(missing_plot_ids)
        ]
        if missing_plot_ids
        else []
    )

    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        render_workspace_notebook(
            workspace_id=context.workspace_id,
            notebook_id=notebook_id,
            title=notebook.title,
            description=notebook.description,
            default_deliverable=notebook.default_deliverable,
        ),
        encoding="utf-8",
    )
    controls_payload_json = controls_payload.model_dump(mode="json")
    write_json(controls_path, controls_payload_json)
    manifest = ArtifactManifest(
        artifact_kind="notebook",
        artifact_id=notebook_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="notebook generate",
        status=status,
        inputs=inputs,
        params={
            "kind": notebook.kind,
            "runtime": "marimo",
            "title": notebook.title,
            "default_deliverable": notebook.default_deliverable,
            "missing_default_plots": missing_plot_ids,
        },
        outputs=[
            ArtifactOutput(path="notebook.py", media_type="text/x-python"),
            ArtifactOutput(path="controls.json", media_type="application/json"),
        ],
        stats={
            "plots": len(plot_ids),
            "deliverables": len(catalog_payload.get("deliverables", [])),
            "runs": len(catalog_payload.get("runs", [])),
            "geometries": len(controls_payload.geometry_switchboard.geometries),
        },
        warnings=warnings,
    )
    write_manifest(notebook_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="notebook generate",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="notebook",
        artifact_id=notebook_id,
        outputs=[notebook_dir.as_posix()],
        inputs={"notebook": notebook_id, "default_deliverable": notebook.default_deliverable},
        warnings=warnings,
        metrics={
            "plots": len(plot_ids),
            "deliverables": len(catalog_payload.get("deliverables", [])),
            "geometries": len(controls_payload.geometry_switchboard.geometries),
            "missing_default_plots": missing_plot_ids,
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="notebook_generate",
        artifact_id=notebook_id,
    )
    return result


def smoke_workspace_notebook(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    if not context.config.notebooks:
        raise WorkspaceValidationError("workspace does not declare a workspace notebook")

    notebook_id, notebook = next(iter(context.config.notebooks.items()))
    notebook_path = _workspace_notebook_path(context, notebook_id)
    catalog = _load_catalog_payload(context)
    health_path = context.output_root / "notebooks" / "health.json"
    controls_path = _workspace_notebook_dir(context, notebook_id) / "controls.json"

    checks = {
        "notebook_exists": notebook_path.is_file(),
        "control_plane_loads": False,
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
    if controls_path.is_file():
        try:
            controls = WorkspaceNotebookControls.model_validate(read_json(controls_path))
            checks["control_plane_loads"] = bool(controls.geometry_switchboard.geometries)
        except Exception as exc:  # pragma: no cover - surfaced in health payload
            warnings.append(f"control_plane_loads failed: {exc}")

    deliverables = catalog.get("deliverables", [])
    plots = catalog.get("plots", [])
    checks["plot_catalog_loads"] = isinstance(deliverables, list) and isinstance(plots, list)

    default_deliverable_plots = [
        row
        for row in plots
        if isinstance(row, dict) and row.get("deliverable_id") == notebook.default_deliverable and row.get("plot_id")
    ]
    checks["default_deliverable_ready"] = bool(default_deliverable_plots) and all(
        artifact_exists(
            context,
            artifact_kind="plot",
            artifact_id=str(row["plot_id"]),
        )
        for row in default_deliverable_plots
    )
    output_paths = [
        context.output_root / str(path_text)
        for row in default_deliverable_plots
        if isinstance(row, dict)
        for path_text in row.get("output_paths", [])
    ]
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
