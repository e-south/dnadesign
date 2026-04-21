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
from ..io.json_io import read_json, write_json
from ..io.manifest_io import write_manifest
from ..notebooks.scaffold import render_workspace_notebook
from ..runs.recorder import record_audit
from ..sources.provenance import source_provenance_digest
from ..version import __version__
from ..workspaces.loader import load_workspace_config
from ._artifact_inputs import artifact_input_from_manifest
from ._artifacts import artifact_dir, artifact_manifest_path
from .notebook_controls_service import build_workspace_notebook_controls_payload


def _workspace_notebook_dir(context, notebook_id: str) -> Path:
    return artifact_dir(context, artifact_kind="notebook", artifact_id=notebook_id)


def _workspace_notebook_path(context, notebook_id: str) -> Path:
    return _workspace_notebook_dir(context, notebook_id) / "notebook.py"


def _notebook_generation_artifact_exists(context, notebook_id: str) -> bool:
    notebook_dir = _workspace_notebook_dir(context, notebook_id)
    return any((notebook_dir / name).exists() for name in ("notebook.py", "controls.json", "manifest.json"))


def _notebook_health_path(context, notebook_id: str) -> Path:
    return _workspace_notebook_dir(context, notebook_id) / "health.json"


def _write_notebook_health(
    context,
    *,
    notebook_id: str,
    status: str,
    checks: dict[str, bool],
    warnings: list[str],
    workspace_id: str | None = None,
) -> dict[str, object]:
    payload = {
        "workspace_id": workspace_id or context.workspace_id,
        "notebook_id": notebook_id,
        "status": status,
        "checks": checks,
        "warnings": warnings,
    }
    health_path = _notebook_health_path(context, notebook_id)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(health_path, payload)
    return payload


def _notebook_plot_ids(context, notebook: WorkspaceNotebookConfig) -> list[str]:
    return list(
        notebook.ordered_plots or context.require_deliverable(notebook.default_deliverable).outputs.get("plots", [])
    )


def _notebook_plot_inputs(
    context, notebook: WorkspaceNotebookConfig
) -> tuple[list[ArtifactInput], list[str], list[str]]:
    plot_ids = _notebook_plot_ids(context, notebook)
    inputs: list[ArtifactInput] = []
    missing_plot_ids: list[str] = []
    for plot_id in plot_ids:
        manifest_path = artifact_manifest_path(context, artifact_kind="plot", artifact_id=plot_id)
        if not manifest_path.exists():
            missing_plot_ids.append(plot_id)
            continue
        inputs.append(artifact_input_from_manifest("plot", plot_id, digest_path=manifest_path))
    return inputs, plot_ids, missing_plot_ids


def _notebook_plot_output_paths(context, notebook: WorkspaceNotebookConfig) -> list[Path]:
    output_paths: list[Path] = []
    for plot_id in _notebook_plot_ids(context, notebook):
        manifest_path = artifact_manifest_path(context, artifact_kind="plot", artifact_id=plot_id)
        if not manifest_path.is_file():
            continue
        manifest_payload = read_json(manifest_path)
        plot_dir = manifest_path.parent
        for output in manifest_payload.get("outputs", []):
            if not isinstance(output, dict):
                continue
            path_text = output.get("path")
            if isinstance(path_text, str) and path_text.strip():
                output_paths.append(plot_dir / path_text)
    return output_paths


def _default_deliverable_status(context, default_deliverable: str):
    from .deliverable_service import deliverable_status_from_context

    return deliverable_status_from_context(context, default_deliverable)


def _default_deliverable_readiness(
    context,
    *,
    notebook_id: str,
    default_deliverable: str,
    missing_plot_ids: list[str],
) -> tuple[str, list[str]]:
    status = _default_deliverable_status(context, default_deliverable)
    relevant_outputs = [entry for entry in status.outputs if entry.name != f"notebook:{notebook_id}"]
    reasons: list[str] = []
    if missing_plot_ids:
        reasons.append("missing ordered plots: " + ", ".join(missing_plot_ids))
    output_reasons = [
        str(entry.reason or entry.status) for entry in [*status.checks, *relevant_outputs] if entry.status != "ok"
    ]
    if output_reasons:
        reasons.append("; ".join(output_reasons))
    if status.warnings:
        reasons.append("; ".join(str(message) for message in status.warnings))
    if not reasons:
        return "ok", []
    resolved_status = status.status if status.status in {"attention", "missing", "error"} else "attention"
    return resolved_status, reasons


def _load_catalog_payload(context) -> dict[str, object]:
    from .catalog_service import workspace_catalog_from_context

    return workspace_catalog_from_context(context)


def _notebook_smoke_status(payload: dict[str, object]) -> str:
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return "error"
    blocking_checks = (
        "notebook_exists",
        "control_plane_loads",
        "imports_resolve",
        "plot_catalog_loads",
    )
    if any(not bool(checks.get(name)) for name in blocking_checks):
        return "error"
    degraded_checks = ("default_deliverable_ready", "static_links_resolve")
    if any(not bool(checks.get(name)) for name in degraded_checks):
        return "attention"
    return "ok"


def _merge_status(*statuses: str) -> str:
    if any(status == "error" for status in statuses):
        return "error"
    if any(status == "attention" for status in statuses):
        return "attention"
    return "ok"


def _extend_unique_warnings(target: list[str], additions: list[str]) -> None:
    seen = set(target)
    for warning in additions:
        if warning in seen:
            continue
        target.append(warning)
        seen.add(warning)


def generate_notebook(workspace: str | Path, notebook_id: str, *, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    notebook = context.require_notebook(notebook_id)
    assert isinstance(notebook, WorkspaceNotebookConfig)

    notebook_dir = _workspace_notebook_dir(context, notebook_id)
    notebook_path = _workspace_notebook_path(context, notebook_id)
    controls_path = notebook_dir / "controls.json"
    if _notebook_generation_artifact_exists(context, notebook_id) and not force:
        raise ArtifactConflictError(f"notebook artifact already exists: {notebook_dir}")
    if force and notebook_dir.exists():
        import shutil

        shutil.rmtree(notebook_dir)

    inputs, plot_ids, missing_plot_ids = _notebook_plot_inputs(context, notebook)
    catalog_payload = _load_catalog_payload(context)
    controls_payload = build_workspace_notebook_controls_payload(
        context,
        notebook_id=notebook_id,
        catalog_payload=catalog_payload,
    )
    default_deliverable_status, default_deliverable_reasons = _default_deliverable_readiness(
        context,
        notebook_id=notebook_id,
        default_deliverable=notebook.default_deliverable,
        missing_plot_ids=missing_plot_ids,
    )
    base_status = "ok" if default_deliverable_status == "ok" and not missing_plot_ids else "attention"
    warnings: list[str] = []
    if missing_plot_ids:
        warnings.append(
            "notebook plot-review inventory is not fully materialized; "
            "notebook generated with an explicit degraded plots state: " + ", ".join(missing_plot_ids)
        )
    if default_deliverable_status != "ok":
        warnings.append(
            "default deliverable requires attention before the notebook is end-to-end ready: "
            + "; ".join(default_deliverable_reasons or [default_deliverable_status])
        )

    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(
        render_workspace_notebook(
            workspace_id=context.workspace_id,
            notebook_id=notebook_id,
            title=notebook.title,
            description=notebook.description,
            default_deliverable=notebook.default_deliverable,
            default_surface=notebook.default_surface,
        ),
        encoding="utf-8",
    )
    controls_payload_json = controls_payload.model_dump(mode="json")
    write_json(controls_path, controls_payload_json)
    status = base_status
    try:
        smoke_payload = smoke_workspace_notebook(workspace, notebook_id=notebook_id)
        _write_notebook_health(
            context,
            notebook_id=notebook_id,
            status=str(smoke_payload.get("status") or "error"),
            checks={key: bool(value) for key, value in dict(smoke_payload.get("checks") or {}).items()},
            warnings=[str(item).strip() for item in smoke_payload.get("warnings", []) if str(item).strip()],
            workspace_id=str(smoke_payload.get("workspace_id") or context.workspace_id),
        )
    except Exception as exc:
        status = "error"
        error_warning = f"notebook health refresh failed: {exc}"
        warnings.append(error_warning)
        _write_notebook_health(
            context,
            notebook_id=notebook_id,
            status="error",
            checks={
                "notebook_exists": notebook_path.is_file(),
                "control_plane_loads": controls_path.is_file(),
                "imports_resolve": False,
                "plot_catalog_loads": False,
                "default_deliverable_ready": default_deliverable_status == "ok",
                "static_links_resolve": False,
            },
            warnings=[error_warning],
        )
    else:
        status = _merge_status(base_status, _notebook_smoke_status(smoke_payload))
        _extend_unique_warnings(
            warnings,
            [str(item).strip() for item in smoke_payload.get("warnings", []) if str(item).strip()],
        )
    source_provenance = [
        {
            "id": "workspace_config",
            "role": "workspace_config",
            "path": context.config_path.as_posix(),
            "digest": source_provenance_digest({"path": context.config_path.as_posix()}),
        }
    ]
    manifest = ArtifactManifest(
        artifact_kind="notebook",
        artifact_id=notebook_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="notebook generate",
        status=status,
        inputs=inputs,
        source_provenance=source_provenance,
        params={
            "kind": notebook.kind,
            "runtime": "marimo",
            "title": notebook.title,
            "default_deliverable": notebook.default_deliverable,
            "default_surface": notebook.default_surface,
            "ordered_plot_ids": plot_ids,
            "missing_ordered_plots": missing_plot_ids,
        },
        outputs=[
            ArtifactOutput(path="notebook.py", media_type="text/x-python"),
            ArtifactOutput(path="controls.json", media_type="application/json"),
        ],
        stats={
            "plots": len(plot_ids),
            "deliverables": len(catalog_payload.get("deliverables", [])),
            "runs": len(catalog_payload.get("runs", [])),
            "geometries": len(controls_payload.geometry_controls.geometries),
            "default_deliverable_status": default_deliverable_status,
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
            "geometries": len(controls_payload.geometry_controls.geometries),
            "default_deliverable_status": default_deliverable_status,
            "missing_ordered_plots": missing_plot_ids,
        },
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="notebook_generate",
        artifact_id=notebook_id,
    )
    return result


def smoke_workspace_notebook(workspace: str | Path, *, notebook_id: str | None = None) -> dict[str, object]:
    context = load_workspace_config(workspace)
    if not context.config.notebooks:
        raise WorkspaceValidationError("workspace does not declare a workspace notebook")

    resolved_notebook_id = notebook_id or next(iter(context.config.notebooks))
    notebook = context.require_notebook(resolved_notebook_id)
    notebook_path = _workspace_notebook_path(context, resolved_notebook_id)
    catalog = _load_catalog_payload(context)
    controls_path = _workspace_notebook_dir(context, resolved_notebook_id) / "controls.json"

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
            checks["control_plane_loads"] = bool(controls.geometry_controls.geometries)
        except Exception as exc:  # pragma: no cover - surfaced in health payload
            warnings.append(f"control_plane_loads failed: {exc}")

    deliverables = catalog.get("deliverables", [])
    plots = catalog.get("plots", [])
    checks["plot_catalog_loads"] = isinstance(deliverables, list) and isinstance(plots, list)
    _, _, missing_plot_ids = _notebook_plot_inputs(context, notebook)
    default_deliverable_status, default_deliverable_reasons = _default_deliverable_readiness(
        context,
        notebook_id=resolved_notebook_id,
        default_deliverable=notebook.default_deliverable,
        missing_plot_ids=missing_plot_ids,
    )
    checks["default_deliverable_ready"] = default_deliverable_status == "ok"
    if not checks["default_deliverable_ready"]:
        warnings.append("default_deliverable_ready failed: " + " | ".join(default_deliverable_reasons))
    output_paths = _notebook_plot_output_paths(context, notebook)
    checks["static_links_resolve"] = bool(output_paths) and all(path.is_file() for path in output_paths)

    status = "ok" if all(checks.values()) else "error"
    return _write_notebook_health(
        context,
        notebook_id=resolved_notebook_id,
        status=status,
        checks=checks,
        warnings=warnings,
    )
