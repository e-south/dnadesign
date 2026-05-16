"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/services/notebook_service.py

Workspace notebook scaffold services for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ..contracts.deliverable import DeliverableStatusResult
from ..contracts.errors import ArtifactConflictError, WorkspaceValidationError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.notebook import WorkspaceNotebookConfig, WorkspaceNotebookControls
from ..contracts.result import CommandResult
from ..io.json_io import read_json, write_json
from ..io.manifest_io import write_manifest
from ..notebooks.browser_runtime import _parse_deliverable_markdown
from ..notebooks.browser_runtime_plot_review import load_plot_review_frames
from ..notebooks.scaffold import render_workspace_notebook
from ..plots.recipes import resolve_plot_spec
from ..runs.recorder import record_audit
from ..sources.provenance import source_provenance_digest
from ..studies.docs_refs import read_docs_ref
from ..version import __version__
from ..workspaces.loader import load_workspace_config
from ._artifact_inputs import artifact_input_from_manifest
from ._artifacts import artifact_dir, artifact_manifest_path
from .freshness_service import FreshnessCache, evaluate_artifact_freshness
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
    deliverable_status: DeliverableStatusResult | None = None,
) -> tuple[str, list[str]]:
    status = deliverable_status or _default_deliverable_status(context, default_deliverable)
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


def _is_freshness_drift_reason(reason: str | None) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "freshness",
            "stale ",
            "artifact manifest is marked error",
        )
    )


def _blocking_default_deliverable_reasons(
    *,
    notebook_id: str,
    deliverable_status: DeliverableStatusResult,
) -> list[str]:
    blockers: list[str] = []
    for entry in [*deliverable_status.checks, *deliverable_status.outputs]:
        if entry.name == f"notebook:{notebook_id}":
            continue
        if entry.status == "error":
            blockers.append(str(entry.reason or f"{entry.name} status=error"))
            continue
        if entry.status == "attention" and _is_freshness_drift_reason(entry.reason):
            blockers.append(str(entry.reason or f"{entry.name} freshness requires attention"))
    for warning in deliverable_status.warnings:
        if _is_freshness_drift_reason(warning):
            blockers.append(str(warning))
    deduped: list[str] = []
    seen: set[str] = set()
    for blocker in blockers:
        if blocker in seen:
            continue
        deduped.append(blocker)
        seen.add(blocker)
    return deduped


def _blocking_ordered_plot_freshness_reasons(
    context,
    *,
    notebook: WorkspaceNotebookConfig,
) -> list[str]:
    cache = FreshnessCache()
    blockers: list[str] = []
    for plot_id in _notebook_plot_ids(context, notebook):
        freshness = evaluate_artifact_freshness(
            context,
            artifact_kind="plot",
            artifact_id=plot_id,
            cache=cache,
        )
        status = str(freshness.get("status") or "")
        reason = str(freshness.get("reason") or "")
        if status == "error":
            blockers.append(f"ordered plot `{plot_id}` status=error: {reason or 'plot artifact error'}")
            continue
        if status == "attention" and _is_freshness_drift_reason(reason):
            blockers.append(f"ordered plot `{plot_id}` freshness requires attention: {reason}")
    return blockers


def _ordered_plot_owner_deliverables(context, *, notebook: WorkspaceNotebookConfig) -> list[str]:
    plot_ids = set(_notebook_plot_ids(context, notebook))
    owners: list[str] = []
    for deliverable_id, deliverable in context.config.deliverables.items():
        if plot_ids.intersection(deliverable.outputs.get("plots", [])):
            owners.append(deliverable_id)
    return owners


def _ordered_plot_study_doc_subsection_warnings(context, *, notebook: WorkspaceNotebookConfig) -> list[str]:
    plot_ids = _notebook_plot_ids(context, notebook)
    warnings: list[str] = []
    for plot_id in plot_ids:
        owner_deliverables = [
            deliverable
            for deliverable in context.config.deliverables.values()
            if plot_id in deliverable.outputs.get("plots", [])
        ]
        docs_refs = [
            docs_ref
            for deliverable in owner_deliverables
            for docs_ref in getattr(deliverable, "docs_refs", [])
            if str(docs_ref).strip()
        ]
        if not docs_refs:
            continue
        covered = False
        read_errors: list[str] = []
        for docs_ref in docs_refs:
            try:
                docs_payload = read_docs_ref(context, str(docs_ref))
            except Exception as exc:
                read_errors.append(f"{docs_ref}: {exc}")
                continue
            parsed = _parse_deliverable_markdown(str(docs_payload.get("content") or ""))
            plot_sections = parsed.get("plot_sections", {})
            if isinstance(plot_sections, dict) and plot_id in plot_sections:
                covered = True
                break
        if covered:
            continue
        if read_errors:
            warnings.append(f"study-doc subsection check failed for `{plot_id}`: " + "; ".join(read_errors))
        else:
            warnings.append(f"missing plot-specific study-doc subsection for `{plot_id}`")
    return warnings


def _load_catalog_payload(context) -> dict[str, object]:
    from .catalog_service import workspace_catalog_from_context

    return workspace_catalog_from_context(context)


def _resolve_notebook_plot_spec(context, *, plot_id: str):
    return resolve_plot_spec(
        plots=context.config.plots,
        plot_id=plot_id,
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        shape_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )


def _frame_load_error(frame) -> str:
    if not isinstance(getattr(frame, "attrs", None), dict):
        return ""
    return str(frame.attrs.get("load_error") or "").strip()


def _ordered_plot_live_inputs_readiness(
    context,
    *,
    notebook: WorkspaceNotebookConfig,
    controls: WorkspaceNotebookControls,
) -> tuple[bool, list[str]]:
    ordered_plot_ids = _notebook_plot_ids(context, notebook)
    if not ordered_plot_ids:
        return True, []
    joinable_tables = [row.model_dump(mode="json") for row in controls.geometry_controls.joinable_tables]
    freshness_cache = FreshnessCache()
    warnings: list[str] = []
    ready = True
    for plot_id in ordered_plot_ids:
        freshness = evaluate_artifact_freshness(
            context,
            artifact_kind="plot",
            artifact_id=plot_id,
            cache=freshness_cache,
        )
        if freshness["status"] != "ok":
            ready = False
            warnings.append(
                f"ordered plot `{plot_id}` freshness requires attention: "
                + str(freshness.get("reason") or freshness["status"])
            )
            continue
        try:
            plot_spec = _resolve_notebook_plot_spec(context, plot_id=plot_id).model_dump(mode="json")
            frames = load_plot_review_frames(
                plot_spec,
                joinable_tables=joinable_tables,
                output_root=context.output_root,
            )
        except Exception as exc:
            ready = False
            warnings.append(f"ordered plot `{plot_id}` live inputs failed: {exc}")
            continue
        frame_errors = list(dict.fromkeys(_frame_load_error(frame) for frame in frames if _frame_load_error(frame)))
        if frame_errors:
            ready = False
            warnings.append(f"ordered plot `{plot_id}` live inputs failed: " + "; ".join(frame_errors))
    return ready, warnings


def _notebook_smoke_status(payload: dict[str, object]) -> str:
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return "error"
    blocking_checks = (
        "notebook_exists",
        "control_plane_loads",
        "imports_resolve",
        "marimo_check_passes",
        "plot_catalog_loads",
    )
    if any(not bool(checks.get(name)) for name in blocking_checks):
        return "error"
    degraded_checks = (
        "default_deliverable_ready",
        "study_doc_subsections_resolve",
        "static_links_resolve",
        "ordered_plot_live_inputs_ready",
    )
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

    default_deliverable_status = _default_deliverable_status(context, notebook.default_deliverable)
    freshness_blockers = _blocking_default_deliverable_reasons(
        notebook_id=notebook_id,
        deliverable_status=default_deliverable_status,
    )
    freshness_blockers.extend(_blocking_ordered_plot_freshness_reasons(context, notebook=notebook))
    if freshness_blockers:
        owner_deliverables = _ordered_plot_owner_deliverables(context, notebook=notebook)
        owner_recipes = sorted(
            {
                context.require_deliverable(deliverable_id).recipe
                for deliverable_id in owner_deliverables
                if deliverable_id in context.config.deliverables
            }
        )
        deliverable_hints = ", ".join(
            f"`latentdna deliverable run {deliverable_id} --workspace {context.workspace_dir}`"
            for deliverable_id in owner_deliverables
        )
        recipe_hints = ", ".join(
            f"`latentdna recipe run {recipe_id} --workspace {context.workspace_dir}`" for recipe_id in owner_recipes
        )
        raise WorkspaceValidationError(
            "notebook generation blocked because one or more notebook inputs have freshness drift: "
            + "; ".join(freshness_blockers)
            + ". Refresh the stale plot-owning deliverables "
            + (
                deliverable_hints
                or f"`latentdna deliverable run {notebook.default_deliverable} --workspace {context.workspace_dir}`"
            )
            + " or rerun their canonical recipe(s) "
            + (
                recipe_hints
                or (
                    f"`latentdna recipe run {context.require_deliverable(notebook.default_deliverable).recipe} "
                    f"--workspace {context.workspace_dir}`"
                )
            )
            + " before regenerating the notebook."
        )

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
        deliverable_status=default_deliverable_status,
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
                "marimo_check_passes": False,
                "plot_catalog_loads": False,
                "default_deliverable_ready": default_deliverable_status == "ok",
                "study_doc_subsections_resolve": False,
                "static_links_resolve": False,
                "ordered_plot_live_inputs_ready": False,
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
        "marimo_check_passes": False,
        "plot_catalog_loads": False,
        "default_deliverable_ready": False,
        "study_doc_subsections_resolve": False,
        "static_links_resolve": False,
        "ordered_plot_live_inputs_ready": False,
    }
    warnings: list[str] = []
    controls: WorkspaceNotebookControls | None = None

    if checks["notebook_exists"]:
        try:
            runpy.run_path(notebook_path.as_posix(), init_globals={"__name__": "__latentdna_smoke__"})
            checks["imports_resolve"] = True
        except Exception as exc:  # pragma: no cover - surfaced in health payload
            warnings.append(f"imports_resolve failed: {exc}")
        marimo_check = subprocess.run(
            [sys.executable, "-m", "marimo", "check", notebook_path.as_posix()],
            capture_output=True,
            text=True,
            check=False,
        )
        checks["marimo_check_passes"] = marimo_check.returncode == 0
        if not checks["marimo_check_passes"]:
            detail = (marimo_check.stderr or marimo_check.stdout or "").strip()
            warnings.append(f"marimo_check_passes failed: {detail or 'marimo check returned nonzero'}")
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
    study_doc_warnings = _ordered_plot_study_doc_subsection_warnings(context, notebook=notebook)
    checks["study_doc_subsections_resolve"] = not study_doc_warnings
    warnings.extend(study_doc_warnings)
    output_paths = _notebook_plot_output_paths(context, notebook)
    checks["static_links_resolve"] = bool(output_paths) and all(path.is_file() for path in output_paths)
    if controls is not None:
        ordered_plot_inputs_ready, ordered_plot_warnings = _ordered_plot_live_inputs_readiness(
            context,
            notebook=notebook,
            controls=controls,
        )
        checks["ordered_plot_live_inputs_ready"] = ordered_plot_inputs_ready
        warnings.extend(ordered_plot_warnings)

    status = "ok" if all(checks.values()) else "error"
    return _write_notebook_health(
        context,
        notebook_id=resolved_notebook_id,
        status=status,
        checks=checks,
        warnings=warnings,
    )
