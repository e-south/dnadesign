"""
Workspace catalog and explain surfaces for latentdna.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.errors import MissingArtifactError
from ..io.json_io import read_json, write_json
from ..studies.docs_refs import read_docs_ref
from ..workspaces.loader import WorkspaceContext, load_workspace_config
from ._artifacts import artifact_dir, artifact_exists, prune_retired_managed_artifacts
from .candidate_inventory_service import build_candidate_inventory
from .deliverable_service import deliverable_status_from_context
from .freshness_service import FreshnessCache
from .plot_service import write_plot_index


def _workspace_state(states: list[str]) -> str:
    if any(state == "error" for state in states):
        return "error"
    if any(state == "missing" for state in states):
        return "missing"
    if any(state == "attention" for state in states):
        return "attention"
    return "ok"


def workspace_catalog_from_context(context: WorkspaceContext) -> dict[str, object]:
    pruned_artifacts = prune_retired_managed_artifacts(
        context,
        artifact_kinds=("alignment_set", "export_bundle", "notebook", "view"),
    )
    deliverable_rows = []
    docs_refs: list[dict[str, str]] = []
    freshness_cache = FreshnessCache()
    for deliverable_id in sorted(context.config.deliverables):
        status = deliverable_status_from_context(context, deliverable_id, freshness_cache=freshness_cache)
        deliverable_rows.append(status.model_dump(mode="json"))
        for row in status.docs_refs:
            docs_refs.append(row)

    plots_payload = write_plot_index(context, freshness_cache=freshness_cache)
    notebooks = []
    for notebook_id in sorted(context.config.notebooks):
        manifest = _artifact_manifest(
            context,
            artifact_kind="notebook",
            artifact_id=notebook_id,
        )
        notebooks.append(
            {
                "notebook_id": notebook_id,
                "path": artifact_dir(context, artifact_kind="notebook", artifact_id=notebook_id).as_posix(),
                "present": artifact_exists(context, artifact_kind="notebook", artifact_id=notebook_id),
                "status": manifest.get("status") if manifest is not None else "missing",
                "warnings": manifest.get("warnings", []) if manifest is not None else [],
            }
        )
    exports = []
    for export_id in sorted(context.config.exports):
        manifest = _artifact_manifest(
            context,
            artifact_kind="export_bundle",
            artifact_id=export_id,
        )
        exports.append(
            {
                "export_id": export_id,
                "path": artifact_dir(context, artifact_kind="export_bundle", artifact_id=export_id).as_posix(),
                "present": artifact_exists(context, artifact_kind="export_bundle", artifact_id=export_id),
                "status": manifest.get("status") if manifest is not None else "missing",
                "warnings": manifest.get("warnings", []) if manifest is not None else [],
            }
        )
    runs = _run_rows(context.output_root / "runs")
    state = _workspace_state([str(row["status"]) for row in deliverable_rows])
    payload = {
        "workspace_id": context.workspace_id,
        "title": context.config.workspace.title or context.workspace_id,
        "state": state,
        "deliverables": deliverable_rows,
        "plots": plots_payload.get("plots", []),
        "notebooks": notebooks,
        "exports": exports,
        "candidate_inventory": build_candidate_inventory(context, freshness_cache=freshness_cache),
        "runs": runs,
        "docs_refs": docs_refs,
        "pruned_artifacts": pruned_artifacts,
    }
    write_json(context.output_root / "catalog.json", payload)
    return payload


def workspace_catalog(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    return workspace_catalog_from_context(context)


def explain_workspace(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    return workspace_catalog_from_context(context)


def explain_deliverable(workspace: str | Path, deliverable_id: str) -> dict[str, object]:
    context = load_workspace_config(workspace)
    status = deliverable_status_from_context(context, deliverable_id).model_dump(mode="json")
    status["docs_context"] = [read_docs_ref(context, row["docs_ref"]) for row in status["docs_refs"]]
    return status


def explain_plot(workspace: str | Path, plot_id: str) -> dict[str, object]:
    context = load_workspace_config(workspace)
    if not artifact_exists(context, artifact_kind="plot", artifact_id=plot_id):
        raise MissingArtifactError(f"plot artifact not found: {plot_id}")
    manifest_path = artifact_dir(context, artifact_kind="plot", artifact_id=plot_id) / "manifest.json"
    plot = context.config.plots.get(plot_id)
    manifest = context.read_manifest(manifest_path)
    params = manifest.get("params") if isinstance(manifest.get("params"), dict) else {}
    return {
        "workspace_id": context.workspace_id,
        "plot_id": plot_id,
        "kind": plot.kind if plot is not None else params.get("kind"),
        "config": plot.model_dump(mode="json") if plot is not None else None,
        "path": artifact_dir(context, artifact_kind="plot", artifact_id=plot_id).as_posix(),
        "manifest": manifest,
    }


def explain_export(workspace: str | Path, export_id: str) -> dict[str, object]:
    context = load_workspace_config(workspace)
    if not artifact_exists(context, artifact_kind="export_bundle", artifact_id=export_id):
        raise MissingArtifactError(f"export artifact not found: {export_id}")
    export = context.require_export(export_id)
    manifest_path = artifact_dir(context, artifact_kind="export_bundle", artifact_id=export_id) / "manifest.json"
    return {
        "workspace_id": context.workspace_id,
        "export_id": export_id,
        "config": export.model_dump(mode="json"),
        "path": artifact_dir(context, artifact_kind="export_bundle", artifact_id=export_id).as_posix(),
        "manifest": context.read_manifest(manifest_path),
    }


def _run_rows(runs_root: Path) -> list[dict[str, object]]:
    items = []
    if not runs_root.is_dir():
        return items
    for run_dir in sorted(candidate for candidate in runs_root.iterdir() if candidate.is_dir()):
        run_json = run_dir / "run.json"
        if run_json.is_file():
            items.append(read_json(run_json))
    return items


def _artifact_manifest(context, *, artifact_kind: str, artifact_id: str) -> dict[str, object] | None:
    manifest_path = artifact_dir(context, artifact_kind=artifact_kind, artifact_id=artifact_id) / "manifest.json"
    if not manifest_path.is_file():
        return None
    return context.read_manifest(manifest_path)
