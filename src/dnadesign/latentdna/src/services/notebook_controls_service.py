"""
Workspace notebook control-plane assembly for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from os.path import relpath
from pathlib import Path

from ..contracts.notebook import (
    WorkspaceNotebookControls,
    WorkspaceNotebookPlotControls,
    WorkspaceNotebookPlotEntry,
    WorkspaceNotebookRuntimePaths,
)
from ..io.json_io import read_json
from .notebook_context_audit import build_workspace_notebook_context_audit
from .notebook_geometry_controls import build_workspace_geometry_controls


def _runtime_paths(context, *, notebook_id: str) -> WorkspaceNotebookRuntimePaths:
    notebook_dir = context.output_root / "notebooks" / notebook_id

    def relative_to_notebook(target: Path) -> str:
        return Path(relpath(target, start=notebook_dir)).as_posix()

    return WorkspaceNotebookRuntimePaths(
        workspace_relative_path=relative_to_notebook(context.workspace_dir),
        output_relative_path=relative_to_notebook(context.output_root),
        catalog_relative_path=relative_to_notebook(context.output_root / "catalog.json"),
        health_relative_path=relative_to_notebook(context.output_root / "notebooks" / notebook_id / "health.json"),
    )


def _plot_controls(
    context,
    *,
    notebook_id: str,
    catalog_payload: dict[str, object] | None = None,
) -> WorkspaceNotebookPlotControls:
    notebook = context.require_notebook(notebook_id)
    ordered_plot_ids = list(getattr(notebook, "ordered_plots", []) or [])
    if not ordered_plot_ids:
        ordered_plot_ids = list(context.require_deliverable(notebook.default_deliverable).outputs.get("plots", []))
    catalog_plot_rows = {
        str(row.get("plot_id")): row
        for row in (catalog_payload or {}).get("plots", [])
        if isinstance(row, dict) and row.get("plot_id")
    }
    plot_entries: list[WorkspaceNotebookPlotEntry] = []
    for plot_id in ordered_plot_ids:
        owner_deliverable_ids = [
            current_deliverable_id
            for current_deliverable_id, deliverable in context.config.deliverables.items()
            if plot_id in deliverable.outputs.get("plots", [])
        ]
        deliverable_id = ""
        if notebook.default_deliverable in owner_deliverable_ids:
            deliverable_id = notebook.default_deliverable
        elif owner_deliverable_ids:
            deliverable_id = owner_deliverable_ids[0]
        deliverable_title = (
            context.config.deliverables[deliverable_id].title if deliverable_id in context.config.deliverables else ""
        )
        plot = context.require_plot(plot_id)
        manifest_path = context.output_root / "plots" / plot_id / "manifest.json"
        manifest: dict[str, object] = {}
        if manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
            except Exception:
                manifest = {"status": "error", "stale": False}
        catalog_row = catalog_plot_rows.get(plot_id, {})
        plot_entries.append(
            WorkspaceNotebookPlotEntry(
                plot_id=plot_id,
                deliverable_id=deliverable_id,
                deliverable_title=deliverable_title,
                visibility_tier=str(getattr(plot, "visibility_tier", "primary") or "primary"),
                status=str(catalog_row.get("status") or manifest.get("status") or "missing"),
                stale=bool(catalog_row.get("stale") if catalog_row.get("stale") is not None else manifest.get("stale")),
            )
        )
    return WorkspaceNotebookPlotControls(
        default_surface=str(getattr(notebook, "default_surface", "plots") or "plots"),
        ordered_plot_ids=ordered_plot_ids,
        plots=plot_entries,
    )


def build_workspace_notebook_controls_payload(
    context,
    *,
    notebook_id: str,
    catalog_payload: dict[str, object] | None = None,
) -> WorkspaceNotebookControls:
    return WorkspaceNotebookControls(
        schema_version="latentdna.workspace_notebook_controls.v4",
        workspace_id=context.workspace_id,
        notebook_id=notebook_id,
        generated_at=datetime.now(UTC).isoformat(),
        runtime_paths=_runtime_paths(context, notebook_id=notebook_id),
        plot_controls=_plot_controls(context, notebook_id=notebook_id, catalog_payload=catalog_payload),
        geometry_controls=build_workspace_geometry_controls(context, notebook_id=notebook_id),
        context_audit=build_workspace_notebook_context_audit(context),
    )
