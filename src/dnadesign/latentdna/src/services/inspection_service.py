"""
Inspection services for latentdna.
"""

from __future__ import annotations

from pathlib import Path

from ..distances.score import _select_indices
from ..io.json_io import read_json
from ..sources.resolver import inspect_source_schema, read_records_table, resolve_source
from ..workspaces.loader import load_workspace_config
from ._artifacts import artifact_dir, artifact_exists, artifact_manifest_path
from .plot_service import write_plot_index
from .run_service import artifact_inventory


def inspect_source(workspace: str | Path, source_id: str) -> dict[str, object]:
    context = load_workspace_config(workspace)
    source = context.require_source(source_id)
    resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
    return {
        "schema_version": "latentdna.inspect_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "data": {"source_id": source_id, "kind": source.kind, **inspect_source_schema(resolved)},
    }


def inspect_views(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    views = []
    for view_id, view in sorted(context.config.views.items()):
        artifact_present = artifact_exists(context, artifact_kind="view", artifact_id=view_id)
        item = {
            "view_id": view_id,
            "coordinate_space_id": view.coordinate_space_id,
            "role": view.role,
            "materialized": artifact_present,
            "path": artifact_dir(context, artifact_kind="view", artifact_id=view_id).as_posix()
            if artifact_present
            else None,
        }
        if hasattr(view, "source"):
            item["declaration_kind"] = "source_backed"
            item["source"] = view.source
        else:
            item["declaration_kind"] = "derived"
            item["derive_kind"] = view.derive.kind
            if view.derive.kind == "vector_difference":
                item["left"] = view.derive.left
                item["right"] = view.derive.right
                item["alignment"] = view.derive.alignment
            elif view.derive.kind in {"concatenate", "block_normalized_concatenate"}:
                item["inputs"] = list(view.derive.inputs)
            else:
                item["input_view"] = view.derive.view
        views.append(item)
    return {
        "schema_version": "latentdna.inspect_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "data": {"views": views},
    }


def inspect_alignment(workspace: str | Path, alignment_id: str) -> dict[str, object]:
    context = load_workspace_config(workspace)
    alignment = context.require_alignment(alignment_id)
    artifact_present = artifact_exists(context, artifact_kind="alignment_set", artifact_id=alignment_id)
    return {
        "schema_version": "latentdna.inspect_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "data": {
            "alignment": {
                "alignment_id": alignment_id,
                "left": alignment.left,
                "right": alignment.right,
                "on": alignment.on,
                "left_on": alignment.left_on,
                "right_on": alignment.right_on,
                "support": alignment.support,
                "left_aggregation": alignment.left_aggregation,
                "right_aggregation": alignment.right_aggregation,
            },
            "artifact": (
                context.read_manifest(
                    artifact_manifest_path(context, artifact_kind="alignment_set", artifact_id=alignment_id)
                )
                if artifact_present
                else None
            ),
            "path": (
                artifact_dir(context, artifact_kind="alignment_set", artifact_id=alignment_id).as_posix()
                if artifact_present
                else None
            ),
        },
    }


def inspect_landmarks(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    landmarks = []
    for landmark_id, landmark in sorted(context.config.landmarks.items()):
        source = context.require_source(landmark.source)
        resolved = resolve_source(landmark.source, source, workspace_dir=context.workspace_dir)
        column = str(landmark.where["column"])
        rows = read_records_table(resolved, columns=[column]).to_pylist() if resolved.records_path is not None else []
        landmarks.append(
            {
                "landmark_id": landmark_id,
                "source": landmark.source,
                "representation_mode": landmark.representation.mode,
                "where": landmark.where,
                "matched_rows": len(_select_indices(rows, landmark.where)) if rows else None,
            }
        )
    return {
        "schema_version": "latentdna.inspect_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "data": {"landmarks": landmarks},
    }


def inspect_missingness(
    workspace: str | Path, source_id: str, *, columns: list[str] | None = None
) -> dict[str, object]:
    context = load_workspace_config(workspace)
    source = context.require_source(source_id)
    resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
    if resolved.records_path is not None:
        table = read_records_table(resolved, columns=columns)
    else:
        if resolved.rows_path is None:
            raise RuntimeError(f"missing rows path for source {source_id}")
        from ..io.parquet_io import read_table

        table = read_table(resolved.rows_path, columns=columns)
    summaries = []
    for column_name in table.column_names:
        column = table[column_name]
        summaries.append(
            {
                "column": column_name,
                "rows": table.num_rows,
                "null_count": column.null_count,
                "non_null_count": table.num_rows - column.null_count,
                "null_fraction": 0.0 if table.num_rows == 0 else column.null_count / table.num_rows,
            }
        )
    return {
        "schema_version": "latentdna.inspect_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "data": {"source_id": source_id, "missingness": summaries},
    }


def inspect_artifacts(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    return {
        "schema_version": "latentdna.inspect_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "data": {"artifacts": artifact_inventory(context)},
    }


def inspect_plots(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    index_path = context.output_root / "plots" / "index.json"
    payload = read_json(index_path) if index_path.is_file() else write_plot_index(context)
    return {
        "schema_version": "latentdna.inspect_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "data": {"plots": payload.get("plots", [])},
    }


def inspect_notebook_health(workspace: str | Path) -> dict[str, object]:
    context = load_workspace_config(workspace)
    notebook_ids = list(getattr(context.config, "notebooks", {}) or {})
    if not notebook_ids:
        raise FileNotFoundError("workspace does not declare any notebooks")
    if len(notebook_ids) == 1:
        health_path = context.output_root / "notebooks" / notebook_ids[0] / "health.json"
        if not health_path.is_file():
            raise FileNotFoundError(f"notebook health artifact not found: {health_path}")
        data = {"health": read_json(health_path)}
    else:
        health_by_notebook: dict[str, object] = {}
        for notebook_id in notebook_ids:
            health_path = context.output_root / "notebooks" / notebook_id / "health.json"
            if not health_path.is_file():
                raise FileNotFoundError(f"notebook health artifact not found: {health_path}")
            health_by_notebook[notebook_id] = read_json(health_path)
        data = {"health_by_notebook": health_by_notebook}
    return {
        "schema_version": "latentdna.inspect_result.v1",
        "workspace_id": context.workspace_id,
        "status": "ok",
        "data": data,
    }
