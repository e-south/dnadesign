"""
Browser runtime assembly for generated latentdna marimo notebooks.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType

import marimo as mo
import numpy as np
import pandas as pd

from ..workspaces.loader import load_workspace_config
from .browser_runtime_compare import (
    compare_pair_payload,
    render_distance_correlation,
    render_rowwise_distribution,
)
from .browser_runtime_projection import enrich_projection_frame, render_projection_grid
from .browser_runtime_support import (
    candidate_hue_columns,
    display_hue_label,
    geometry_map,
    include_hue_column,
    load_json,
    load_table,
    load_workspace_notebook_controls,
    option_key_for_value,
    read_text,
    render_plot_asset,
    unique_in_order,
)

__all__ = ["build_workspace_browser_runtime", "load_workspace_notebook_controls"]


@dataclass(frozen=True)
class BrowserIdentity:
    description: str | None
    default_deliverable: str
    dimensionality_text: str
    notebook_id: str
    output_root: Path
    row_count_text: str
    source_labels: list[str]
    title: str
    vector_columns: list[str]
    visual_families: list[str]
    workspace_dir: Path
    workspace_id: str


@dataclass(frozen=True)
class BrowserCatalog:
    controls: dict[str, object]
    default_section: str
    deliverables: list[dict[str, object]]
    exports: list[dict[str, object]]
    health: dict[str, object]
    notebooks: list[dict[str, object]]
    plots: list[dict[str, object]]
    runs: list[dict[str, object]]
    section_names: list[str]


@dataclass(frozen=True)
class BrowserGeometry:
    compare_left_default: str
    compare_metrics: dict[str, object]
    compare_right_default: str
    comparison_bases: list[dict[str, object]]
    geometry_control: dict[str, object]
    geometry_rows: list[dict[str, object]]
    geometry_rows_by_id: dict[str, dict[str, object]]
    global_hue_columns: list[str]
    joinable_artifact_suffixes: set[str]
    joinable_tables: list[dict[str, object]]
    layout_default: str
    layout_options: dict[str, str]
    layout_presets: list[dict[str, object]]
    model_default: str
    model_values: list[str]
    preferred_hues: list[str]
    reference_labels: list[str]
    selected_hue_default: str


@dataclass(frozen=True)
class BrowserSupport:
    candidate_hue_columns: Callable[[pd.DataFrame, list[str], set[str] | None], list[str]]
    display_hue_label: Callable[[str], str]
    json: ModuleType
    load_json: Callable[[Path], dict[str, object]]
    load_table: Callable[[Path], pd.DataFrame]
    mo: ModuleType
    option_key_for_value: Callable[[dict[str, object], object], str | None]
    pd: ModuleType
    read_text: Callable[[str | None], str | None]
    unique_in_order: Callable[[object], list[str]]


@dataclass(frozen=True)
class BrowserRenderers:
    compare_pair_payload: Callable[..., dict[str, object]]
    enrich_projection_frame: Callable[[pd.DataFrame, list[dict[str, object]]], pd.DataFrame]
    render_distance_correlation: Callable[..., object]
    render_plot_asset: Callable[[Path], object]
    render_projection_grid: Callable[..., object]
    render_rowwise_distribution: Callable[..., object]


@dataclass(frozen=True)
class WorkspaceBrowserRuntime:
    catalog: BrowserCatalog
    geometry: BrowserGeometry
    identity: BrowserIdentity
    renderers: BrowserRenderers
    support: BrowserSupport


def build_workspace_browser_runtime(
    *,
    title: str,
    description: str | None,
    workspace_id: str,
    notebook_id: str,
    default_deliverable: str,
    workspace_dir: Path,
    output_root: Path,
    catalog_path: Path,
    health_path: Path,
    controls: dict[str, object],
) -> WorkspaceBrowserRuntime:
    context = load_workspace_config(workspace_dir)
    catalog = load_json(catalog_path)
    health = load_json(health_path)
    deliverables = [
        row for row in catalog.get("deliverables", []) if isinstance(row, dict) and row.get("deliverable_id")
    ]
    plots = [row for row in catalog.get("plots", []) if isinstance(row, dict) and row.get("plot_id")]
    exports = [row for row in catalog.get("exports", []) if isinstance(row, dict)]
    notebooks = [row for row in catalog.get("notebooks", []) if isinstance(row, dict)]
    runs = [row for row in catalog.get("runs", []) if isinstance(row, dict)]

    section_names = unique_in_order(row.get("section") for row in deliverables)
    default_deliverable_row = next(
        (row for row in deliverables if str(row.get("deliverable_id")) == default_deliverable),
        deliverables[0] if deliverables else None,
    )
    default_section = (
        str(default_deliverable_row.get("section") or "Unsectioned")
        if default_deliverable_row is not None
        else (section_names[0] if section_names else "Unsectioned")
    )

    source_labels = []
    for source_id, source in context.config.sources.items():
        if hasattr(source, "dataset"):
            source_labels.append(f"{source_id}:{source.dataset}")
        elif hasattr(source, "path"):
            source_labels.append(f"{source_id}:{source.path}")
        else:
            source_labels.append(source_id)
    vector_columns = sorted(
        {
            view.vector.name
            for view in context.config.views.values()
            if hasattr(view, "vector") and getattr(view.vector, "kind", None) == "column"
        }
    )
    visual_families = unique_in_order(
        getattr(view, "tags", {}).get("family")
        for view in context.config.views.values()
        if getattr(view, "tags", {}).get("family") is not None
    )
    matrix_shapes = []
    for view_id in context.config.views:
        matrix_path = output_root / "views" / view_id / "matrix.npy"
        if not matrix_path.is_file():
            continue
        matrix = np.load(matrix_path, mmap_mode="r")
        matrix_shapes.append({"view_id": view_id, "rows": int(matrix.shape[0]), "dims": int(matrix.shape[1])})
    row_count_text = "unknown"
    dimensionality_text = "unknown"
    if matrix_shapes:
        row_count_text = ", ".join(f"{row['view_id']}={row['rows']}" for row in matrix_shapes[:4])
        dimensionality_text = ", ".join(f"{row['view_id']}={row['dims']}" for row in matrix_shapes[:4])

    geometry_control = controls.get("geometry_switchboard", {})
    geometry_rows = [
        row for row in geometry_control.get("geometries", []) if isinstance(row, dict) and row.get("view_id")
    ]
    geometry_rows_by_id = geometry_map(geometry_rows)
    joinable_tables = [
        row for row in geometry_control.get("joinable_tables", []) if isinstance(row, dict) and row.get("relative_path")
    ]
    layout_presets = [
        row for row in geometry_control.get("layout_presets", []) if isinstance(row, dict) and row.get("id")
    ]
    comparison_bases = [
        row for row in geometry_control.get("comparison_bases", []) if isinstance(row, dict) and row.get("id")
    ]
    joinable_artifact_suffixes = {
        str(row.get("artifact_id"))
        for row in joinable_tables
        if isinstance(row.get("artifact_id"), str) and str(row.get("artifact_id"))
    }
    compare_metrics = geometry_control.get("compare_metrics", {})
    preferred_hues = [str(item) for item in geometry_control.get("preferred_hues", []) if isinstance(item, str)]
    reference_labels = [str(item) for item in geometry_control.get("reference_labels", []) if isinstance(item, str)]
    global_hue_columns = unique_in_order(
        preferred_hues
        + [
            str(column)
            for item in joinable_tables
            for column in item.get("columns", [])
            if isinstance(column, str) and include_hue_column(str(column), joinable_artifact_suffixes)
        ]
    )
    model_values = unique_in_order(row.get("model") for row in geometry_rows) or ["20b"]
    model_default = (
        str(geometry_control.get("default_model"))
        if str(geometry_control.get("default_model")) in model_values
        else model_values[0]
    )
    layout_options = {str(row["label"]): str(row["id"]) for row in layout_presets} or {"Single view": "single_view"}
    layout_default = (
        str(geometry_control.get("default_layout"))
        if str(geometry_control.get("default_layout")) in set(layout_options.values())
        else next(iter(layout_options.values()))
    )
    selected_hue_default = (
        "design_family"
        if "design_family" in global_hue_columns
        else (global_hue_columns[0] if global_hue_columns else "")
    )

    enrich_projection_frame_for_output = partial(enrich_projection_frame, output_root=output_root)
    render_plot_asset_for_workspace = partial(render_plot_asset, workspace_dir=workspace_dir)
    render_projection_grid_for_workspace = partial(
        render_projection_grid,
        output_root=output_root,
        workspace_dir=workspace_dir,
    )
    compare_pair_payload_for_output = partial(compare_pair_payload, output_root=output_root)

    return WorkspaceBrowserRuntime(
        identity=BrowserIdentity(
            description=description,
            default_deliverable=default_deliverable,
            dimensionality_text=dimensionality_text,
            notebook_id=notebook_id,
            output_root=output_root,
            row_count_text=row_count_text,
            source_labels=source_labels,
            title=title,
            vector_columns=vector_columns,
            visual_families=visual_families,
            workspace_dir=workspace_dir,
            workspace_id=workspace_id,
        ),
        catalog=BrowserCatalog(
            controls=controls,
            default_section=default_section,
            deliverables=deliverables,
            exports=exports,
            health=health,
            notebooks=notebooks,
            plots=plots,
            runs=runs,
            section_names=section_names,
        ),
        geometry=BrowserGeometry(
            compare_left_default=str(geometry_control.get("default_compare_left") or ""),
            compare_metrics=compare_metrics if isinstance(compare_metrics, dict) else {},
            compare_right_default=str(geometry_control.get("default_compare_right") or ""),
            comparison_bases=comparison_bases,
            geometry_control=geometry_control,
            geometry_rows=geometry_rows,
            geometry_rows_by_id=geometry_rows_by_id,
            global_hue_columns=global_hue_columns,
            joinable_artifact_suffixes=joinable_artifact_suffixes,
            joinable_tables=joinable_tables,
            layout_default=layout_default,
            layout_options=layout_options,
            layout_presets=layout_presets,
            model_default=model_default,
            model_values=model_values,
            preferred_hues=preferred_hues,
            reference_labels=reference_labels,
            selected_hue_default=selected_hue_default,
        ),
        support=BrowserSupport(
            candidate_hue_columns=candidate_hue_columns,
            display_hue_label=display_hue_label,
            json=json,
            load_json=load_json,
            load_table=load_table,
            mo=mo,
            option_key_for_value=option_key_for_value,
            pd=pd,
            read_text=read_text,
            unique_in_order=unique_in_order,
        ),
        renderers=BrowserRenderers(
            compare_pair_payload=compare_pair_payload_for_output,
            enrich_projection_frame=enrich_projection_frame_for_output,
            render_distance_correlation=render_distance_correlation,
            render_plot_asset=render_plot_asset_for_workspace,
            render_projection_grid=render_projection_grid_for_workspace,
            render_rowwise_distribution=render_rowwise_distribution,
        ),
    )
