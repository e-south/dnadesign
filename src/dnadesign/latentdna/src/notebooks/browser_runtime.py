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

from ..studies.docs_refs import read_docs_ref
from ..workspaces.loader import load_workspace_config
from ..workspaces.plot_semantics import resolve_plot_semantics
from .browser_runtime_compare import (
    compare_pair_payload,
    render_distance_correlation,
    render_rowwise_distribution,
)
from .browser_runtime_projection import enrich_projection_frame, render_projection_grid
from .browser_runtime_support import (
    available_hues_for_frames,
    candidate_hue_columns,
    category_color_map,
    display_hue_label,
    geometry_map,
    include_hue_column,
    key_value_table,
    load_json,
    load_table,
    load_workspace_notebook_controls,
    notebook_theme,
    option_key_for_value,
    read_text,
    render_plot_asset,
    select_plot_render_path,
    style_notebook_axes,
    table_from_records,
    unique_in_order,
)

__all__ = ["build_workspace_browser_runtime", "load_workspace_notebook_controls", "resolve_plot_doc_block"]


_ALLOWED_RUNTIME_HUE_KINDS = {"categorical", "binary", "continuous"}


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
    hue_kinds: dict[str, str]
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
class BrowserPlotReview:
    default_surface: str
    ordered_plot_ids: list[str]
    sections: list[dict[str, object]]


@dataclass(frozen=True)
class BrowserSupport:
    available_hues_for_frames: Callable[..., list[str]]
    candidate_hue_columns: Callable[[pd.DataFrame, list[str], set[str] | None], list[str]]
    category_color_map: Callable[[list[str]], dict[str, str]]
    display_hue_label: Callable[[str], str]
    json: ModuleType
    key_value_table: Callable[..., object]
    load_json: Callable[[Path], dict[str, object]]
    load_table: Callable[[Path], pd.DataFrame]
    mo: ModuleType
    notebook_theme: Callable[[], object]
    option_key_for_value: Callable[[dict[str, object], object], str | None]
    pd: ModuleType
    read_text: Callable[[str | None], str | None]
    select_plot_render_path: Callable[[list[Path]], Path | None]
    style_notebook_axes: Callable[..., None]
    table_from_records: Callable[..., object]
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
    plot_review: BrowserPlotReview
    renderers: BrowserRenderers
    support: BrowserSupport


def _humanize_plot_id(plot_id: str) -> str:
    return plot_id.replace("_", " ").strip().title()


def _parse_deliverable_markdown(markdown: str) -> dict[str, object]:
    lines = markdown.splitlines()
    summary_lines: list[str] = []
    plot_sections: dict[str, dict[str, str]] = {}

    first_h1 = next((index for index, line in enumerate(lines) if line.startswith("# ")), None)
    if first_h1 is not None:
        index = first_h1 + 1
        while index < len(lines):
            line = lines[index]
            if line.startswith("## "):
                break
            summary_lines.append(line)
            index += 1

    heading_indices = [index for index, line in enumerate(lines) if line.startswith("### ")]
    heading_indices.append(len(lines))
    for start, end in zip(heading_indices, heading_indices[1:], strict=False):
        heading = lines[start][4:].strip()
        if "|" not in heading:
            continue
        plot_id_text, title_text = (part.strip() for part in heading.split("|", 1))
        plot_sections[plot_id_text] = {
            "title": title_text,
            "markdown": "\n".join(lines[start + 1 : end]).strip(),
        }

    return {
        "summary_markdown": "\n".join(summary_lines).strip(),
        "plot_sections": plot_sections,
    }


def resolve_plot_doc_block(
    *,
    plot_id: str,
    deliverable_summary: str,
    parsed_markdown: dict[str, object] | None,
) -> dict[str, str | None]:
    plot_sections = parsed_markdown.get("plot_sections", {}) if isinstance(parsed_markdown, dict) else {}
    summary_markdown = (
        str(parsed_markdown.get("summary_markdown") or "").strip() if isinstance(parsed_markdown, dict) else ""
    )
    plot_entry = plot_sections.get(plot_id) if isinstance(plot_sections, dict) else None
    if isinstance(plot_entry, dict):
        return {
            "title": str(plot_entry.get("title") or _humanize_plot_id(plot_id)),
            "markdown": str(plot_entry.get("markdown") or "").strip(),
            "warning": None,
        }
    fallback_markdown = summary_markdown or deliverable_summary.strip()
    return {
        "title": _humanize_plot_id(plot_id),
        "markdown": fallback_markdown,
        "warning": f"Missing plot-specific study-doc subsection for `{plot_id}`.",
    }


def resolve_runtime_hue_kinds(
    global_hue_columns: list[str],
    configured_hue_kinds: dict[str, object],
) -> dict[str, str]:
    return {
        column: str(configured_hue_kinds.get(column))
        for column in global_hue_columns
        if str(configured_hue_kinds.get(column)) in _ALLOWED_RUNTIME_HUE_KINDS
    }


def _plot_review_sections(context, *, output_root: Path, controls: dict[str, object]) -> BrowserPlotReview:
    plot_controls = controls.get("plot_controls", {})
    ordered_plot_ids = [str(item) for item in plot_controls.get("ordered_plot_ids", []) if isinstance(item, str)]
    default_surface = str(plot_controls.get("default_surface") or "plots")
    if not ordered_plot_ids:
        return BrowserPlotReview(default_surface=default_surface, ordered_plot_ids=[], sections=[])

    plot_entries = {
        str(item.get("plot_id")): item
        for item in plot_controls.get("plots", [])
        if isinstance(item, dict) and item.get("plot_id")
    }
    docs_cache: dict[str, dict[str, object]] = {}
    sections: list[dict[str, object]] = []
    current_section: dict[str, object] | None = None

    for plot_id in ordered_plot_ids:
        entry = plot_entries.get(plot_id, {})
        deliverable_id = str(entry.get("deliverable_id") or "")
        deliverable = context.config.deliverables.get(deliverable_id) if deliverable_id else None
        deliverable_title = str(entry.get("deliverable_title") or "") or (
            deliverable.title if deliverable is not None else _humanize_plot_id(deliverable_id)
        )
        deliverable_summary = deliverable.summary if deliverable is not None else ""
        if deliverable_id not in docs_cache:
            parsed: dict[str, object] | None = None
            if deliverable is not None and deliverable.docs_refs:
                try:
                    docs_payload = read_docs_ref(context, deliverable.docs_refs[0])
                    parsed = _parse_deliverable_markdown(str(docs_payload.get("content") or ""))
                except Exception:
                    parsed = None
            docs_cache[deliverable_id] = parsed or {}
        doc_block = resolve_plot_doc_block(
            plot_id=plot_id,
            deliverable_summary=deliverable_summary,
            parsed_markdown=docs_cache.get(deliverable_id),
        )
        plot_dir = output_root / "plots" / plot_id
        manifest = load_json(plot_dir / "manifest.json")
        semantics = (
            manifest.get("semantics", {})
            if isinstance(manifest.get("semantics"), dict)
            else resolve_plot_semantics(context, plot_id=plot_id, allow_generated_fallback=True).model_dump(mode="json")
        )
        output_paths = [
            plot_dir / str(output.get("path"))
            for output in manifest.get("outputs", [])
            if isinstance(output, dict) and output.get("path")
        ]
        render_path = select_plot_render_path(output_paths)
        visibility_tier = str(
            entry.get("visibility_tier") or getattr(context.require_plot(plot_id), "visibility_tier", "primary")
        )
        badge = "Appendix" if visibility_tier == "appendix" else "Primary"
        guardrail_text = None
        guardrails = [str(item).strip() for item in semantics.get("interpretation_guardrails", []) if str(item).strip()]
        if visibility_tier == "appendix":
            guardrail_text = "Appendix orientation surface; not primary decision evidence."
        elif guardrails:
            guardrail_text = guardrails[0]

        if current_section is None or str(current_section.get("deliverable_id")) != deliverable_id:
            current_section = {
                "deliverable_id": deliverable_id,
                "title": deliverable_title,
                "summary": deliverable_summary,
                "cards": [],
            }
            sections.append(current_section)
        current_section["cards"].append(
            {
                "plot_id": plot_id,
                "title": str(doc_block.get("title") or _humanize_plot_id(plot_id)),
                "badge": badge,
                "visibility_tier": visibility_tier,
                "render_path": render_path,
                "caption_md": str(semantics.get("caption_md") or "").strip(),
                "study_doc_md": str(doc_block.get("markdown") or "").strip(),
                "study_doc_warning": doc_block.get("warning"),
                "guardrail_text": guardrail_text,
            }
        )

    return BrowserPlotReview(
        default_surface=default_surface,
        ordered_plot_ids=ordered_plot_ids,
        sections=sections,
    )


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

    geometry_control = controls.get("geometry_controls", {})
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
    configured_hue_kinds = geometry_control.get("hue_kinds", {})
    reference_labels = [str(item) for item in geometry_control.get("reference_labels", []) if isinstance(item, str)]
    global_hue_columns = unique_in_order(
        [
            str(column)
            for item in joinable_tables
            for column in item.get("columns", [])
            if isinstance(column, str)
            and str(column) in preferred_hues
            and include_hue_column(str(column), joinable_artifact_suffixes)
        ]
    )
    hue_kinds = resolve_runtime_hue_kinds(global_hue_columns, configured_hue_kinds)
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
    plot_review = _plot_review_sections(context, output_root=output_root, controls=controls)

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
            hue_kinds=hue_kinds,
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
        plot_review=plot_review,
        support=BrowserSupport(
            available_hues_for_frames=available_hues_for_frames,
            candidate_hue_columns=candidate_hue_columns,
            category_color_map=category_color_map,
            display_hue_label=display_hue_label,
            json=json,
            key_value_table=key_value_table,
            load_json=load_json,
            load_table=load_table,
            mo=mo,
            notebook_theme=notebook_theme,
            option_key_for_value=option_key_for_value,
            pd=pd,
            read_text=read_text,
            select_plot_render_path=select_plot_render_path,
            style_notebook_axes=style_notebook_axes,
            table_from_records=table_from_records,
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
