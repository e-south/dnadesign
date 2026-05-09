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
import pandas as pd

from ..labels import humanize_plot_title
from ..plots.recipes import resolve_plot_spec
from ..studies.docs_refs import read_docs_ref
from ..workspaces.loader import load_workspace_config
from ..workspaces.plot_semantics import resolve_plot_semantics
from .browser_runtime_compare import (
    compare_pair_payload,
    render_distance_correlation,
    render_rowwise_distribution,
)
from .browser_runtime_plot_review import load_plot_review_frames, render_plot_review_surface
from .browser_runtime_projection import enrich_projection_frame, load_projection_frame, render_projection_grid
from .browser_runtime_support import (
    available_hues_for_frames,
    candidate_hue_columns,
    category_color_map,
    display_hue_label,
    geometry_map,
    include_hue_column,
    key_value_table,
    labeled_options,
    load_json,
    load_table,
    load_workspace_notebook_controls,
    notebook_theme,
    option_key_for_value,
    read_text,
    style_notebook_axes,
    table_from_records,
    unique_in_order,
)
from .rendering import render_math_markdown, render_plot_asset, select_plot_render_path

__all__ = ["build_workspace_browser_runtime", "load_workspace_notebook_controls", "resolve_plot_doc_block"]


_ALLOWED_RUNTIME_HUE_KINDS = {"categorical", "binary", "continuous", "ordinal"}
_PLOT_REVIEW_LIVE_RENDER_KINDS = {
    "projection_grid",
    "xy_scatter_grid",
    "paired_xy_scatter_grid",
    "categorical_count",
    "metric_panel_grid",
    "distribution_grid",
    "curve_grid",
}


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
    axis_styles: dict[str, dict[str, object]]
    candidate_sets: list[dict[str, object]]
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
    row_metadata_hues: list[str]
    reference_annotation_default: str
    reference_annotation_options: dict[str, str]
    reference_hue_columns: list[str]
    reference_hue_options: dict[str, str]
    reference_labels: list[str]
    reference_required_columns: list[str]
    reference_sets: list[dict[str, object]]
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
    labeled_options: Callable[..., dict[str, object]]
    load_json: Callable[[Path], dict[str, object]]
    load_table: Callable[..., pd.DataFrame]
    mo: ModuleType
    notebook_theme: Callable[[], object]
    option_key_for_value: Callable[[dict[str, object], object], str | None]
    pd: ModuleType
    read_text: Callable[[str | None], str | None]
    render_math_markdown: Callable[[str], object]
    select_plot_render_path: Callable[[list[Path]], Path | None]
    style_notebook_axes: Callable[..., None]
    table_from_records: Callable[..., object]
    unique_in_order: Callable[[object], list[str]]


@dataclass(frozen=True)
class BrowserRenderers:
    compare_pair_payload: Callable[..., dict[str, object]]
    enrich_projection_frame: Callable[[pd.DataFrame, list[dict[str, object]]], pd.DataFrame]
    load_projection_frame: Callable[..., pd.DataFrame]
    load_plot_review_frames: Callable[..., list[pd.DataFrame]]
    render_distance_correlation: Callable[..., object]
    render_plot_asset: Callable[[Path], object]
    render_plot_review_surface: Callable[..., object]
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
    return humanize_plot_title(plot_id)


def _resolved_plot_semantics_payload(
    context,
    *,
    plot_id: str,
    manifest: dict[str, object],
) -> dict[str, object]:
    del manifest
    return resolve_plot_semantics(context, plot_id=plot_id).model_dump(mode="json")


def _runtime_hue_columns(
    *,
    joinable_tables: list[dict[str, object]],
    preferred_hues: list[str],
    row_metadata_hues: list[str],
    configured_hue_kinds: dict[str, object],
    joinable_artifact_suffixes: set[str],
) -> tuple[list[str], dict[str, str]]:
    actual_columns = unique_in_order(
        str(column)
        for row in joinable_tables
        if isinstance(row, dict) and list(row.get("view_ids") or [])
        for column in row.get("columns", [])
        if isinstance(column, str) and include_hue_column(str(column), joinable_artifact_suffixes)
    )
    row_metadata_set = set(row_metadata_hues)
    ordered_candidates = unique_in_order(
        [
            *[column for column in preferred_hues if column in actual_columns or column in row_metadata_set],
            *[column for column in actual_columns if column in configured_hue_kinds],
        ]
    )
    hue_kinds = resolve_runtime_hue_kinds(ordered_candidates, configured_hue_kinds)
    return [column for column in ordered_candidates if column in hue_kinds], hue_kinds


def _candidate_inventory_from_control_plane(
    *,
    controls: dict[str, object],
    catalog: dict[str, object],
) -> list[dict[str, object]]:
    control_rows = controls.get("candidate_inventory")
    if isinstance(control_rows, list):
        rows = [row for row in control_rows if isinstance(row, dict) and row.get("view_id")]
        if rows:
            return rows
    catalog_rows = catalog.get("candidate_inventory")
    if isinstance(catalog_rows, list):
        return [row for row in catalog_rows if isinstance(row, dict) and row.get("view_id")]
    return []


def _matrix_shapes_from_control_plane(
    *,
    candidate_inventory: list[dict[str, object]],
    geometry_rows: list[dict[str, object]],
) -> list[dict[str, int | str]]:
    shapes: list[dict[str, int | str]] = []
    seen: set[str] = set()
    for row in candidate_inventory:
        view_id = str(row.get("view_id") or "").strip()
        rows = row.get("n_rows")
        dims = row.get("n_dims")
        if (
            not view_id
            or view_id in seen
            or str(row.get("modality") or "") != "vector"
            or str(row.get("materialization_status") or "") != "materialized"
            or rows is None
            or dims is None
        ):
            continue
        shapes.append({"view_id": view_id, "rows": int(rows), "dims": int(dims)})
        seen.add(view_id)
    if shapes:
        return shapes
    for row in geometry_rows:
        view_id = str(row.get("view_id") or "").strip()
        rows = row.get("rows")
        dims = row.get("dims")
        if not view_id or view_id in seen or rows is None or dims is None:
            continue
        shapes.append({"view_id": view_id, "rows": int(rows), "dims": int(dims)})
        seen.add(view_id)
    return shapes


def _reference_set_option_label(row: dict[str, object]) -> str:
    configured_label = str(row.get("label") or "").strip()
    if configured_label:
        return configured_label
    return _humanize_plot_id(str(row.get("reference_set_id") or "reference_set"))


def _reference_annotation_options(reference_sets: list[dict[str, object]]) -> dict[str, str]:
    options = {"Off": ""}
    seen_ids: set[str] = set()
    for row in reference_sets:
        if not isinstance(row, dict):
            continue
        reference_set_id = str(row.get("reference_set_id") or "").strip()
        if not reference_set_id or reference_set_id in seen_ids:
            continue
        label = _reference_set_option_label(row)
        if label in options:
            label = f"{label} ({reference_set_id})"
        options[label] = reference_set_id
        seen_ids.add(reference_set_id)
    return options


def _reference_required_columns(reference_sets: list[dict[str, object]]) -> list[str]:
    columns: list[str] = []
    for row in reference_sets:
        if not isinstance(row, dict):
            continue
        for column in [
            row.get("match_column"),
            row.get("label_column"),
            *list(row.get("selector_columns") or []),
        ]:
            text = str(column or "").strip()
            if text and text not in columns:
                columns.append(text)
    return columns


def _live_plot_status_rows(catalog_plots: list[dict[str, object]] | None) -> dict[str, dict[str, object]]:
    return {
        str(row.get("plot_id")): row for row in (catalog_plots or []) if isinstance(row, dict) and row.get("plot_id")
    }


def _projection_grid_render_mode(
    *,
    output_root: Path,
    plot_spec: dict[str, object],
) -> tuple[bool, str | None]:
    del output_root, plot_spec
    return True, None


def _resolve_plot_review_render_mode(
    *,
    output_root: Path,
    plot_spec: dict[str, object],
) -> tuple[bool, str | None]:
    kind = str(plot_spec.get("kind") or "")
    if kind not in _PLOT_REVIEW_LIVE_RENDER_KINDS:
        return False, None
    if kind == "projection_grid":
        return _projection_grid_render_mode(output_root=output_root, plot_spec=plot_spec)
    return True, None


def _markdown_heading_level(line: str) -> int | None:
    hash_count = len(line) - len(line.lstrip("#"))
    if hash_count == 0 or hash_count > 6:
        return None
    if len(line) <= hash_count or line[hash_count] != " ":
        return None
    return hash_count


def _next_heading_at_or_above(lines: list[str], start: int, level: int) -> int:
    for index in range(start + 1, len(lines)):
        heading_level = _markdown_heading_level(lines[index])
        if heading_level is not None and heading_level <= level:
            return index
    return len(lines)


def _parse_deliverable_markdown(markdown: str) -> dict[str, object]:
    lines = markdown.splitlines()
    summary_lines: list[str] = []
    plot_sections: dict[str, dict[str, str]] = {}

    first_h1 = next((index for index, line in enumerate(lines) if line.startswith("# ")), None)
    if first_h1 is not None:
        index = first_h1 + 1
        while index < len(lines):
            line = lines[index]
            heading_level = _markdown_heading_level(line)
            if heading_level is not None and heading_level <= 2:
                break
            summary_lines.append(line)
            index += 1

    heading_indices = [index for index, line in enumerate(lines) if _markdown_heading_level(line) == 3]
    for start in heading_indices:
        end = _next_heading_at_or_above(lines, start, 3)
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


def _extract_plot_details(markdown: str) -> str:
    lines = markdown.splitlines()
    heading_indices = [index for index, line in enumerate(lines) if _markdown_heading_level(line) == 4]
    if not heading_indices:
        return ""

    for start in heading_indices:
        end = _next_heading_at_or_above(lines, start, 4)
        title = lines[start][5:].strip()
        if title.casefold() != "plot details":
            continue
        return "\n".join(lines[start + 1 : end]).strip()
    return ""


def _strip_plot_details(markdown: str) -> str:
    lines = markdown.splitlines()
    heading_indices = [index for index, line in enumerate(lines) if _markdown_heading_level(line) == 4]
    if not heading_indices:
        return markdown.strip()

    kept_blocks: list[str] = []
    cursor = 0
    for start in heading_indices:
        end = _next_heading_at_or_above(lines, start, 4)
        if cursor < start:
            kept_blocks.append("\n".join(lines[cursor:start]).strip())
        title = lines[start][5:].strip()
        if title.casefold() != "plot details":
            kept_blocks.append("\n".join(lines[start:end]).strip())
        cursor = end
    if cursor < len(lines):
        kept_blocks.append("\n".join(lines[cursor:]).strip())
    return "\n\n".join(block for block in kept_blocks if block).strip()


def resolve_plot_doc_block(
    *,
    plot_id: str,
    deliverable_summary: str,
    parsed_markdown: dict[str, object] | None,
) -> dict[str, object]:
    plot_sections = parsed_markdown.get("plot_sections", {}) if isinstance(parsed_markdown, dict) else {}
    summary_markdown = (
        str(parsed_markdown.get("summary_markdown") or "").strip() if isinstance(parsed_markdown, dict) else ""
    )
    plot_entry = plot_sections.get(plot_id) if isinstance(plot_sections, dict) else None
    if isinstance(plot_entry, dict):
        markdown = str(plot_entry.get("markdown") or "").strip()
        plot_details_md = _extract_plot_details(markdown)
        return {
            "title": str(plot_entry.get("title") or _humanize_plot_id(plot_id)),
            "markdown": _strip_plot_details(markdown),
            "plot_details_md": plot_details_md,
            "warning": None,
        }
    fallback_markdown = summary_markdown or deliverable_summary.strip()
    return {
        "title": _humanize_plot_id(plot_id),
        "markdown": fallback_markdown,
        "plot_details_md": "",
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


def _resolve_review_plot_spec(context, *, plot_id: str):
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


def _plot_review_sections(
    context,
    *,
    output_root: Path,
    controls: dict[str, object],
    catalog_plots: list[dict[str, object]] | None = None,
) -> BrowserPlotReview:
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
    live_plot_rows = _live_plot_status_rows(catalog_plots)
    docs_cache: dict[str, dict[str, object]] = {}
    sections: list[dict[str, object]] = []
    current_section: dict[str, object] | None = None

    for plot_id in ordered_plot_ids:
        entry = plot_entries.get(plot_id, {})
        live_entry = live_plot_rows.get(plot_id, {})
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
        manifest_warning: str | None = None
        try:
            manifest = load_json(plot_dir / "manifest.json")
        except Exception as exc:
            manifest = {"status": "error", "stale": False}
            manifest_warning = f"Plot manifest could not be read for `{plot_id}`: {exc}"
        semantics = _resolved_plot_semantics_payload(context, plot_id=plot_id, manifest=manifest)
        output_paths = [
            plot_dir / str(output.get("path"))
            for output in manifest.get("outputs", [])
            if isinstance(output, dict) and output.get("path")
        ]
        render_path = select_plot_render_path(output_paths)
        visibility_tier = str(
            entry.get("visibility_tier") or getattr(context.require_plot(plot_id), "visibility_tier", "primary")
        )
        resolved_spec = _resolve_review_plot_spec(context, plot_id=plot_id)
        plot_spec_payload = resolved_spec.model_dump(mode="json")
        live_render, render_mode_note = _resolve_plot_review_render_mode(
            output_root=output_root,
            plot_spec=plot_spec_payload,
        )

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
                "deliverable_id": deliverable_id,
                "title": str(doc_block.get("title") or _humanize_plot_id(plot_id)),
                "visibility_tier": visibility_tier,
                "render_path": render_path,
                "question": str(semantics.get("question") or "").strip(),
                "decision_role": str(semantics.get("decision_role") or "").strip(),
                "encoding": str(semantics.get("encoding") or "").strip(),
                "scope": str(semantics.get("scope") or "").strip(),
                "guardrails": [str(item).strip() for item in semantics.get("guardrails", []) if str(item).strip()],
                "caption_md": str(semantics.get("caption") or "").strip(),
                "alt_text": str(semantics.get("alt_text") or doc_block.get("title") or plot_id).strip(),
                "study_doc_md": str(doc_block.get("markdown") or "").strip(),
                "plot_details_md": str(doc_block.get("plot_details_md") or "").strip(),
                "preprocessing_md": str(semantics.get("preprocessing_md") or "").strip(),
                "math_md": str(semantics.get("math_md") or "").strip(),
                "rationale_md": str(semantics.get("rationale_md") or "").strip(),
                "limitations_md": str(semantics.get("limitations_md") or "").strip(),
                "failure_modes_md": str(semantics.get("failure_modes_md") or "").strip(),
                "study_doc_warning": doc_block.get("warning"),
                "artifact_warning": manifest_warning,
                "status": (
                    "missing"
                    if render_path is None
                    and not live_render
                    and str(live_entry.get("status") or entry.get("status") or manifest.get("status") or "missing")
                    == "ok"
                    else str(live_entry.get("status") or entry.get("status") or manifest.get("status") or "missing")
                ),
                "stale": bool(
                    live_entry.get("stale")
                    if live_entry.get("stale") is not None
                    else entry.get("stale")
                    if entry.get("stale") is not None
                    else manifest.get("stale")
                ),
                "live_render": live_render,
                "render_mode_note": render_mode_note,
                "plot_spec": plot_spec_payload,
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
    candidate_inventory = _candidate_inventory_from_control_plane(controls=controls, catalog=catalog)
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
    row_metadata_hues = [str(item) for item in geometry_control.get("row_metadata_hues", []) if isinstance(item, str)]
    configured_hue_kinds = geometry_control.get("hue_kinds", {})
    axis_styles = {
        str(column): style
        for column, style in dict(geometry_control.get("axis_styles", {}) or {}).items()
        if isinstance(style, dict)
    }
    reference_labels = [str(item) for item in geometry_control.get("reference_labels", []) if isinstance(item, str)]
    reference_sets = [
        row
        for row in geometry_control.get("reference_sets", [])
        if isinstance(row, dict) and row.get("reference_set_id")
    ]
    candidate_sets = [
        row
        for row in geometry_control.get("candidate_sets", [])
        if isinstance(row, dict) and row.get("candidate_set_id")
    ]

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
    matrix_shapes = _matrix_shapes_from_control_plane(
        candidate_inventory=candidate_inventory,
        geometry_rows=geometry_rows,
    )
    row_count_text = "unknown"
    dimensionality_text = "unknown"
    if matrix_shapes:
        row_count_text = ", ".join(f"{row['view_id']}={row['rows']}" for row in matrix_shapes[:4])
        dimensionality_text = ", ".join(f"{row['view_id']}={row['dims']}" for row in matrix_shapes[:4])
    reference_annotation_options = _reference_annotation_options(reference_sets)
    configured_default_reference_set = str(geometry_control.get("default_reference_set") or "").strip()
    reference_annotation_default = (
        configured_default_reference_set
        if configured_default_reference_set in set(reference_annotation_options.values())
        else ""
    )
    reference_hue_options = {
        "Black stars": "",
        "Reference strength": "promoter_standard__strength_value_numeric",
    }
    reference_hue_columns = [value for value in reference_hue_options.values() if value]
    reference_required_columns = list(
        dict.fromkeys([*_reference_required_columns(reference_sets), *reference_hue_columns])
    )
    global_hue_columns, hue_kinds = _runtime_hue_columns(
        joinable_tables=joinable_tables,
        preferred_hues=preferred_hues,
        row_metadata_hues=row_metadata_hues,
        configured_hue_kinds=configured_hue_kinds,
        joinable_artifact_suffixes=joinable_artifact_suffixes,
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
    load_projection_frame_for_output = partial(load_projection_frame, output_root=output_root)
    render_plot_asset_for_workspace = partial(render_plot_asset, workspace_dir=workspace_dir)
    load_plot_review_frames_for_workspace = partial(
        load_plot_review_frames,
        output_root=output_root,
    )
    render_plot_review_surface_for_workspace = partial(
        render_plot_review_surface,
        output_root=output_root,
        workspace_dir=workspace_dir,
        axis_styles=axis_styles,
    )
    render_projection_grid_for_workspace = partial(
        render_projection_grid,
        output_root=output_root,
        workspace_dir=workspace_dir,
        axis_styles=axis_styles,
    )
    compare_pair_payload_for_output = partial(compare_pair_payload, output_root=output_root)
    plot_review = _plot_review_sections(
        context,
        output_root=output_root,
        controls=controls,
        catalog_plots=plots,
    )

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
            axis_styles=axis_styles,
            candidate_sets=candidate_sets,
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
            row_metadata_hues=row_metadata_hues,
            reference_annotation_default=reference_annotation_default,
            reference_annotation_options=reference_annotation_options,
            reference_hue_columns=reference_hue_columns,
            reference_hue_options=reference_hue_options,
            reference_labels=reference_labels,
            reference_required_columns=reference_required_columns,
            reference_sets=reference_sets,
            selected_hue_default=selected_hue_default,
        ),
        plot_review=plot_review,
        support=BrowserSupport(
            available_hues_for_frames=available_hues_for_frames,
            candidate_hue_columns=candidate_hue_columns,
            category_color_map=partial(category_color_map, axis_styles=axis_styles),
            display_hue_label=partial(display_hue_label, axis_styles=axis_styles),
            json=json,
            key_value_table=key_value_table,
            load_json=load_json,
            load_table=load_table,
            mo=mo,
            notebook_theme=notebook_theme,
            option_key_for_value=option_key_for_value,
            labeled_options=labeled_options,
            pd=pd,
            read_text=read_text,
            render_math_markdown=render_math_markdown,
            select_plot_render_path=select_plot_render_path,
            style_notebook_axes=style_notebook_axes,
            table_from_records=table_from_records,
            unique_in_order=unique_in_order,
        ),
        renderers=BrowserRenderers(
            compare_pair_payload=compare_pair_payload_for_output,
            enrich_projection_frame=enrich_projection_frame_for_output,
            load_projection_frame=load_projection_frame_for_output,
            load_plot_review_frames=load_plot_review_frames_for_workspace,
            render_distance_correlation=render_distance_correlation,
            render_plot_asset=render_plot_asset_for_workspace,
            render_plot_review_surface=render_plot_review_surface_for_workspace,
            render_projection_grid=render_projection_grid_for_workspace,
            render_rowwise_distribution=render_rowwise_distribution,
        ),
    )
