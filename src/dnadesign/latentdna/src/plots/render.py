"""
Artifact-driven plotting helpers for latentdna.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..contracts.plot import SUPPORTED_PLOT_KINDS, ResolvedPlotSpec, metric_panel_uses_square_axes
from ..contracts.plot_semantics import PlotSemantics
from ..labels import humanize_candidate
from ..visual_style import (
    DEFAULT_PLOT_PNG_DPI,
    GRID_COLOR,
    PANEL_BACKGROUND_COLOR,
    PLOT_FONT_FAMILY,
    PLOT_LABEL_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    PUBLICATION_PALETTE,
    SPINE_COLOR,
    TEXT_COLOR,
    ZERO_LINE_COLOR,
    categorical_color_map,
    humanize_display_text,
    ordered_categories,
    scatter_style,
    wrap_plot_title,
)
from ..workspaces.loader import WorkspaceContext

_SHAPE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]


def _pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = PLOT_FONT_FAMILY
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    return plt


def _category_color_map(row_groups: list[list[dict]], column: str | None) -> tuple[dict[str, str], list[str]]:
    if column is None:
        return {}, []
    flattened = [row for rows in row_groups for row in rows]
    if flattened and column not in flattened[0]:
        raise ContractViolationError(f"plot color column is missing: {column!r}")
    categories = ordered_categories(str(row[column]) for row in flattened)
    color_map = categorical_color_map(categories)
    return color_map, categories


def _color_series(
    rows: list[dict],
    column: str | None,
    *,
    color_map: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    if column is None:
        return [PUBLICATION_PALETTE[0]] * len(rows), []
    if rows and column not in rows[0]:
        raise ContractViolationError(f"plot color column is missing: {column!r}")
    resolved_map = color_map or _category_color_map([rows], column)[0]
    categories = ordered_categories(resolved_map)
    return [resolved_map[str(row[column])] for row in rows], categories


def _shape_marker_map(row_groups: list[list[dict]], column: str | None) -> tuple[dict[str, str], list[str]]:
    if column is None:
        return {}, []
    flattened = [row for rows in row_groups for row in rows]
    if flattened and column not in flattened[0]:
        raise ContractViolationError(f"plot shape column is missing: {column!r}")
    categories = sorted({str(row[column]) for row in flattened})
    shape_map = {name: _SHAPE_MARKERS[index % len(_SHAPE_MARKERS)] for index, name in enumerate(categories)}
    return shape_map, categories


def _effective_shape_column(spec: ResolvedPlotSpec) -> str | None:
    if spec.hue_options and spec.kind in {
        "projection_scatter",
        "projection_grid",
        "xy_scatter",
        "xy_scatter_grid",
        "paired_xy_scatter_grid",
    }:
        return None
    return spec.shape_column


def _table_rows(table_path: Path) -> list[dict]:
    return pq.read_table(table_path).to_pylist()


def _numeric_columns(table: pa.Table) -> list[str]:
    numeric: list[str] = []
    for field in table.schema:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            numeric.append(field.name)
    return numeric


def _secondary_numeric_column(table: pa.Table, *, primary: str) -> str:
    for candidate in _numeric_columns(table):
        if candidate != primary:
            return candidate
    raise ContractViolationError(
        f"plot rendering requires at least two numeric columns when {primary!r} is used as the first axis"
    )


def _candidate_row_label(
    row: dict[str, object],
    *,
    fallback_column: str,
    include_family: bool = True,
) -> str:
    candidate_fields = {
        key: str(row.get(key) or "").strip()
        for key in ("candidate_model", "candidate_scope", "candidate_family")
        if str(row.get(key) or "").strip()
    }
    if not include_family:
        candidate_fields.pop("candidate_family", None)
    if candidate_fields:
        return humanize_candidate(candidate_fields)
    return humanize_display_text(str(row.get(fallback_column) or ""))


def _derived_panel_label(identifier: str) -> str:
    candidate_key = str(identifier or "")
    for prefix in (
        "context_geometry_metrics_",
        "wildtype_reference_margins_",
        "synthetic_centroid_margins_",
        "tradeoff_",
        "pca_",
    ):
        if candidate_key.startswith(prefix):
            candidate_key = candidate_key[len(prefix) :]
            break
    candidate_key = candidate_key.replace("_anchor_to_full_context", "")
    return humanize_candidate(candidate_key)


def _apply_axes_style(ax: Any, *, grid: bool, square: bool = False) -> None:
    ax.set_facecolor(PANEL_BACKGROUND_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(colors=TEXT_COLOR, labelsize=PLOT_TICK_FONT_SIZE, length=4.5, width=0.8, direction="out")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_fontsize(PLOT_TITLE_FONT_SIZE)
    ax.title.set_fontweight("semibold")
    ax.title.set_fontfamily(PLOT_FONT_FAMILY)
    ax.margins(x=0.04, y=0.05)
    if square:
        ax.set_box_aspect(1)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.75, alpha=0.58)
        ax.set_axisbelow(True)


def _style_legend(legend: Any) -> None:
    if legend is None:
        return
    title = legend.get_title()
    if title is not None:
        title.set_visible(False)
    for text in legend.get_texts():
        text.set_fontsize(PLOT_LEGEND_FONT_SIZE)
        text.set_color(TEXT_COLOR)
        text.set_fontfamily(PLOT_FONT_FAMILY)


def _legend_handles(plt: Any, categories: list[str], color_map: dict[str, str]) -> list[Any]:
    return [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=7.5,
            color=color_map[category],
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=humanize_display_text(category),
        )
        for category in categories
    ]


def _shape_legend_handles(plt: Any, categories: list[str], shape_map: dict[str, str]) -> list[Any]:
    return [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker=shape_map[category],
            markersize=7,
            markerfacecolor="#7A8794",
            markeredgecolor="#111111",
            color="#7A8794",
            label=humanize_display_text(category),
        )
        for category in categories
    ]


def _scatter_points(
    ax: Any,
    rows: list[dict[str, object]],
    *,
    resolved_x: str,
    resolved_y: str,
    color_column: str | None,
    color_map: dict[str, str],
    shape_column: str | None,
    shape_map: dict[str, str],
    point_size: float,
    alpha: float,
    rasterized: bool = False,
    edgecolors: str = "white",
    linewidths: float = 0.25,
) -> None:
    if shape_column is None:
        colors, _ = _color_series(rows, color_column, color_map=color_map if color_map else None)
        ax.scatter(
            [float(row[resolved_x]) for row in rows],
            [float(row[resolved_y]) for row in rows],
            c=colors,
            s=point_size,
            alpha=alpha,
            edgecolors=edgecolors,
            linewidths=linewidths,
            rasterized=rasterized,
        )
        return
    if rows and shape_column not in rows[0]:
        raise ContractViolationError(f"plot shape column is missing: {shape_column!r}")
    for shape_category, marker in shape_map.items():
        group_rows = [row for row in rows if str(row[shape_column]) == shape_category]
        if not group_rows:
            continue
        colors, _ = _color_series(group_rows, color_column, color_map=color_map if color_map else None)
        ax.scatter(
            [float(row[resolved_x]) for row in group_rows],
            [float(row[resolved_y]) for row in group_rows],
            c=colors,
            s=point_size,
            alpha=alpha,
            marker=marker,
            edgecolors=edgecolors,
            linewidths=linewidths,
            rasterized=rasterized,
        )


def _add_axis_legends(
    ax: Any,
    plt: Any,
    *,
    color_categories: list[str],
    color_map: dict[str, str],
    color_title: str | None,
    shape_categories: list[str],
    shape_map: dict[str, str],
    shape_title: str | None,
) -> None:
    color_legend = None
    if color_categories and color_title is not None:
        color_legend = ax.legend(
            handles=_legend_handles(plt, color_categories, color_map),
            frameon=False,
            loc="upper left",
        )
        _style_legend(color_legend)
    if shape_categories and shape_title is not None:
        if color_legend is not None:
            ax.add_artist(color_legend)
        shape_legend = ax.legend(
            handles=_shape_legend_handles(plt, shape_categories, shape_map),
            frameon=False,
            loc="lower right",
        )
        _style_legend(shape_legend)


def _add_figure_legends(
    fig: Any,
    plt: Any,
    *,
    color_categories: list[str],
    color_map: dict[str, str],
    color_title: str | None,
    shape_categories: list[str],
    shape_map: dict[str, str],
    shape_title: str | None,
) -> float:
    legend_specs: list[list[Any]] = []
    if color_categories and color_title is not None:
        legend_specs.append(_legend_handles(plt, color_categories, color_map))
    if shape_categories and shape_title is not None:
        legend_specs.append(_shape_legend_handles(plt, shape_categories, shape_map))
    if not legend_specs:
        return 0.0

    legend_y = 0.012
    for handles in legend_specs:
        legend = fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=max(1, len(handles)),
            frameon=False,
            borderaxespad=0.0,
            columnspacing=0.95,
            handletextpad=0.45,
        )
        _style_legend(legend)
        legend_y += 0.055
    return min(max(legend_y + 0.014, 0.1), 0.18)


def _selected_label_rows(rows: list[dict], *, label_column: str | None, label_values: list[str]) -> list[dict]:
    if label_column is None or not label_values:
        return []
    if rows and label_column not in rows[0]:
        raise ContractViolationError(f"plot label column is missing: {label_column!r}")
    selected = {str(value) for value in label_values}
    return [row for row in rows if str(row[label_column]) in selected]


def _add_zero_reference_lines(ax: Any, *, x_values: list[float], y_values: list[float]) -> None:
    if x_values and min(x_values) < 0.0 < max(x_values):
        ax.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9, zorder=0)
    if y_values and min(y_values) < 0.0 < max(y_values):
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9, zorder=0)


def _annotation_offsets() -> list[tuple[float, float]]:
    return [
        (10.0, 10.0),
        (10.0, -18.0),
        (-72.0, 10.0),
        (-72.0, -18.0),
        (18.0, 24.0),
        (-80.0, 24.0),
        (18.0, -32.0),
        (-80.0, -32.0),
        (40.0, 0.0),
        (-94.0, 0.0),
    ]


def _draw_annotation_callouts(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    label_texts: list[str],
    marker_colors: list[str],
) -> None:
    if not rows:
        return
    x_values = [float(row[resolved_x]) for row in rows]
    y_values = [float(row[resolved_y]) for row in rows]
    offsets = _annotation_offsets()
    placed: list[tuple[float, float]] = []
    ax.scatter(
        x_values,
        y_values,
        c=marker_colors,
        s=128,
        marker="*",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    for row, label_text in sorted(
        zip(rows, label_texts, strict=True),
        key=lambda item: item[1].casefold(),
    ):
        point_x = float(row[resolved_x])
        point_y = float(row[resolved_y])
        display_x, display_y = ax.transData.transform((point_x, point_y))
        target_offset = offsets[0]
        for offset_x, offset_y in offsets:
            candidate_x = display_x + offset_x
            candidate_y = display_y + offset_y
            if all(
                abs(candidate_x - placed_x) > 54.0 or abs(candidate_y - placed_y) > 24.0
                for placed_x, placed_y in placed
            ):
                target_offset = (offset_x, offset_y)
                break
        placed.append((display_x + target_offset[0], display_y + target_offset[1]))
        annotation = ax.annotate(
            label_text,
            xy=(point_x, point_y),
            xytext=target_offset,
            textcoords="offset points",
            fontsize=9.5,
            fontweight="semibold",
            color=TEXT_COLOR,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.94},
            arrowprops={"arrowstyle": "-", "color": SPINE_COLOR, "linewidth": 0.9},
            zorder=6,
        )
        annotation.set_clip_on(True)
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_clip_on(True)


def _resolve_annotation_rows(
    context: WorkspaceContext,
    rows: list[dict],
    *,
    spec: ResolvedPlotSpec,
) -> tuple[list[dict], str | None, list[str], dict[str, object]]:
    if spec.annotation is None:
        selected_rows = _selected_label_rows(rows, label_column=spec.label_column, label_values=spec.label_values)
        return (
            selected_rows,
            spec.label_column,
            list(spec.label_values),
            {
                "reference_set": None,
                "expected_ids": list(spec.label_values),
                "matched_ids": (
                    [str(row[spec.label_column]) for row in selected_rows] if spec.label_column is not None else []
                ),
                "complete": True,
            },
        )

    reference_set = context.config.reference_sets[spec.annotation.reference_set]
    match_column = reference_set.match_column
    label_column = reference_set.label_column or match_column
    expected_ids = [str(value) for value in reference_set.ids]
    missing_columns = [column for column in (match_column, label_column) if rows and column not in rows[0]]
    if missing_columns:
        return (
            [],
            None,
            expected_ids,
            {
                "reference_set": spec.annotation.reference_set,
                "match_column": match_column,
                "label_column": label_column,
                "expected_ids": expected_ids,
                "matched_ids": [],
                "complete": False,
                "error": "missing_reference_columns",
                "missing_columns": missing_columns,
            },
        )
    selected_by_id = {str(row[match_column]): row for row in rows if str(row[match_column]) in set(expected_ids)}
    missing_ids = [value for value in expected_ids if value not in selected_by_id]
    if missing_ids and spec.annotation.missing_policy == "fail":
        raise ContractViolationError(
            f"reference_set {spec.annotation.reference_set!r} is missing required ids: {missing_ids}"
        )
    selected = [selected_by_id[value] for value in expected_ids if value in selected_by_id]
    return (
        selected,
        label_column,
        expected_ids,
        {
            "reference_set": spec.annotation.reference_set,
            "match_column": match_column,
            "label_column": label_column,
            "expected_ids": expected_ids,
            "matched_ids": [value for value in expected_ids if value in selected_by_id],
            "complete": not missing_ids,
        },
    )


def _annotation_label_text(
    context: WorkspaceContext,
    *,
    spec: ResolvedPlotSpec,
    row: dict[str, object],
    resolved_label_column: str,
) -> str:
    if spec.annotation is None:
        return str(row[resolved_label_column])
    reference_set = context.config.reference_sets[spec.annotation.reference_set]
    display_labels = dict(getattr(reference_set, "display_labels", {}) or {})
    match_column = reference_set.match_column
    match_value = str(row.get(match_column, ""))
    return str(display_labels.get(match_value, row[resolved_label_column]))


def _table_artifact_path(context: WorkspaceContext, spec: ResolvedPlotSpec) -> tuple[str, str, Path]:
    candidates = [
        (
            "scalar_table",
            spec.scalar_id,
            context.output_root / "scalars" / spec.scalar_id / "table.parquet" if spec.scalar_id is not None else None,
        ),
        (
            "distance_set",
            spec.distance_id,
            context.output_root / "distances" / spec.distance_id / "table.parquet"
            if spec.distance_id is not None
            else None,
        ),
        (
            "enrichment_set",
            spec.enrichment_id,
            context.output_root / "enrichments" / spec.enrichment_id / "table.parquet"
            if spec.enrichment_id is not None
            else None,
        ),
        (
            "agreement_set",
            spec.agreement_id,
            context.output_root / "agreements" / spec.agreement_id / "table.parquet"
            if spec.agreement_id is not None
            else None,
        ),
    ]
    selected = [(kind, artifact_id, path) for kind, artifact_id, path in candidates if artifact_id is not None]
    if len(selected) != 1:
        raise ContractViolationError(
            "plot rendering requires exactly one table-backed artifact input for this plot kind"
        )
    artifact_kind, artifact_id, artifact_path = selected[0]
    assert artifact_path is not None
    if not artifact_path.exists():
        raise MissingArtifactError(f"{artifact_kind} artifact is missing for plot rendering: {artifact_id}")
    return artifact_kind, str(artifact_id), artifact_path


def _ordered_numeric_axes(
    table: pa.Table,
    *,
    x_column: str | None,
    y_column: str | None,
    value_column: str | None,
) -> tuple[str, str]:
    numeric_columns = _numeric_columns(table)
    if len(numeric_columns) < 2:
        raise ContractViolationError("scatter rendering requires at least two numeric columns")
    resolved_x = x_column or value_column or numeric_columns[0]
    if resolved_x not in numeric_columns:
        raise ContractViolationError(f"scatter x column is missing or non-numeric: {resolved_x!r}")
    resolved_y = y_column or _secondary_numeric_column(table, primary=resolved_x)
    if resolved_y not in numeric_columns:
        raise ContractViolationError(f"scatter y column is missing or non-numeric: {resolved_y!r}")
    return resolved_x, resolved_y


def _shared_row_key_columns(left_rows: list[dict], right_rows: list[dict]) -> list[str]:
    if not left_rows or not right_rows:
        raise ContractViolationError("correspondence_heatmap requires non-empty cluster assignments")
    left_columns = set(left_rows[0]) - {"cluster_label"}
    right_columns = set(right_rows[0]) - {"cluster_label"}
    preferred_order = ["id", "subject_id", "record_key", "subject_key", "context_id", "context_key"]
    shared = [column for column in preferred_order if column in left_columns and column in right_columns]
    if shared:
        return shared
    shared = sorted(left_columns.intersection(right_columns))
    if not shared:
        raise ContractViolationError("correspondence_heatmap requires at least one shared row key column")
    return shared


def _agreement_summary_metrics(summary: dict[str, object]) -> list[tuple[str, float]]:
    metrics: list[tuple[str, float]] = []
    knn_summary = summary.get("knn_overlap")
    if isinstance(knn_summary, dict) and "mean_overlap_fraction" in knn_summary:
        metrics.append(("kNN overlap", float(knn_summary["mean_overlap_fraction"])))
    cluster_summary = summary.get("cluster_agreement")
    if isinstance(cluster_summary, dict):
        if "adjusted_rand_index" in cluster_summary:
            metrics.append(("ARI", float(cluster_summary["adjusted_rand_index"])))
        if "normalized_mutual_information" in cluster_summary:
            metrics.append(("NMI", float(cluster_summary["normalized_mutual_information"])))
    landmark_summary = summary.get("landmark_neighbor_overlap")
    if isinstance(landmark_summary, dict) and "mean_jaccard_overlap" in landmark_summary:
        metrics.append(("Landmark Jaccard", float(landmark_summary["mean_jaccard_overlap"])))
    return metrics


def _panel_grid_dimensions(panel_count: int) -> tuple[int, int]:
    if panel_count <= 1:
        return 1, 1
    if panel_count == 4:
        return 2, 2
    if panel_count == 8:
        return 2, 4
    columns = min(4, max(1, int(math.ceil(math.sqrt(panel_count)))))
    rows = int(np.ceil(panel_count / columns))
    return rows, columns


def _grid_figure_size(panel_count: int, *, square_panels: bool) -> tuple[float, float]:
    if panel_count <= 1:
        return (5.15, 5.0 if square_panels else 4.7)
    rows, columns = _panel_grid_dimensions(panel_count)
    panel_width = 4.15 if columns >= 4 else 4.3
    panel_height = 4.35 if square_panels else 4.05
    return (panel_width * columns, panel_height * rows)


def _scatter_axis_label(
    rows: list[dict[str, object]],
    *,
    resolved_column: str,
    display_column: str,
) -> str:
    labels = {str(row.get(display_column) or "").strip() for row in rows if str(row.get(display_column) or "").strip()}
    if len(labels) == 1:
        return humanize_display_text(next(iter(labels)))
    return humanize_display_text(resolved_column)


def _wrapped_tick_label(value: object, *, width: int = 16) -> str:
    return wrap_plot_title(humanize_display_text(str(value)), width=width)


def _wrapped_axis_label(value: object, *, width: int = 22) -> str:
    return wrap_plot_title(humanize_display_text(str(value)), width=width)


def _render_xy_panel(
    ax: Any,
    rows: list[dict[str, object]],
    *,
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    resolved_x: str,
    resolved_y: str,
    panel_title: str,
    color_map: dict[str, str],
    shape_map: dict[str, str],
) -> dict[str, object]:
    x_values = [float(row[resolved_x]) for row in rows]
    y_values = [float(row[resolved_y]) for row in rows]
    render_mode = spec.render_mode or "points"
    colors, _ = _color_series(rows, spec.color_column, color_map=color_map if color_map else None)
    if render_mode == "hexbin":
        ax.hexbin(x_values, y_values, gridsize=max(12, min(48, int(np.sqrt(len(rows))) * 2)), cmap="cividis")
    elif render_mode == "density_contour":
        bins = max(10, min(30, int(np.sqrt(len(rows))) * 2))
        histogram, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=bins)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax.contour(x_centers, y_centers, histogram.T, levels=4, cmap="cividis")
        density_style = scatter_style(len(rows))
        ax.scatter(
            x_values,
            y_values,
            c=colors,
            s=density_style.point_size,
            alpha=max(0.16, density_style.alpha * 0.72),
            edgecolors="none",
            rasterized=True,
        )
    else:
        point_style = scatter_style(len(rows))
        _scatter_points(
            ax,
            rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            color_column=spec.color_column,
            color_map=color_map,
            shape_column=_effective_shape_column(spec),
            shape_map=shape_map,
            point_size=point_style.point_size,
            alpha=point_style.alpha,
            rasterized=point_style.rasterized,
            edgecolors=point_style.edgecolors,
            linewidths=point_style.linewidths,
        )
    _add_zero_reference_lines(ax, x_values=x_values, y_values=y_values)
    ax.set_xlabel(
        _wrapped_axis_label(_scatter_axis_label(rows, resolved_column=resolved_x, display_column="x_display_name"))
    )
    ax.set_ylabel(
        _wrapped_axis_label(_scatter_axis_label(rows, resolved_column=resolved_y, display_column="y_display_name"))
    )
    ax.set_title(wrap_plot_title(panel_title, width=24), pad=8)
    _apply_axes_style(ax, grid=True, square=True)
    selected_rows, resolved_label_column, _, annotation_state = _resolve_annotation_rows(context, rows, spec=spec)
    if selected_rows and resolved_label_column is not None:
        label_mode = (
            "label_and_highlight"
            if spec.annotation is None
            else context.config.reference_sets[spec.annotation.reference_set].label_mode
        )
        if label_mode == "label_and_highlight":
            highlight_colors = (
                ["#111111"] * len(selected_rows)
                if spec.annotation is not None
                else _color_series(
                    selected_rows,
                    spec.color_column,
                    color_map=color_map if color_map else None,
                )[0]
            )
            _draw_annotation_callouts(
                ax,
                rows=selected_rows,
                resolved_x=resolved_x,
                resolved_y=resolved_y,
                label_texts=[
                    _annotation_label_text(
                        context,
                        spec=spec,
                        row=row,
                        resolved_label_column=resolved_label_column,
                    )
                    for row in selected_rows
                ],
                marker_colors=highlight_colors,
            )
    return annotation_state


def _render_agreement_summary_panel(
    ax: Any,
    *,
    metrics: list[tuple[str, float]],
    panel_title: str,
) -> None:
    labels = [label for label, _ in metrics]
    values = [value for _, value in metrics]
    bars = ax.bar(
        labels,
        values,
        color=[PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)] for index in range(len(labels))],
    )
    low = min(0.0, min(values))
    high = max(1.0, max(values))
    if low == high:
        high = low + 1.0
    padding = max((high - low) * 0.15, 0.05)
    ax.set_ylim(low - padding, high + padding)
    ax.axhline(0.0, color=SPINE_COLOR, linewidth=0.8)
    ax.set_ylabel("Score")
    ax.set_title(wrap_plot_title(panel_title, width=24), pad=8)
    _apply_axes_style(ax, grid=True)
    for bar, value in zip(bars, values, strict=True):
        va = "bottom" if value >= 0 else "top"
        offset = 0.02 if value >= 0 else -0.02
        ax.text(bar.get_x() + (bar.get_width() / 2.0), value + offset, f"{value:.2f}", ha="center", va=va)


def _render_distribution_panel(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    metric_column: str,
    color_column: str | None,
    render_mode: str,
    panel_title: str,
    square: bool = False,
) -> None:
    values = np.asarray([float(row[metric_column]) for row in rows], dtype=np.float32)
    bin_count = max(5, min(30, int(np.sqrt(values.size)) + 1))
    if render_mode == "ecdf":
        if color_column is None:
            ordered = np.sort(values)
            cumulative = np.arange(1, len(ordered) + 1, dtype=np.float32) / float(len(ordered))
            ax.step(ordered, cumulative, where="post", color=PUBLICATION_PALETTE[0], linewidth=2.0)
        else:
            if rows and color_column not in rows[0]:
                raise ContractViolationError(f"distribution color column is missing: {color_column!r}")
            categories = sorted({str(row[color_column]) for row in rows})
            for index, category in enumerate(categories):
                category_values = np.sort(
                    np.asarray(
                        [float(row[metric_column]) for row in rows if str(row[color_column]) == category],
                        dtype=np.float32,
                    )
                )
                cumulative = np.arange(1, len(category_values) + 1, dtype=np.float32) / float(len(category_values))
                ax.step(
                    category_values,
                    cumulative,
                    where="post",
                    label=humanize_display_text(category),
                    color=PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)],
                    linewidth=2.0,
                )
            legend = ax.legend(frameon=False)
            _style_legend(legend)
        ax.set_ylabel("ECDF")
    elif render_mode == "violin_box":
        if color_column is None:
            violin = ax.violinplot([values], showmeans=False, showmedians=False)
            for body in violin["bodies"]:
                body.set_facecolor(PUBLICATION_PALETTE[0])
                body.set_alpha(0.5)
            ax.boxplot([values], widths=0.18)
            ax.set_xticks([1], [humanize_display_text(metric_column)])
        else:
            if rows and color_column not in rows[0]:
                raise ContractViolationError(f"distribution color column is missing: {color_column!r}")
            categories = sorted({str(row[color_column]) for row in rows})
            grouped_values = [
                np.asarray(
                    [float(row[metric_column]) for row in rows if str(row[color_column]) == category],
                    dtype=np.float32,
                )
                for category in categories
            ]
            violin = ax.violinplot(grouped_values, showmeans=False, showmedians=False)
            for index, body in enumerate(violin["bodies"]):
                body.set_facecolor(PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)])
                body.set_alpha(0.45)
            ax.boxplot(grouped_values, widths=0.18)
            ax.set_xticks(
                range(1, len(categories) + 1),
                [humanize_display_text(category) for category in categories],
                rotation=25,
                ha="right",
            )
        ax.set_ylabel(_wrapped_axis_label(humanize_display_text(metric_column), width=18))
    else:
        if color_column is None:
            ax.hist(values, bins=bin_count, color=PUBLICATION_PALETTE[0], edgecolor="white", alpha=0.9)
        else:
            if rows and color_column not in rows[0]:
                raise ContractViolationError(f"distribution color column is missing: {color_column!r}")
            categories = sorted({str(row[color_column]) for row in rows})
            for index, category in enumerate(categories):
                category_values = np.asarray(
                    [float(row[metric_column]) for row in rows if str(row[color_column]) == category],
                    dtype=np.float32,
                )
                ax.hist(
                    category_values,
                    bins=bin_count,
                    alpha=0.55,
                    label=humanize_display_text(category),
                    color=PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)],
                    edgecolor="white",
                )
            legend = ax.legend(frameon=False)
            _style_legend(legend)
        ax.set_ylabel("Count")
    ax.set_xlabel(_wrapped_axis_label(metric_column, width=20))
    ax.set_title(wrap_plot_title(panel_title, width=24), pad=8)
    _apply_axes_style(ax, grid=True, square=square)


def _metric_axis_label(
    *,
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
) -> str:
    base_source = str(spec.value_label or spec.value_column or "metric value")
    if base_source.strip().casefold() == "metric value" and spec.panel_column is not None:
        panel_labels = {
            str(row.get(spec.panel_column) or "").strip()
            for row in rows
            if str(row.get(spec.panel_column) or "").strip()
        }
        if len(panel_labels) == 1:
            base_source = next(iter(panel_labels))
    base_label = humanize_display_text(base_source)
    if spec.unit_column is None:
        return base_label
    units = {
        str(row.get(spec.unit_column) or "").strip() for row in rows if str(row.get(spec.unit_column) or "").strip()
    }
    if len(units) != 1:
        return base_label
    unit = next(iter(units))
    return f"{base_label} ({humanize_display_text(unit)})"


def _sorted_metric_rows(rows: list[dict[str, object]], *, spec: ResolvedPlotSpec) -> list[dict[str, object]]:
    label_column = spec.label_column or spec.column_column
    if label_column is None:
        raise ContractViolationError("metric_panel_grid rendering requires a label column")
    sort_rule = spec.sort_rule or "panel_direction"
    if sort_rule == "label_asc":
        return sorted(rows, key=lambda row: str(row.get(label_column) or "").casefold())
    if sort_rule == "value_asc":
        return sorted(rows, key=lambda row: float(row[spec.value_column]))
    if sort_rule == "value_desc":
        return sorted(rows, key=lambda row: float(row[spec.value_column]), reverse=True)
    direction = ""
    if spec.direction_column is not None and rows:
        direction = str(rows[0].get(spec.direction_column) or "").strip().lower()
    reverse = direction != "lower_is_better"
    return sorted(rows, key=lambda row: float(row[spec.value_column]), reverse=reverse)


def _render_metric_panel(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
    panel_title: str,
    color_map: dict[str, str],
    square: bool = False,
) -> None:
    if spec.value_column is None:
        raise ContractViolationError("metric_panel_grid rendering requires value_column")
    label_column = spec.label_column or spec.column_column
    if label_column is None:
        raise ContractViolationError("metric_panel_grid rendering requires label_column")
    ordered_rows = _sorted_metric_rows(rows, spec=spec)
    include_family = not (spec.color_column == "candidate_family")
    label_width = 11 if include_family else 10
    labels = [
        _wrapped_tick_label(
            _candidate_row_label(
                row,
                fallback_column=label_column,
                include_family=include_family,
            ),
            width=label_width,
        )
        for row in ordered_rows
    ]
    values = np.asarray([float(row[spec.value_column]) for row in ordered_rows], dtype=np.float64)
    grouped_family_bars = (
        spec.color_column == "candidate_family"
        and all(str(row.get("candidate_model") or "").strip() for row in ordered_rows)
        and all(str(row.get("candidate_scope") or "").strip() for row in ordered_rows)
    )
    if spec.color_column is not None:
        if ordered_rows and spec.color_column not in ordered_rows[0]:
            raise ContractViolationError(f"metric_panel_grid color column is missing: {spec.color_column!r}")
        bar_colors = [color_map[str(row[spec.color_column])] for row in ordered_rows]
    else:
        bar_colors = [PUBLICATION_PALETTE[0]] * len(ordered_rows)
    ci_enabled = spec.ci_lower_column is not None and spec.ci_upper_column is not None and ordered_rows

    bar_value_pairs: list[tuple[Any, float]] = []
    errorbar_specs: list[tuple[float, float, float, float]] = []
    if grouped_family_bars:
        family_order = ordered_categories([str(row["candidate_family"]) for row in ordered_rows])
        group_keys = list(
            dict.fromkeys(
                (
                    str(row["candidate_model"]),
                    str(row["candidate_scope"]),
                )
                for row in ordered_rows
            )
        )
        group_labels = [
            _wrapped_tick_label(
                humanize_candidate({"candidate_model": model, "candidate_scope": scope}),
                width=10,
            )
            for model, scope in group_keys
        ]
        group_positions = np.arange(len(group_keys), dtype=float)
        group_width = min(0.78, 0.32 * max(len(family_order), 1))
        bar_width = group_width / max(len(family_order), 1)
        offsets = np.linspace(
            -(group_width / 2.0) + (bar_width / 2.0),
            (group_width / 2.0) - (bar_width / 2.0),
            max(len(family_order), 1),
        )
        for family, offset in zip(family_order, offsets, strict=False):
            family_rows = {
                (str(row["candidate_model"]), str(row["candidate_scope"])): row
                for row in ordered_rows
                if str(row["candidate_family"]) == family
            }
            family_positions: list[float] = []
            family_values: list[float] = []
            for group_position, group_key in zip(group_positions, group_keys, strict=False):
                row = family_rows.get(group_key)
                if row is None:
                    continue
                family_positions.append(float(group_position + offset))
                family_values.append(float(row[spec.value_column]))
                family_bars = ax.bar(
                    family_positions,
                    family_values,
                    width=bar_width * 0.9,
                    color=color_map[family],
                    edgecolor="white",
                    linewidth=0.6,
                    alpha=0.92,
                )
            bar_value_pairs.extend(zip(family_bars, family_values, strict=True))
            if ci_enabled:
                family_ci_rows = [family_rows[key] for key in group_keys if key in family_rows]
                for bar, row in zip(family_bars, family_ci_rows, strict=False):
                    lower = row.get(spec.ci_lower_column)
                    upper = row.get(spec.ci_upper_column)
                    if lower is None or upper is None:
                        continue
                    errorbar_specs.append(
                        (
                            float(bar.get_x() + (bar.get_width() / 2.0)),
                            float(row[spec.value_column]),
                            float(lower),
                            float(upper),
                        )
                    )
        ax.set_xticks(group_positions, group_labels)
    else:
        positions = np.arange(len(ordered_rows), dtype=float)
        bars = ax.bar(
            positions,
            values,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        bar_value_pairs.extend(zip(bars, values, strict=True))
        if ci_enabled:
            for position, row in zip(positions, ordered_rows, strict=True):
                lower = row.get(spec.ci_lower_column)
                upper = row.get(spec.ci_upper_column)
                if lower is None or upper is None:
                    continue
                errorbar_specs.append(
                    (
                        float(position),
                        float(row[spec.value_column]),
                        float(lower),
                        float(upper),
                    )
                )
        ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", pad=6)
    if spec.reference_line is not None:
        ax.axhline(float(spec.reference_line), color=SPINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
    if values.size and float(values.min()) < 0.0 < float(values.max()):
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
    if errorbar_specs:
        xs = np.asarray([item[0] for item in errorbar_specs], dtype=np.float64)
        ys = np.asarray([item[1] for item in errorbar_specs], dtype=np.float64)
        lowers = np.asarray([max(item[1] - item[2], 0.0) for item in errorbar_specs], dtype=np.float64)
        uppers = np.asarray([max(item[3] - item[1], 0.0) for item in errorbar_specs], dtype=np.float64)
        ax.errorbar(
            xs,
            ys,
            yerr=np.vstack([lowers, uppers]),
            fmt="none",
            ecolor=SPINE_COLOR,
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.85,
        )
    ax.set_xlabel("")
    ax.set_ylabel(_wrapped_axis_label(_metric_axis_label(rows=ordered_rows, spec=spec), width=20))
    ax.set_title(wrap_plot_title(panel_title, width=24), pad=8)
    _apply_axes_style(ax, grid=True, square=square)
    span = float(values.max() - values.min()) if values.size else 0.0
    offset = max(span * 0.04, 0.02) if span > 0 else 0.02
    low = min(0.0, float(values.min())) if values.size else 0.0
    high = max(0.0, float(values.max())) if values.size else 1.0
    padding = max((high - low) * 0.16, 0.06)
    ax.set_ylim(low - padding, high + padding)
    for bar, value in bar_value_pairs:
        y_text = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            y_text,
            f"{value:.3g}",
            va=va,
            ha="center",
            fontsize=9,
            color=TEXT_COLOR,
        )


def _render_curve_panel(
    ax: Any,
    *,
    reducer_id: str,
    summary: dict[str, object],
    panel_title: str,
    square: bool = False,
    show_legend: bool = True,
) -> None:
    ratios = summary.get("explained_variance_ratio")
    if not isinstance(ratios, list) or not ratios:
        raise ContractViolationError(f"curve rendering requires explained_variance_ratio for {reducer_id}")
    explained = np.asarray([float(value) for value in ratios], dtype=np.float32)
    cumulative = np.cumsum(explained)
    components = np.arange(1, len(explained) + 1, dtype=np.int64)
    ax.plot(components, explained, marker="o", linewidth=1.8, color=PUBLICATION_PALETTE[0], label="Explained")
    ax.plot(
        components,
        cumulative,
        marker="s",
        linewidth=1.8,
        color=PUBLICATION_PALETTE[2],
        label="Cumulative",
    )
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance ratio")
    ax.set_ylim(0.0, max(1.0, float(cumulative.max()) * 1.05))
    ax.set_title(wrap_plot_title(panel_title, width=24), pad=8)
    if show_legend:
        legend = ax.legend(frameon=False)
        _style_legend(legend)
    _apply_axes_style(ax, grid=True, square=square)


def _inject_svg_accessibility(output_path: Path, *, semantics: PlotSemantics) -> None:
    import xml.etree.ElementTree as ET

    tree = ET.parse(output_path)
    root = tree.getroot()
    title = ET.Element("title")
    title.text = semantics.question
    desc = ET.Element("desc")
    desc.text = semantics.alt_text
    root.insert(0, desc)
    root.insert(0, title)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def _write_plot_outputs(fig: Any, artifact_dir: Path, *, formats: list[str], semantics: PlotSemantics) -> list[str]:
    outputs: list[str] = []
    for format_name in formats:
        if format_name not in {"svg", "pdf", "png"}:
            raise ContractViolationError(f"unsupported plot output format: {format_name!r}")
        output_path = artifact_dir / f"plot.{format_name}"
        save_kwargs = {"bbox_inches": "tight", "facecolor": "white", "edgecolor": "none"}
        if format_name == "png":
            fig.savefig(output_path, dpi=DEFAULT_PLOT_PNG_DPI, **save_kwargs)
        else:
            fig.savefig(output_path, **save_kwargs)
        if format_name == "svg":
            _inject_svg_accessibility(output_path, semantics=semantics)
        outputs.append(output_path.as_posix())
    return outputs


def render_plot_artifact(
    context: WorkspaceContext,
    *,
    spec: ResolvedPlotSpec,
    output_dir: Path,
    semantics: PlotSemantics,
) -> tuple[Path, list[str], dict[str, object]]:
    if spec.kind not in SUPPORTED_PLOT_KINDS:
        raise ContractViolationError(f"unsupported plot kind: {spec.kind}")
    if spec.kind in {"projection_scatter", "projection_grid"} and not spec.projection_ids:
        raise ContractViolationError("plot rendering requires at least one projection artifact")
    if spec.kind == "heatmap" and spec.enrichment_id is None and spec.scalar_id is None:
        raise ContractViolationError("heatmap rendering requires an enrichment or scalar artifact")
    if spec.kind == "distance_scatter" and spec.distance_id is None:
        raise ContractViolationError("distance_scatter rendering requires a distance artifact")
    if spec.kind == "xy_scatter" and spec.scalar_id is None and spec.distance_id is None:
        raise ContractViolationError("xy_scatter rendering requires a scalar or distance artifact")
    if spec.kind in {"xy_scatter_grid", "paired_xy_scatter_grid"} and not spec.scalar_ids:
        raise ContractViolationError("xy_scatter_grid rendering requires at least one scalar artifact")
    if spec.kind == "categorical_count" and spec.scalar_id is None:
        raise ContractViolationError("categorical_count rendering requires a scalar artifact")
    if spec.kind == "metric_panel_grid" and spec.scalar_id is None:
        raise ContractViolationError("metric_panel_grid rendering requires a scalar artifact")
    if spec.kind == "curve" and spec.reducer_id is None:
        raise ContractViolationError("curve rendering requires a reducer artifact")
    if spec.kind == "distribution_grid" and not spec.scalar_ids:
        raise ContractViolationError("distribution_grid rendering requires at least one scalar artifact")
    if spec.kind == "curve_grid" and not spec.reducer_ids:
        raise ContractViolationError("curve_grid rendering requires at least one reducer artifact")
    if spec.kind == "correspondence_heatmap" and (spec.left_cluster_id is None or spec.right_cluster_id is None):
        raise ContractViolationError("correspondence_heatmap rendering requires two cluster artifacts")
    if spec.kind == "agreement_summary" and spec.agreement_id is None:
        raise ContractViolationError("agreement_summary rendering requires an agreement artifact")
    if spec.kind == "agreement_summary_grid" and not spec.agreement_ids:
        raise ContractViolationError("agreement_summary_grid rendering requires at least one agreement artifact")

    plt = _pyplot()

    plot_metadata: dict[str, object] = {}

    if spec.kind == "heatmap":
        from matplotlib import colors as mcolors

        if spec.enrichment_id is not None:
            table_path = context.output_root / "enrichments" / spec.enrichment_id / "table.parquet"
            if not table_path.exists():
                raise MissingArtifactError(
                    f"enrichment artifact is missing for heatmap rendering: {spec.enrichment_id}"
                )
            row_column = spec.row_column or "landmark_id"
            column_column = spec.column_column or "cohort_value"
            metric_column = spec.value_column or "enrichment_delta"
        else:
            assert spec.scalar_id is not None
            table_path = context.output_root / "scalars" / spec.scalar_id / "table.parquet"
            if not table_path.exists():
                raise MissingArtifactError(f"scalar artifact is missing for heatmap rendering: {spec.scalar_id}")
            if spec.row_column is None or spec.column_column is None:
                raise ContractViolationError("scalar-backed heatmap rendering requires row_column and column_column")
            row_column = spec.row_column
            column_column = spec.column_column
            metric_column = spec.value_column or "metric_value"
        rows = _table_rows(table_path)
        if not rows:
            raise ContractViolationError("heatmap rendering requires at least one input row")
        if metric_column not in rows[0]:
            raise ContractViolationError(f"heatmap value column is missing from table: {metric_column!r}")
        if row_column not in rows[0]:
            raise ContractViolationError(f"heatmap row column is missing: {row_column!r}")
        if column_column not in rows[0]:
            raise ContractViolationError(f"heatmap column column is missing: {column_column!r}")
        column_values = sorted({str(row[column_column]) for row in rows})
        row_values = list(dict.fromkeys(str(row[row_column]) for row in rows))
        row_index = {row_value: index for index, row_value in enumerate(row_values)}
        column_index = {column_value: index for index, column_value in enumerate(column_values)}
        grid = np.full((len(row_values), len(column_values)), np.nan, dtype=np.float32)
        for row in rows:
            grid[
                row_index[str(row[row_column])],
                column_index[str(row[column_column])],
            ] = float(row[metric_column])

        finite = np.asarray(grid[np.isfinite(grid)], dtype=np.float32)
        if finite.size == 0:
            raise ContractViolationError("heatmap rendering requires at least one finite value")
        max_abs = max(float(np.max(np.abs(finite))), 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

        fig, ax = plt.subplots(figsize=(2 + 1.5 * len(column_values), 1.5 + 1.2 * len(row_values)))
        image = ax.imshow(grid, cmap="PuOr", norm=norm, aspect="auto")
        ax.set_xticks(
            range(len(column_values)),
            [humanize_display_text(value) for value in column_values],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(range(len(row_values)), [humanize_display_text(value) for value in row_values])
        ax.set_xlabel(humanize_display_text(column_column))
        ax.set_ylabel(humanize_display_text(row_column))
        ax.set_title(wrap_plot_title(spec.plot_id, width=24), pad=8)
        for row_index_value in range(len(row_values)):
            for column_index_value in range(len(column_values)):
                value = grid[row_index_value, column_index_value]
                if not np.isfinite(value):
                    label = "NA"
                    text_color = TEXT_COLOR
                else:
                    label = f"{value:.2f}"
                    text_color = "white" if abs(value) > max_abs * 0.45 else TEXT_COLOR
                ax.text(
                    column_index_value,
                    row_index_value,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )
        colorbar = fig.colorbar(image, ax=ax, label=metric_column)
        colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
        colorbar.set_label(metric_column, fontsize=11, color=TEXT_COLOR)
        _apply_axes_style(ax, grid=False)
    elif spec.kind in {"distance_scatter", "xy_scatter"}:
        _, _, table_path = _table_artifact_path(context, spec)
        table = pq.read_table(table_path)
        resolved_x, resolved_y = _ordered_numeric_axes(
            table,
            x_column=spec.x_column,
            y_column=spec.y_column,
            value_column=spec.value_column,
        )

        rows = _table_rows(table_path)
        fig, ax = plt.subplots(figsize=_grid_figure_size(1, square_panels=True))
        color_map, categories = _category_color_map([rows], spec.color_column)
        effective_shape_column = _effective_shape_column(spec)
        shape_map, shape_categories = _shape_marker_map([rows], effective_shape_column)
        annotation_state = _render_xy_panel(
            ax,
            rows,
            context=context,
            spec=spec,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            panel_title=spec.plot_id,
            color_map=color_map,
            shape_map=shape_map,
        )
        plot_metadata["reference_panels"] = {
            spec.scalar_id or spec.distance_id or spec.plot_id: annotation_state,
        }
        if (spec.render_mode or "points") == "points":
            _add_axis_legends(
                ax,
                plt,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_shape_column,
            )
    elif spec.kind in {"xy_scatter_grid", "paired_xy_scatter_grid"}:
        scalar_tables: list[tuple[str, list[dict[str, object]], str, str]] = []
        for scalar_id in spec.scalar_ids:
            table_path = context.output_root / "scalars" / scalar_id / "table.parquet"
            if not table_path.exists():
                raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {scalar_id}")
            table = pq.read_table(table_path)
            resolved_x, resolved_y = _ordered_numeric_axes(
                table,
                x_column=spec.x_column,
                y_column=spec.y_column,
                value_column=spec.value_column,
            )
            scalar_tables.append((scalar_id, _table_rows(table_path), resolved_x, resolved_y))
        rows_count, columns = _panel_grid_dimensions(len(scalar_tables))
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=_grid_figure_size(len(scalar_tables), square_panels=True),
            squeeze=False,
        )
        color_map, categories = _category_color_map([rows for _, rows, _, _ in scalar_tables], spec.color_column)
        effective_shape_column = _effective_shape_column(spec)
        shape_map, shape_categories = _shape_marker_map(
            [rows for _, rows, _, _ in scalar_tables],
            effective_shape_column,
        )
        titles = spec.panel_titles or [scalar_id for scalar_id, _, _, _ in scalar_tables]
        for axis in axes.ravel()[len(scalar_tables) :]:
            axis.axis("off")
        for axis, (scalar_id, rows, resolved_x, resolved_y), panel_title in zip(
            axes.ravel(),
            scalar_tables,
            titles,
            strict=False,
        ):
            annotation_state = _render_xy_panel(
                axis,
                rows,
                context=context,
                spec=spec,
                resolved_x=resolved_x,
                resolved_y=resolved_y,
                panel_title=panel_title,
                color_map=color_map,
                shape_map=shape_map,
            )
            plot_metadata.setdefault("reference_panels", {})[scalar_id] = annotation_state
        grid_legend_bottom_margin = 0.0
        if (spec.render_mode or "points") == "points":
            grid_legend_bottom_margin = _add_figure_legends(
                fig,
                plt,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_shape_column,
            )
    elif spec.kind == "categorical_count":
        assert spec.scalar_id is not None
        table_path = context.output_root / "scalars" / spec.scalar_id / "table.parquet"
        if not table_path.exists():
            raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {spec.scalar_id}")
        rows = _table_rows(table_path)
        if not rows:
            raise ContractViolationError("categorical_count rendering requires at least one row")
        required_columns = [spec.row_column, spec.column_column, spec.value_column]
        missing_columns = [column for column in required_columns if column is not None and column not in rows[0]]
        if missing_columns:
            raise ContractViolationError(f"categorical_count columns are missing: {missing_columns}")
        panel_values = (
            list(dict.fromkeys(str(row[spec.panel_column]) for row in rows))
            if spec.panel_column is not None
            else [None]
        )
        if len(panel_values) <= 2:
            rows_count, columns = len(panel_values), 1
        else:
            rows_count, columns = _panel_grid_dimensions(len(panel_values))
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(6.6 * columns, 5.8 * rows_count),
            squeeze=False,
        )
        color_map, categories = _category_color_map([rows], spec.color_column)
        for axis in axes.ravel()[len(panel_values) :]:
            axis.axis("off")
        for axis, panel_value in zip(axes.ravel(), panel_values, strict=False):
            panel_rows = (
                [row for row in rows if str(row[spec.panel_column]) == panel_value]
                if panel_value is not None and spec.panel_column is not None
                else rows
            )
            if panel_rows and "order" in panel_rows[0]:
                panel_rows = sorted(panel_rows, key=lambda row: float(row.get("order", 0)))
            if spec.color_column is not None:
                if spec.color_column not in panel_rows[0]:
                    raise ContractViolationError(f"categorical_count color column is missing: {spec.color_column!r}")
                bar_colors = [color_map[str(row[spec.color_column])] for row in panel_rows]
            else:
                bar_colors = [PUBLICATION_PALETTE[0]] * len(panel_rows)
            values = [float(row[spec.value_column]) for row in panel_rows]
            y_positions = np.arange(len(panel_rows), dtype=float)
            axis.barh(
                y_positions,
                values,
                color=bar_colors,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
            )
            axis.set_yticks(
                y_positions,
                [_wrapped_tick_label(row[spec.column_column], width=22) for row in panel_rows],
            )
            axis.invert_yaxis()
            show_as_percent = spec.value_column in {"fraction", "percent"} or all(
                0.0 <= value <= 1.0 for value in values
            )
            axis.set_xlabel(
                "Percent of N" if show_as_percent else humanize_display_text(spec.value_column or "row_count")
            )
            axis.set_ylabel("")
            denominator_values = {
                int(float(row["denominator"])) for row in panel_rows if row.get("denominator") is not None
            }
            panel_title = wrap_plot_title(str(panel_value) if panel_value is not None else spec.plot_id, width=24)
            if len(denominator_values) == 1:
                panel_title = f"{panel_title}\nN = {next(iter(denominator_values)):,}"
            axis.set_title(
                panel_title,
                pad=8,
            )
            max_value = max(values, default=0.0)
            axis.set_xlim(0, max_value * 1.12 if max_value > 0 else 1.0)
            if show_as_percent:
                from matplotlib.ticker import PercentFormatter

                axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0 if max_value <= 1.0 else 100.0))
            for y_position, value in zip(y_positions, values, strict=False):
                count_text = None
                if panel_rows[int(y_position)].get("count") is not None:
                    count_text = f"{int(float(panel_rows[int(y_position)]['count'])):,}"
                percent_text = None
                if panel_rows[int(y_position)].get("percent") is not None:
                    percent_text = f"{float(panel_rows[int(y_position)]['percent']):.1f}%"
                label_parts = [part for part in [count_text, percent_text] if part is not None]
                axis.text(
                    value + (max_value * 0.02 if max_value > 0 else 0.05),
                    y_position,
                    " | ".join(label_parts)
                    if label_parts
                    else (f"{int(value):,}" if float(value).is_integer() else f"{value:.1f}"),
                    va="center",
                    ha="left",
                    fontsize=9.5,
                    color=TEXT_COLOR,
                )
            _apply_axes_style(axis, grid=True)
        grid_legend_bottom_margin = 0.0
        if categories and spec.color_column is not None:
            grid_legend_bottom_margin = _add_figure_legends(
                fig,
                plt,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=[],
                shape_map={},
                shape_title=None,
            )
    elif spec.kind == "metric_panel_grid":
        assert spec.scalar_id is not None
        table_path = context.output_root / "scalars" / spec.scalar_id / "table.parquet"
        if not table_path.exists():
            raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {spec.scalar_id}")
        rows = _table_rows(table_path)
        if not rows:
            raise ContractViolationError("metric_panel_grid rendering requires at least one row")
        resolved_value_column = spec.value_column
        if resolved_value_column is not None and resolved_value_column not in rows[0]:
            if "metric_value" in rows[0]:
                resolved_value_column = "metric_value"
            elif "row_count" in rows[0]:
                resolved_value_column = "row_count"
        required_columns = [
            spec.row_column,
            spec.panel_column,
            spec.column_column,
            spec.label_column,
            resolved_value_column,
        ]
        missing_columns = [column for column in required_columns if column is not None and column not in rows[0]]
        if missing_columns:
            raise ContractViolationError(f"metric_panel_grid columns are missing: {missing_columns}")
        resolved_spec = spec.model_copy(update={"value_column": resolved_value_column})
        panel_values = list(dict.fromkeys(str(row[spec.row_column]) for row in rows))
        rows_count, columns = _panel_grid_dimensions(len(panel_values))
        square_metric_panels = metric_panel_uses_square_axes(spec.plot_id)
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=_grid_figure_size(len(panel_values), square_panels=square_metric_panels),
            squeeze=False,
        )
        color_map, categories = _category_color_map([rows], spec.color_column)
        panel_rows_by_value = {
            panel_value: [row for row in rows if str(row[spec.row_column]) == panel_value]
            for panel_value in panel_values
        }
        for axis in axes.ravel()[len(panel_values) :]:
            axis.axis("off")
        for axis, panel_value in zip(axes.ravel(), panel_values, strict=False):
            panel_rows = panel_rows_by_value[panel_value]
            panel_title = str(panel_rows[0][spec.panel_column]) if spec.panel_column is not None else panel_value
            _render_metric_panel(
                axis,
                rows=panel_rows,
                spec=resolved_spec,
                panel_title=panel_title,
                color_map=color_map,
                square=square_metric_panels,
            )
        plot_metadata["metric_columns"] = panel_values
        grid_legend_bottom_margin = 0.0
        if categories and spec.color_column is not None:
            grid_legend_bottom_margin = _add_figure_legends(
                fig,
                plt,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=[],
                shape_map={},
                shape_title=None,
            )
    elif spec.kind == "distribution":
        artifact_kind, artifact_id, table_path = _table_artifact_path(context, spec)
        table = pq.read_table(table_path)
        numeric_columns = _numeric_columns(table)
        if not numeric_columns:
            raise ContractViolationError(
                f"distribution rendering requires at least one numeric column in {artifact_kind}"
            )
        metric_column = spec.value_column or numeric_columns[0]
        if metric_column not in numeric_columns:
            raise ContractViolationError(f"distribution value column is missing or non-numeric: {metric_column!r}")

        rows = _table_rows(table_path)
        if not rows:
            raise ContractViolationError("distribution rendering requires at least one row")
        square_distribution_panel = spec.plot_id == "context_delta_distributions"
        fig, ax = plt.subplots(figsize=(5.4, 5.2 if square_distribution_panel else 4.8))
        render_mode = spec.render_mode or "histogram"
        _render_distribution_panel(
            ax,
            rows=rows,
            metric_column=metric_column,
            color_column=spec.color_column,
            render_mode=render_mode,
            panel_title=artifact_id,
            square=square_distribution_panel,
        )
    elif spec.kind == "distribution_grid":
        scalar_tables: list[tuple[str, list[dict[str, object]], str]] = []
        configured_metric_columns = list(spec.metric_columns or [])
        for index, scalar_id in enumerate(spec.scalar_ids):
            table_path = context.output_root / "scalars" / scalar_id / "table.parquet"
            if not table_path.exists():
                raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {scalar_id}")
            table = pq.read_table(table_path)
            numeric_columns = _numeric_columns(table)
            if not numeric_columns:
                raise ContractViolationError(
                    f"distribution_grid rendering requires at least one numeric column in scalar {scalar_id}"
                )
            rows = _table_rows(table_path)
            if not rows:
                raise ContractViolationError("distribution_grid rendering requires at least one row per panel")
            if configured_metric_columns:
                for metric_column in configured_metric_columns:
                    if metric_column not in numeric_columns:
                        raise ContractViolationError(
                            f"distribution_grid value column is missing or non-numeric: {metric_column!r}"
                        )
                    scalar_tables.append((scalar_id, rows, metric_column))
                continue
            metric_column = (
                spec.value_columns[index]
                if index < len(spec.value_columns)
                else spec.value_column or numeric_columns[0]
            )
            if metric_column not in numeric_columns:
                raise ContractViolationError(
                    f"distribution_grid value column is missing or non-numeric: {metric_column!r}"
                )
            scalar_tables.append((scalar_id, rows, metric_column))
        rows_count, columns = _panel_grid_dimensions(len(scalar_tables))
        square_distribution_panels = spec.plot_id == "context_delta_distributions"
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=_grid_figure_size(len(scalar_tables), square_panels=square_distribution_panels),
            squeeze=False,
        )
        titles = spec.panel_titles or [
            (
                f"{_derived_panel_label(scalar_id)} · {humanize_display_text(metric_column)}"
                if _derived_panel_label(scalar_id)
                else humanize_display_text(metric_column)
            )
            for scalar_id, _, metric_column in scalar_tables
        ]
        for axis in axes.ravel()[len(scalar_tables) :]:
            axis.axis("off")
        for axis, (_, rows, metric_column), panel_title in zip(axes.ravel(), scalar_tables, titles, strict=False):
            _render_distribution_panel(
                axis,
                rows=rows,
                metric_column=metric_column,
                color_column=spec.color_column,
                render_mode=spec.render_mode or "histogram",
                panel_title=panel_title,
                square=square_distribution_panels,
            )
        if configured_metric_columns:
            plot_metadata["metric_columns"] = configured_metric_columns
    elif spec.kind == "curve":
        reducer_path = context.output_root / "reducers" / spec.reducer_id / "summary.json"
        if not reducer_path.exists():
            raise MissingArtifactError(f"reducer artifact is missing for curve rendering: {spec.reducer_id}")
        summary = json.loads(reducer_path.read_text(encoding="utf-8"))
        square_curve_panel = spec.plot_id == "representation_scree_diagnostic"
        fig, ax = plt.subplots(figsize=(5.5, 5.3 if square_curve_panel else 4.7))
        _render_curve_panel(
            ax,
            reducer_id=str(spec.reducer_id),
            summary=summary,
            panel_title=spec.plot_id,
            square=square_curve_panel,
        )
    elif spec.kind == "curve_grid":
        reducer_summaries: list[tuple[str, dict[str, object]]] = []
        for reducer_id in spec.reducer_ids:
            reducer_path = context.output_root / "reducers" / reducer_id / "summary.json"
            if not reducer_path.exists():
                raise MissingArtifactError(f"reducer artifact is missing for curve rendering: {reducer_id}")
            reducer_summaries.append((reducer_id, json.loads(reducer_path.read_text(encoding="utf-8"))))
        rows_count, columns = _panel_grid_dimensions(len(reducer_summaries))
        square_curve_panels = spec.plot_id == "representation_scree_diagnostic"
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=_grid_figure_size(len(reducer_summaries), square_panels=square_curve_panels),
            squeeze=False,
        )
        titles = spec.panel_titles or [reducer_id for reducer_id, _ in reducer_summaries]
        for axis in axes.ravel()[len(reducer_summaries) :]:
            axis.axis("off")
        for axis, (reducer_id, summary), panel_title in zip(
            axes.ravel(),
            reducer_summaries,
            titles,
            strict=False,
        ):
            _render_curve_panel(
                axis,
                reducer_id=reducer_id,
                summary=summary,
                panel_title=panel_title,
                square=square_curve_panels,
                show_legend=False,
            )
        legend = fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linewidth=1.8,
                    color=PUBLICATION_PALETTE[0],
                    label="Explained variance ratio",
                ),
                plt.Line2D(
                    [],
                    [],
                    marker="s",
                    linewidth=1.8,
                    color=PUBLICATION_PALETTE[2],
                    label="Cumulative variance ratio",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.1,
            handletextpad=0.5,
        )
        _style_legend(legend)
        grid_legend_bottom_margin = 0.11
    elif spec.kind == "correspondence_heatmap":
        left_path = context.output_root / "clusters" / spec.left_cluster_id / "assignments.parquet"
        right_path = context.output_root / "clusters" / spec.right_cluster_id / "assignments.parquet"
        if not left_path.exists():
            raise MissingArtifactError(
                f"cluster artifact is missing for correspondence rendering: {spec.left_cluster_id}"
            )
        if not right_path.exists():
            raise MissingArtifactError(
                f"cluster artifact is missing for correspondence rendering: {spec.right_cluster_id}"
            )
        left_rows = _table_rows(left_path)
        right_rows = _table_rows(right_path)
        key_columns = _shared_row_key_columns(left_rows, right_rows)
        left_by_key: dict[tuple[object, ...], int] = {}
        right_by_key: dict[tuple[object, ...], int] = {}
        for row in left_rows:
            key = tuple(row[column] for column in key_columns)
            left_by_key[key] = int(row["cluster_label"])
        for row in right_rows:
            key = tuple(row[column] for column in key_columns)
            right_by_key[key] = int(row["cluster_label"])
        shared_keys = sorted(set(left_by_key).intersection(right_by_key))
        if not shared_keys:
            raise ContractViolationError("correspondence_heatmap found no aligned support between cluster assignments")
        left_labels = sorted({left_by_key[key] for key in shared_keys})
        right_labels = sorted({right_by_key[key] for key in shared_keys})
        left_index = {label: index for index, label in enumerate(left_labels)}
        right_index = {label: index for index, label in enumerate(right_labels)}
        grid = np.zeros((len(left_labels), len(right_labels)), dtype=np.float32)
        for key in shared_keys:
            grid[left_index[left_by_key[key]], right_index[right_by_key[key]]] += 1.0
        fig, ax = plt.subplots(figsize=(2 + 1.2 * len(right_labels), 1.8 + 1.1 * len(left_labels)))
        image = ax.imshow(grid, cmap="cividis", aspect="auto")
        ax.set_xticks(
            range(len(right_labels)),
            [humanize_display_text(str(label)) for label in right_labels],
            rotation=25,
            ha="right",
        )
        ax.set_yticks(range(len(left_labels)), [humanize_display_text(str(label)) for label in left_labels])
        ax.set_xlabel(humanize_display_text(spec.right_cluster_id))
        ax.set_ylabel(humanize_display_text(spec.left_cluster_id))
        ax.set_title(wrap_plot_title(spec.plot_id, width=24), pad=8)
        for row_index in range(len(left_labels)):
            for column_index in range(len(right_labels)):
                ax.text(
                    column_index,
                    row_index,
                    f"{int(grid[row_index, column_index])}",
                    ha="center",
                    va="center",
                    color="white" if grid[row_index, column_index] > (grid.max() * 0.45) else TEXT_COLOR,
                    fontsize=10,
                )
        colorbar = fig.colorbar(image, ax=ax, label="Overlap count")
        colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
        colorbar.set_label("Overlap count", fontsize=11, color=TEXT_COLOR)
        _apply_axes_style(ax, grid=False)
    elif spec.kind == "agreement_summary":
        summary_path = context.output_root / "agreements" / spec.agreement_id / "summary.json"
        if not summary_path.exists():
            raise MissingArtifactError(f"agreement artifact is missing for plot rendering: {spec.agreement_id}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = _agreement_summary_metrics(summary)
        if not metrics:
            raise ContractViolationError(
                f"agreement_summary rendering found no plottable metrics for {spec.agreement_id}"
            )
        fig, ax = plt.subplots(figsize=(2 + 1.6 * len(metrics), 4.5))
        _render_agreement_summary_panel(ax, metrics=metrics, panel_title=spec.plot_id)
    elif spec.kind == "agreement_summary_grid":
        agreement_summaries: list[tuple[str, list[tuple[str, float]]]] = []
        for agreement_id in spec.agreement_ids:
            summary_path = context.output_root / "agreements" / agreement_id / "summary.json"
            if not summary_path.exists():
                raise MissingArtifactError(f"agreement artifact is missing for plot rendering: {agreement_id}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = _agreement_summary_metrics(summary)
            if not metrics:
                raise ContractViolationError(
                    f"agreement_summary_grid rendering found no plottable metrics for {agreement_id}"
                )
            agreement_summaries.append((agreement_id, metrics))
        rows_count, columns = _panel_grid_dimensions(len(agreement_summaries))
        fig, axes = plt.subplots(rows_count, columns, figsize=(6 * columns, 4.5 * rows_count), squeeze=False)
        titles = spec.panel_titles or [agreement_id for agreement_id, _ in agreement_summaries]
        for axis in axes.ravel()[len(agreement_summaries) :]:
            axis.axis("off")
        for axis, (_, metrics), panel_title in zip(
            axes.ravel(),
            agreement_summaries,
            titles,
            strict=False,
        ):
            _render_agreement_summary_panel(axis, metrics=metrics, panel_title=panel_title)
    else:
        projection_tables = []
        for projection_id in spec.projection_ids:
            projection_path = context.output_root / "projections" / projection_id / "coords.parquet"
            if not projection_path.exists():
                raise MissingArtifactError(f"projection artifact is missing for plot rendering: {projection_id}")
            projection_tables.append(_table_rows(projection_path))
        if spec.kind == "projection_scatter":
            rows = projection_tables[0]
            color_map, categories = _category_color_map([rows], spec.color_column)
            effective_shape_column = _effective_shape_column(spec)
            shape_map, shape_categories = _shape_marker_map([rows], effective_shape_column)
            fig, ax = plt.subplots(figsize=_grid_figure_size(1, square_panels=True))
            point_style = scatter_style(len(rows))
            _scatter_points(
                ax,
                rows,
                resolved_x="x",
                resolved_y="y",
                color_column=spec.color_column,
                color_map=color_map,
                shape_column=effective_shape_column,
                shape_map=shape_map,
                point_size=point_style.point_size,
                alpha=point_style.alpha,
                rasterized=point_style.rasterized,
                edgecolors=point_style.edgecolors,
                linewidths=point_style.linewidths,
            )
            ax.set_xlabel("Projection 1")
            ax.set_ylabel("Projection 2")
            ax.set_title(wrap_plot_title(spec.projection_ids[0], width=28), pad=8)
            _apply_axes_style(ax, grid=True, square=True)
            selected_rows, resolved_label_column, _, annotation_state = _resolve_annotation_rows(
                context,
                rows,
                spec=spec,
            )
            if selected_rows and resolved_label_column is not None:
                label_mode = (
                    "label_and_highlight"
                    if spec.annotation is None
                    else context.config.reference_sets[spec.annotation.reference_set].label_mode
                )
                if label_mode == "label_and_highlight":
                    highlight_colors = (
                        ["#111111"] * len(selected_rows)
                        if spec.annotation is not None
                        else _color_series(
                            selected_rows,
                            spec.color_column,
                            color_map=color_map if color_map else None,
                        )[0]
                    )
                    _draw_annotation_callouts(
                        ax,
                        rows=selected_rows,
                        resolved_x="x",
                        resolved_y="y",
                        label_texts=[
                            _annotation_label_text(
                                context,
                                spec=spec,
                                row=row,
                                resolved_label_column=resolved_label_column,
                            )
                            for row in selected_rows
                        ],
                        marker_colors=highlight_colors,
                    )
            plot_metadata["reference_panels"] = {spec.projection_ids[0]: annotation_state}
            _add_axis_legends(
                ax,
                plt,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_shape_column,
            )
        else:
            columns = min(2, max(1, len(projection_tables)))
            rows_count = int(np.ceil(len(projection_tables) / columns))
            fig, axes = plt.subplots(
                rows_count,
                columns,
                figsize=_grid_figure_size(len(projection_tables), square_panels=True),
                squeeze=False,
            )
            color_map, categories = _category_color_map(projection_tables, spec.color_column)
            effective_shape_column = _effective_shape_column(spec)
            shape_map, shape_categories = _shape_marker_map(projection_tables, effective_shape_column)
            titles = spec.panel_titles or list(spec.projection_ids)
            for axis in axes.ravel()[len(projection_tables) :]:
                axis.axis("off")
            for axis, projection_rows, projection_id, panel_title in zip(
                axes.ravel(),
                projection_tables,
                spec.projection_ids,
                titles,
                strict=False,
            ):
                point_style = scatter_style(len(projection_rows))
                _scatter_points(
                    axis,
                    projection_rows,
                    resolved_x="x",
                    resolved_y="y",
                    color_column=spec.color_column,
                    color_map=color_map,
                    shape_column=effective_shape_column,
                    shape_map=shape_map,
                    point_size=point_style.point_size,
                    alpha=point_style.alpha,
                    rasterized=point_style.rasterized,
                    edgecolors=point_style.edgecolors,
                    linewidths=point_style.linewidths,
                )
                axis.set_title(wrap_plot_title(panel_title, width=24), pad=8)
                axis.set_xlabel("Projection 1")
                axis.set_ylabel("Projection 2")
                _apply_axes_style(axis, grid=True, square=True)
                selected_rows, resolved_label_column, _, annotation_state = _resolve_annotation_rows(
                    context,
                    projection_rows,
                    spec=spec,
                )
                if selected_rows and resolved_label_column is not None:
                    label_mode = (
                        "label_and_highlight"
                        if spec.annotation is None
                        else context.config.reference_sets[spec.annotation.reference_set].label_mode
                    )
                    if label_mode == "label_and_highlight":
                        highlight_colors = (
                            ["#111111"] * len(selected_rows)
                            if spec.annotation is not None
                            else _color_series(
                                selected_rows,
                                spec.color_column,
                                color_map=color_map if color_map else None,
                            )[0]
                        )
                        _draw_annotation_callouts(
                            axis,
                            rows=selected_rows,
                            resolved_x="x",
                            resolved_y="y",
                            label_texts=[
                                _annotation_label_text(
                                    context,
                                    spec=spec,
                                    row=row,
                                    resolved_label_column=resolved_label_column,
                                )
                                for row in selected_rows
                            ],
                            marker_colors=highlight_colors,
                        )
                plot_metadata.setdefault("reference_panels", {})[projection_id] = annotation_state
            grid_legend_bottom_margin = _add_figure_legends(
                fig,
                plt,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_shape_column,
            )
    grid_legend_bottom_margin = float(locals().get("grid_legend_bottom_margin", 0.0))
    if (
        spec.kind
        in {
            "projection_grid",
            "xy_scatter_grid",
            "paired_xy_scatter_grid",
            "categorical_count",
            "metric_panel_grid",
            "curve_grid",
        }
        and grid_legend_bottom_margin > 0.0
    ):
        fig.tight_layout(rect=(0.0, grid_legend_bottom_margin, 1.0, 0.995), pad=0.95, h_pad=1.4, w_pad=0.95)
    else:
        fig.tight_layout(pad=0.95, h_pad=1.4, w_pad=0.95)

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        outputs = _write_plot_outputs(
            fig,
            output_dir,
            formats=context.config.defaults.plot_formats,
            semantics=semantics,
        )
    finally:
        plt.close(fig)
    if spec.annotation is not None:
        panel_states = list((plot_metadata.get("reference_panels") or {}).values())
        plot_metadata["reference_set_complete"] = bool(panel_states) and all(
            bool(panel.get("complete")) for panel in panel_states
        )
    return output_dir, outputs, plot_metadata
