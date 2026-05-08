"""
Artifact-driven plotting helpers for latentdna.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..annotation_layout import choose_annotation_placement
from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..contracts.plot import SUPPORTED_PLOT_KINDS, ResolvedPlotSpec, metric_panel_uses_square_axes
from ..contracts.plot_semantics import PlotSemantics
from ..labels import humanize_candidate
from ..metadata_axes import (
    AxisStyle,
    axis_color_map,
    axis_display_text,
    axis_style_map_from_config,
    legend_categories,
    normalize_axis_category,
    ordered_categories_for_axis,
)
from ..reference_sets import resolve_reference_set_rows
from ..visual_style import (
    ANNOTATION_LABEL_BOX_ALPHA,
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
    compact_candidate_title,
    humanize_display_text,
    legend_layout,
    reference_annotation_label,
    scatter_style,
    wrap_plot_title,
)
from ..workspaces.loader import WorkspaceContext

_SHAPE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
_SINGLE_ROW_PANEL_PLOT_IDS = frozenset(
    {
        "balanced_design_family_margin_gallery",
        "design_centroid_margin_gallery",
        "representation_scree_diagnostic",
        "appendix_umap_gallery",
    }
)


def _pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = PLOT_FONT_FAMILY
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    return plt


def _compact_repeated_alpha_prefix(parts: list[str]) -> str | None:
    if len(parts) < 2:
        return None
    first_match = re.fullmatch(r"([A-Za-z_ -]+)([0-9A-Za-z_.-]+)", parts[0])
    if first_match is None:
        return None
    prefix = first_match.group(1)
    suffixes = [first_match.group(2)]
    for part in parts[1:]:
        match = re.fullmatch(r"([A-Za-z_ -]+)([0-9A-Za-z_.-]+)", part)
        if match is None or match.group(1) != prefix:
            return None
        suffixes.append(match.group(2))
    return prefix + "+".join(suffixes)


def _category_key(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, list | tuple | set):
        values = sorted(value, key=lambda part: str(part)) if isinstance(value, set) else value
        parts = [" ".join(str(part or "").split()) for part in values]
        parts = [part for part in parts if part]
        if not parts:
            return "None"
        compact = _compact_repeated_alpha_prefix(parts)
        return compact or "+".join(parts)
    return " ".join(str(value).split()) or "None"


def _hue_option_type(spec: ResolvedPlotSpec, column: str | None) -> str | None:
    if column is None:
        return None
    for option in spec.hue_options:
        if option.column == column:
            return option.type
    return None


def _hue_display_label(spec: ResolvedPlotSpec, column: str | None) -> str:
    if spec.colorbar_label:
        return str(spec.colorbar_label)
    if column is None:
        return "Value"
    for option in spec.hue_options:
        if option.column == column:
            return option.label
    return humanize_display_text(column)


def _axis_style(axis_styles: dict[str, AxisStyle] | None, column: str | None) -> AxisStyle | None:
    if column is None:
        return None
    return (axis_styles or {}).get(str(column))


def _axis_category_value(
    row: dict[str, object],
    column: str,
    *,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> str:
    style = _axis_style(axis_styles, column)
    if style is not None:
        return normalize_axis_category(style, row[column], row=row)
    return _category_key(row[column])


def _axis_categories(
    values: list[str],
    *,
    column: str | None,
    axis_styles: dict[str, AxisStyle] | None = None,
    legend_only: bool = False,
) -> list[str]:
    style = _axis_style(axis_styles, column)
    if style is not None:
        return legend_categories(style, values) if legend_only else ordered_categories_for_axis(style, values)
    return ordered_categories_for_axis(None, values)


def _axis_category_label(
    value: object,
    *,
    column: str | None,
    axis_styles: dict[str, AxisStyle] | None = None,
    compact: bool = False,
) -> str:
    style = _axis_style(axis_styles, column)
    if style is not None:
        return axis_display_text(style, value, compact=compact)
    return humanize_display_text(value)


def _continuous_color_encoding(
    rows: list[dict],
    spec: ResolvedPlotSpec,
    *,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> dict[str, object] | None:
    column = spec.color_column
    hue_type = _hue_option_type(spec, column)
    if hue_type in {"categorical", "binary", "ordinal"}:
        return None
    if hue_type == "continuous":
        return _continuous_scatter_encoding(rows, column)
    style = _axis_style(axis_styles, column)
    if style is not None and style.kind in {"categorical", "binary", "ordinal"}:
        return None
    return _continuous_scatter_encoding(rows, column)


def _add_continuous_colorbar(fig: Any, ax: Any, *, spec: ResolvedPlotSpec, color_encoding: dict[str, object]) -> None:
    from matplotlib.cm import ScalarMappable

    label = _explicit_axis_label(_hue_display_label(spec, spec.color_column), width=24, max_lines=3) or "Value"
    colorbar = fig.colorbar(
        ScalarMappable(norm=color_encoding["norm"], cmap=str(color_encoding["cmap"])),
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label=label,
    )
    colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
    colorbar.set_label(label, fontsize=11, color=TEXT_COLOR)


def _category_color_map(
    row_groups: list[list[dict]],
    column: str | None,
    *,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> tuple[dict[str, str], list[str]]:
    if column is None:
        return {}, []
    flattened = [row for rows in row_groups for row in rows]
    if flattened and column not in flattened[0]:
        raise ContractViolationError(f"plot color column is missing: {column!r}")
    style = _axis_style(axis_styles, column)
    if style is not None:
        values = [_axis_category_value(row, column, axis_styles=axis_styles) for row in flattened]
        categories = _axis_categories(values, column=column, axis_styles=axis_styles, legend_only=True)
        color_map = axis_color_map(style, categories, fallback_palette=PUBLICATION_PALETTE)
        return color_map, categories
    categories = ordered_categories_for_axis(None, [_category_key(row[column]) for row in flattened])
    color_map = axis_color_map(None, categories, fallback_palette=PUBLICATION_PALETTE)
    return color_map, categories


def _color_series(
    rows: list[dict],
    column: str | None,
    *,
    color_map: dict[str, str] | None = None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> tuple[list[str], list[str]]:
    if column is None:
        return [PUBLICATION_PALETTE[0]] * len(rows), []
    if rows and column not in rows[0]:
        raise ContractViolationError(f"plot color column is missing: {column!r}")
    resolved_map = color_map or _category_color_map([rows], column, axis_styles=axis_styles)[0]
    categories = _axis_categories(list(resolved_map), column=column, axis_styles=axis_styles)
    return [
        resolved_map.get(_axis_category_value(row, column, axis_styles=axis_styles), "#9AA5B1") for row in rows
    ], categories


def _continuous_scatter_encoding(rows: list[dict], column: str | None) -> dict[str, object] | None:
    if column is None:
        return None
    if rows and column not in rows[0]:
        raise ContractViolationError(f"plot color column is missing: {column!r}")
    numeric = np.asarray(
        [
            _coerce_finite_float(row.get(column)) if _coerce_finite_float(row.get(column)) is not None else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    finite = numeric[np.isfinite(numeric)]
    if finite.size < 2 or float(np.nanmin(finite)) == float(np.nanmax(finite)):
        return None
    from matplotlib import colors as mcolors

    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    if minimum < 0.0 < maximum:
        max_abs = max(abs(minimum), abs(maximum), 1e-6)
        return {
            "values": numeric,
            "cmap": "PuOr",
            "norm": mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
            "vmin": None,
            "vmax": None,
        }
    return {
        "values": numeric,
        "cmap": "cividis",
        "norm": mcolors.Normalize(vmin=minimum, vmax=maximum),
        "vmin": minimum,
        "vmax": maximum,
    }


def _scatter_point_sizes(
    rows: list[dict],
    *,
    size_column: str | None,
    default_size: float,
    size_range: tuple[float, float] | None,
) -> np.ndarray:
    base = np.full(len(rows), float(default_size), dtype=np.float64)
    if size_column is None:
        return base
    if rows and size_column not in rows[0]:
        raise ContractViolationError(f"plot size column is missing: {size_column!r}")
    values = np.asarray(
        [
            (
                _coerce_finite_float(row.get(size_column))
                if _coerce_finite_float(row.get(size_column)) is not None
                else np.nan
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return base
    size_min, size_max = size_range or (max(default_size * 0.75, 18.0), max(default_size * 2.4, 48.0))
    if float(np.nanmin(finite)) == float(np.nanmax(finite)):
        midpoint = (size_min + size_max) / 2.0
        base[np.isfinite(values)] = midpoint
        return base
    scaled = (values - float(np.nanmin(finite))) / (float(np.nanmax(finite)) - float(np.nanmin(finite)))
    base[np.isfinite(values)] = size_min + (scaled[np.isfinite(values)] * (size_max - size_min))
    return base


def _shape_marker_map(row_groups: list[list[dict]], column: str | None) -> tuple[dict[str, str], list[str]]:
    if column is None:
        return {}, []
    flattened = [row for rows in row_groups for row in rows]
    if flattened and column not in flattened[0]:
        raise ContractViolationError(f"plot shape column is missing: {column!r}")
    categories = sorted({_category_key(row[column]) for row in flattened})
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
    matched_prefix = False
    for prefix in (
        "context_delta_distribution_",
        "context_geometry_metrics_",
        "wildtype_reference_margins_",
        "synthetic_centroid_margins_",
        "tradeoff_",
        "pca_",
    ):
        if candidate_key.startswith(prefix):
            candidate_key = candidate_key[len(prefix) :]
            matched_prefix = True
            break
    if not matched_prefix:
        return ""
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


def _render_placeholder_panel(
    ax: Any,
    *,
    panel_title: str,
    message: str,
    detail: str | None = None,
    square: bool = False,
) -> None:
    ax.cla()
    ax.set_title(wrap_plot_title(panel_title, width=24, max_lines=2), pad=8)
    _apply_axes_style(ax, grid=False, square=square)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(
        0.5,
        0.58,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.0,
        color=TEXT_COLOR,
        fontweight="semibold",
    )
    if detail:
        ax.text(
            0.5,
            0.42,
            detail,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.0,
            color=SPINE_COLOR,
        )


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


def _legend_handles(
    plt: Any,
    categories: list[str],
    color_map: dict[str, str],
    *,
    column: str | None = None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> list[Any]:
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
            label=_axis_category_label(category, column=column, axis_styles=axis_styles),
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
    point_sizes: np.ndarray | None = None,
    alpha: float,
    continuous_color: dict[str, object] | None = None,
    axis_styles: dict[str, AxisStyle] | None = None,
    rasterized: bool = False,
    edgecolors: str = "white",
    linewidths: float = 0.25,
) -> None:
    sizes = point_sizes if point_sizes is not None else np.full(len(rows), float(point_size), dtype=np.float64)
    if shape_column is None:
        if continuous_color is not None:
            values = np.asarray(continuous_color["values"], dtype=np.float64)
            valid = np.isfinite(values)
            invalid = ~valid
            if np.any(valid):
                ax.scatter(
                    [float(rows[index][resolved_x]) for index in np.flatnonzero(valid)],
                    [float(rows[index][resolved_y]) for index in np.flatnonzero(valid)],
                    c=values[valid],
                    cmap=str(continuous_color["cmap"]),
                    norm=continuous_color["norm"],
                    s=sizes[valid],
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    rasterized=rasterized,
                )
            if np.any(invalid):
                ax.scatter(
                    [float(rows[index][resolved_x]) for index in np.flatnonzero(invalid)],
                    [float(rows[index][resolved_y]) for index in np.flatnonzero(invalid)],
                    c="#9AA5B1",
                    s=sizes[invalid],
                    alpha=alpha,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    rasterized=rasterized,
                )
        else:
            colors, _ = _color_series(
                rows,
                color_column,
                color_map=color_map if color_map else None,
                axis_styles=axis_styles,
            )
            ax.scatter(
                [float(row[resolved_x]) for row in rows],
                [float(row[resolved_y]) for row in rows],
                c=colors,
                s=sizes,
                alpha=alpha,
                edgecolors=edgecolors,
                linewidths=linewidths,
                rasterized=rasterized,
            )
        return
    if rows and shape_column not in rows[0]:
        raise ContractViolationError(f"plot shape column is missing: {shape_column!r}")
    for shape_category, marker in shape_map.items():
        group_indices = [index for index, row in enumerate(rows) if _category_key(row[shape_column]) == shape_category]
        group_rows = [rows[index] for index in group_indices]
        if not group_rows:
            continue
        if continuous_color is not None:
            values = np.asarray(continuous_color["values"], dtype=np.float64)[group_indices]
            valid = np.isfinite(values)
            invalid = ~valid
            group_sizes = sizes[np.asarray(group_indices, dtype=np.int64)]
            if np.any(valid):
                ax.scatter(
                    [float(group_rows[index][resolved_x]) for index in np.flatnonzero(valid)],
                    [float(group_rows[index][resolved_y]) for index in np.flatnonzero(valid)],
                    c=values[valid],
                    cmap=str(continuous_color["cmap"]),
                    norm=continuous_color["norm"],
                    s=group_sizes[valid],
                    alpha=alpha,
                    marker=marker,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    rasterized=rasterized,
                )
            if np.any(invalid):
                ax.scatter(
                    [float(group_rows[index][resolved_x]) for index in np.flatnonzero(invalid)],
                    [float(group_rows[index][resolved_y]) for index in np.flatnonzero(invalid)],
                    c="#9AA5B1",
                    s=group_sizes[invalid],
                    alpha=alpha,
                    marker=marker,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    rasterized=rasterized,
                )
        else:
            colors, _ = _color_series(
                group_rows,
                color_column,
                color_map=color_map if color_map else None,
                axis_styles=axis_styles,
            )
            ax.scatter(
                [float(row[resolved_x]) for row in group_rows],
                [float(row[resolved_y]) for row in group_rows],
                c=colors,
                s=sizes[np.asarray(group_indices, dtype=np.int64)],
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
    axis_styles: dict[str, AxisStyle] | None = None,
) -> None:
    color_legend = None
    if color_categories and color_title is not None:
        color_legend = ax.legend(
            handles=_legend_handles(plt, color_categories, color_map, column=color_title, axis_styles=axis_styles),
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
    plot_id: str | None,
    color_categories: list[str],
    color_map: dict[str, str],
    color_title: str | None,
    shape_categories: list[str],
    shape_map: dict[str, str],
    shape_title: str | None,
    single_row: bool | None = True,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> float:
    legend_specs: list[list[Any]] = []
    if color_categories and color_title is not None:
        legend_specs.append(
            _legend_handles(plt, color_categories, color_map, column=color_title, axis_styles=axis_styles)
        )
    if shape_categories and shape_title is not None:
        legend_specs.append(_shape_legend_handles(plt, shape_categories, shape_map))
    if not legend_specs:
        return 0.0

    lowered_plot_ids = {
        "balanced_design_family_margin_gallery",
        "design_centroid_margin_gallery",
        "appendix_umap_gallery",
    }
    legend_y = 0.008 if plot_id in lowered_plot_ids else 0.012
    base_margin = 0.08 if plot_id in lowered_plot_ids else 0.055
    for handles in legend_specs:
        legend_labels = [handle.get_label() for handle in handles]
        resolved_single_row = False if single_row and len(legend_labels) > 12 else single_row
        layout = legend_layout(
            legend_labels,
            plot_id=plot_id,
            default_anchor_y=legend_y,
            default_base_margin=base_margin,
            row_step=0.048 if resolved_single_row is False else 0.038,
            max_columns=4,
            single_row=resolved_single_row,
        )
        legend = fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, layout.anchor_y),
            ncol=layout.columns,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.05,
            handletextpad=0.5,
        )
        _style_legend(legend)
        legend_y = layout.anchor_y + layout.bottom_margin
    return min(max(legend_y + 0.014, 0.1), 0.40)


def _add_side_figure_legends(
    fig: Any,
    plt: Any,
    *,
    color_categories: list[str],
    color_map: dict[str, str],
    color_title: str | None,
    shape_categories: list[str],
    shape_map: dict[str, str],
    shape_title: str | None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> float:
    legend_specs: list[list[Any]] = []
    if color_categories and color_title is not None:
        legend_specs.append(
            _legend_handles(plt, color_categories, color_map, column=color_title, axis_styles=axis_styles)
        )
    if shape_categories and shape_title is not None:
        legend_specs.append(_shape_legend_handles(plt, shape_categories, shape_map))
    if not legend_specs:
        return 0.0

    width, height = fig.get_size_inches()
    fig.set_size_inches(max(width + 2.6, 7.35), height, forward=True)
    for index, handles in enumerate(legend_specs):
        legend = fig.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(0.985, 0.5 - (index * 0.22)),
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.0,
            handletextpad=0.5,
        )
        _style_legend(legend)
    return 0.30


def _tight_layout_kwargs(
    spec: ResolvedPlotSpec,
    *,
    legend_bottom: float,
    legend_right: float = 0.0,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "pad": 0.95,
        "h_pad": 1.4,
        "w_pad": 0.95,
    }
    if spec.plot_id == "balanced_design_family_margin_gallery":
        kwargs["w_pad"] = 1.38
    if spec.plot_id == "design_centroid_margin_gallery":
        kwargs["w_pad"] = 1.18
    if spec.plot_id == "representation_health_summary":
        kwargs["w_pad"] = 1.85
    if spec.plot_id == "dataset_overview":
        kwargs["w_pad"] = 1.2
    if legend_bottom > 0.0 or legend_right > 0.0:
        kwargs["rect"] = (0.0, legend_bottom, max(0.58, 1.0 - legend_right), 0.995)
    return kwargs


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


def _draw_annotation_callouts(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    label_texts: list[str],
    marker_colors: list[str],
    font_size: float = 9.5,
    marker_size: float = 128.0,
    marker: str | None = "*",
) -> None:
    if not rows:
        return
    x_values = [float(row[resolved_x]) for row in rows]
    y_values = [float(row[resolved_y]) for row in rows]
    placed_boxes: list[tuple[float, float, float, float]] = []
    axes_box = ax.get_window_extent()
    display_x_mid = float((axes_box.x0 + axes_box.x1) / 2.0)
    display_y_mid = float((axes_box.y0 + axes_box.y1) / 2.0)
    if marker is not None and marker_size > 0.0:
        ax.scatter(
            x_values,
            y_values,
            c=marker_colors,
            s=marker_size,
            marker=marker,
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
        placement = choose_annotation_placement(
            display_x=display_x,
            display_y=display_y,
            label_text=label_text,
            axes_box=axes_box,
            placed_boxes=placed_boxes,
            x_mid=display_x_mid,
            y_mid=display_y_mid,
            font_size=font_size,
            left_padding_px=10.0,
            right_padding_px=10.0,
        )
        placed_boxes.append(placement.box)
        annotation = ax.annotate(
            label_text,
            xy=(point_x, point_y),
            xytext=(placement.offset_x, placement.offset_y),
            textcoords="offset pixels",
            fontsize=font_size,
            fontweight="semibold",
            ha=placement.ha,
            va=placement.va,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": ANNOTATION_LABEL_BOX_ALPHA,
            },
            arrowprops={"arrowstyle": "-", "color": SPINE_COLOR, "linewidth": 0.9},
            zorder=6,
        )
        annotation.set_clip_on(True)
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_clip_on(True)


def _draw_annotation_highlights(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    marker_size: float = 96.0,
) -> None:
    if not rows:
        return
    ax.scatter(
        [float(row[resolved_x]) for row in rows],
        [float(row[resolved_y]) for row in rows],
        s=marker_size,
        marker="*",
        facecolors="#111111",
        edgecolors="#111111",
        linewidths=0.75,
        zorder=5,
    )


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
    resolution = resolve_reference_set_rows(reference_set, rows)
    expected_ids = resolution.expected_ids
    if resolution.missing_columns:
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
                "missing_columns": resolution.missing_columns,
            },
        )
    missing_ids = [value for value in expected_ids if value not in resolution.matched_ids]
    if missing_ids and spec.annotation.missing_policy == "fail":
        raise ContractViolationError(
            f"reference_set {spec.annotation.reference_set!r} is missing required ids: {missing_ids}"
        )
    if not expected_ids and spec.annotation.missing_policy == "fail" and reference_set.require_non_empty:
        raise ContractViolationError(f"reference_set {spec.annotation.reference_set!r} matched no rows")
    complete = not missing_ids and (bool(expected_ids) or not reference_set.require_non_empty)
    if spec.annotation.missing_policy == "allow" and resolution.matched_ids:
        complete = True
    return (
        resolution.selected_rows,
        label_column,
        expected_ids,
        {
            "reference_set": spec.annotation.reference_set,
            "match_column": match_column,
            "label_column": label_column,
            "expected_ids": expected_ids,
            "matched_ids": resolution.matched_ids,
            "missing_ids": missing_ids,
            "complete": complete,
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
        return reference_annotation_label(str(row[resolved_label_column]))
    reference_set = context.config.reference_sets[spec.annotation.reference_set]
    display_labels = dict(getattr(reference_set, "display_labels", {}) or {})
    match_column = reference_set.match_column
    match_value = str(row.get(match_column, ""))
    return reference_annotation_label(str(display_labels.get(match_value, row[resolved_label_column])))


def _draw_resolved_annotations(
    ax: Any,
    *,
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    rows: list[dict[str, object]],
    resolved_x: str,
    resolved_y: str,
    resolved_label_column: str | None,
    color_map: dict[str, str],
    font_size: float = 9.5,
    marker_size: float = 128.0,
    marker: str | None = "*",
) -> None:
    if not rows or resolved_label_column is None:
        return
    label_mode = (
        "label_and_highlight"
        if spec.annotation is None
        else context.config.reference_sets[spec.annotation.reference_set].label_mode
    )
    if label_mode == "highlight_only" or len(rows) > 5:
        _draw_annotation_highlights(
            ax,
            rows=rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
        )
        return
    if label_mode != "label_and_highlight":
        return
    highlight_colors = (
        ["#111111"] * len(rows)
        if spec.annotation is not None
        else _color_series(
            rows,
            spec.color_column,
            color_map=color_map if color_map else None,
        )[0]
    )
    _draw_annotation_callouts(
        ax,
        rows=rows,
        resolved_x=resolved_x,
        resolved_y=resolved_y,
        label_texts=[
            _annotation_label_text(
                context,
                spec=spec,
                row=row,
                resolved_label_column=resolved_label_column,
            )
            for row in rows
        ],
        marker_colors=highlight_colors,
        font_size=font_size,
        marker_size=marker_size,
        marker=marker,
    )


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


def _prefer_single_row_panel_layout(plot_id: str | None, panel_count: int, *, configured: object = None) -> bool:
    if configured is not None:
        return bool(configured) and 1 < panel_count <= 4
    return bool(plot_id in _SINGLE_ROW_PANEL_PLOT_IDS and 1 < panel_count <= 4)


_HORIZONTAL_GROUPED_METRIC_PLOT_IDS: frozenset[str] = frozenset(
    {
        "design_structure_summary",
        "reference_alignment_summary",
        "representation_health_summary",
    }
)


def _panel_grid_dimensions(panel_count: int, *, prefer_single_row: bool = False) -> tuple[int, int]:
    if panel_count <= 1:
        return 1, 1
    if prefer_single_row and panel_count <= 4:
        return 1, panel_count
    if panel_count == 5:
        return 2, 3
    if panel_count == 6:
        return 2, 3
    if panel_count in {7, 8}:
        return 2, 4
    if panel_count == 4:
        return 2, 2
    columns = min(4, max(1, int(math.ceil(math.sqrt(panel_count)))))
    rows = int(np.ceil(panel_count / columns))
    return rows, columns


def _grid_figure_size(panel_count: int, *, square_panels: bool, prefer_single_row: bool = False) -> tuple[float, float]:
    if panel_count <= 1:
        return (5.15, 5.0 if square_panels else 4.7)
    rows, columns = _panel_grid_dimensions(panel_count, prefer_single_row=prefer_single_row)
    panel_width = 3.55 if prefer_single_row and columns >= 4 else 4.15 if columns >= 4 else 4.3
    panel_height = 4.2 if square_panels and prefer_single_row else 4.35 if square_panels else 4.05
    return (panel_width * columns, panel_height * rows)


def metric_panel_grid_layout(plot_id: str | None, panel_count: int) -> tuple[int, int, tuple[float, float]]:
    square_panels = metric_panel_uses_square_axes(plot_id)
    if plot_id == "representation_health_summary" and panel_count > 1:
        rows, columns = _panel_grid_dimensions(panel_count, prefer_single_row=True)
        figsize = _grid_figure_size(panel_count, square_panels=square_panels, prefer_single_row=True)
        return rows, columns, (figsize[0] + (1.25 * columns), figsize[1])
    rows, columns = _panel_grid_dimensions(panel_count)
    figsize = _grid_figure_size(panel_count, square_panels=square_panels)
    if plot_id == "representation_health_summary":
        figsize = (figsize[0] + (1.45 * columns), figsize[1])
    return rows, columns, figsize


def _ordered_heatmap_axis_values(rows: list[dict[str, object]], column: str, configured_order: list[str]) -> list[str]:
    observed = list(dict.fromkeys(str(row[column]) for row in rows))
    if not configured_order:
        return observed
    ordered = [value for value in configured_order if value in set(observed)]
    return ordered or observed


def _heatmap_grid_from_rows(
    rows: list[dict[str, object]],
    *,
    row_column: str,
    column_column: str,
    value_column: str,
    row_order: list[str],
    column_order: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    if not rows:
        raise ContractViolationError("heatmap rendering requires at least one input row")
    if value_column not in rows[0]:
        raise ContractViolationError(f"heatmap value column is missing from table: {value_column!r}")
    if row_column not in rows[0]:
        raise ContractViolationError(f"heatmap row column is missing: {row_column!r}")
    if column_column not in rows[0]:
        raise ContractViolationError(f"heatmap column column is missing: {column_column!r}")
    row_values = _ordered_heatmap_axis_values(rows, row_column, row_order)
    column_values = _ordered_heatmap_axis_values(rows, column_column, column_order)
    row_index = {row_value: index for index, row_value in enumerate(row_values)}
    column_index = {column_value: index for index, column_value in enumerate(column_values)}
    grid = np.full((len(row_values), len(column_values)), np.nan, dtype=np.float32)
    for row in rows:
        row_key = str(row[row_column])
        column_key = str(row[column_column])
        if row_key not in row_index or column_key not in column_index:
            continue
        grid[
            row_index[row_key],
            column_index[column_key],
        ] = float(row[value_column])
    return grid, row_values, column_values


def _heatmap_color_params(
    grids: list[np.ndarray],
    *,
    color_scale: str | None,
) -> tuple[str, object]:
    from matplotlib import colors as mcolors

    finite_chunks = [np.asarray(grid[np.isfinite(grid)], dtype=np.float32) for grid in grids if np.isfinite(grid).any()]
    if not finite_chunks:
        raise ContractViolationError("heatmap rendering requires at least one finite value")
    finite = np.concatenate(finite_chunks)
    if finite.size == 0:
        raise ContractViolationError("heatmap rendering requires at least one finite value")
    resolved_scale = str(color_scale or "auto")
    if resolved_scale == "auto":
        resolved_scale = "diverging" if float(np.min(finite)) < 0.0 < float(np.max(finite)) else "sequential"
    if resolved_scale == "diverging":
        max_abs = max(float(np.max(np.abs(finite))), 1e-6)
        return "PuOr", mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if minimum == maximum:
        maximum = minimum + 1e-6
    return "cividis", mcolors.Normalize(vmin=minimum, vmax=maximum)


def _render_heatmap_panel(
    ax: Any,
    *,
    grid: np.ndarray,
    row_values: list[str],
    column_values: list[str],
    row_column: str,
    column_column: str,
    title: str,
    cmap: str,
    norm: object,
    square_cells: bool = False,
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    show_y_tick_labels: bool = True,
    show_y_axis_label: bool = True,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> None:
    grid = np.asarray(grid, dtype=np.float32)
    image = ax.imshow(grid, cmap=cmap, norm=norm, aspect="equal" if square_cells else "auto")
    x_tick_labels = [
        _axis_category_label(value, column=column_column, axis_styles=axis_styles, compact=square_cells)
        for value in column_values
    ]
    y_tick_labels = [
        _axis_category_label(value, column=row_column, axis_styles=axis_styles, compact=square_cells)
        for value in row_values
    ]
    ax.set_xticks(
        range(len(column_values)),
        x_tick_labels,
        rotation=0 if square_cells else 30,
        ha="center" if square_cells else "right",
    )
    if show_y_tick_labels:
        ax.set_yticks(range(len(row_values)), y_tick_labels)
    else:
        ax.set_yticks(range(len(row_values)), [])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel(_resolved_axis_label(explicit_label=x_axis_label, fallback_label=column_column, width=20))
    ax.set_ylabel(
        _resolved_axis_label(explicit_label=y_axis_label, fallback_label=row_column, width=20)
        if show_y_axis_label
        else ""
    )
    ax.set_title(wrap_plot_title(title, width=24), pad=8)
    finite = np.asarray(grid[np.isfinite(grid)], dtype=np.float32)
    contrast_midpoint = float(np.mean(finite)) if finite.size else 0.0
    for row_index_value in range(len(row_values)):
        for column_index_value in range(len(column_values)):
            value = grid[row_index_value, column_index_value]
            if not np.isfinite(value):
                label = "NA"
                text_color = TEXT_COLOR
            else:
                label = f"{value:.2f}"
                text_color = "white" if float(value) >= contrast_midpoint else TEXT_COLOR
            ax.text(
                column_index_value,
                row_index_value,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9.2,
            )
    _apply_axes_style(ax, grid=False)
    if square_cells:
        _style_compact_category_tick_labels(ax, axis="x")
    if square_cells and show_y_tick_labels:
        _style_compact_category_tick_labels(ax, axis="y")
    return image


def _coerce_finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


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


def _wrapped_tick_label(value: object, *, width: int = 16, max_lines: int | None = None) -> str:
    return wrap_plot_title(humanize_display_text(str(value)), width=width, max_lines=max_lines)


def _style_compact_category_tick_labels(ax: Any, *, axis: str = "x", font_size: float = 9.2) -> None:
    tick_labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in tick_labels:
        label.set_fontsize(font_size)
        label.set_linespacing(0.92)
        label.set_rotation(0)
        label.set_rotation_mode("default")
        label.set_ha("center" if axis == "x" else "right")
        label.set_va("top" if axis == "x" else "center")


def _wrapped_axis_label(value: object, *, width: int = 22, max_lines: int | None = 4) -> str:
    return wrap_plot_title(humanize_display_text(str(value)), width=width, max_lines=max_lines)


def _contains_math_text(value: object) -> bool:
    text = str(value or "")
    return "$" in text or "\\(" in text or "\\[" in text


def _explicit_axis_label(value: object | None, *, width: int = 22, max_lines: int | None = 4) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    if _contains_math_text(text):
        return text
    return wrap_plot_title(text, width=width, max_lines=max_lines)


def _resolved_axis_label(
    *,
    explicit_label: object | None,
    fallback_label: object,
    width: int = 22,
    max_lines: int | None = 4,
) -> str:
    return _explicit_axis_label(explicit_label, width=width, max_lines=max_lines) or _wrapped_axis_label(
        fallback_label,
        width=width,
        max_lines=max_lines,
    )


def _short_candidate_model(value: object) -> str:
    text = humanize_display_text(value)
    normalized = text.casefold()
    if "20b" in normalized:
        return "20B"
    if "7b" in normalized:
        return "7B"
    return text


def _short_candidate_scope(value: object) -> str:
    text = humanize_display_text(value)
    normalized = text.casefold()
    if normalized in {
        "60 bp anchor",
        "anchor-source insert",
        "mixed-length anchor-source insert",
        "anchor-source insert mean",
    }:
        return "anchor insert"
    if normalized == "1 kb construct context":
        return "1 kb ctx"
    if normalized == "1 kb context anchor mean":
        return "1 kb anchor mean"
    if normalized == "reverse complement context 1 kb":
        return "RC 1 kb ctx"
    if normalized == "reverse complement context anchor mean":
        return "RC 1 kb anchor mean"
    if normalized == "reference core60":
        return "ref core60"
    if normalized == "reference context forward 1 kb":
        return "ref forward 1 kb"
    if normalized == "reference context forward anchor mean":
        return "ref forward anchor mean"
    if normalized == "reference context reverse complement 1 kb":
        return "ref RC 1 kb"
    if normalized == "reference context reverse complement anchor mean":
        return "ref RC anchor mean"
    if normalized == "anchor + anchor-mean concat":
        return "anchor + anchor-mean"
    if normalized == "anchor + 1 kb context concat":
        return "anchor + 1 kb ctx"
    return text


def _compact_candidate_scope(value: object) -> str:
    text = humanize_display_text(value)
    normalized = text.casefold()
    if normalized in {
        "60 bp anchor",
        "anchor-source insert",
        "mixed-length anchor-source insert",
        "anchor-source insert mean",
    }:
        return "anchor insert"
    if normalized == "1 kb construct context":
        return "1kb ctx"
    if normalized == "1 kb context anchor mean":
        return "1kb anchor mean"
    if normalized == "reverse complement context 1 kb":
        return "RC 1kb ctx"
    if normalized == "reverse complement context anchor mean":
        return "RC 1kb anchor"
    if normalized == "reference core60":
        return "ref core60"
    if normalized == "reference context forward 1 kb":
        return "ref fwd 1kb"
    if normalized == "reference context forward anchor mean":
        return "ref fwd anchor"
    if normalized == "reference context reverse complement 1 kb":
        return "ref RC 1kb"
    if normalized == "reference context reverse complement anchor mean":
        return "ref RC anchor"
    if normalized == "anchor + anchor-mean concat":
        return "anchor+anchor-mean"
    if normalized == "anchor + 1 kb context concat":
        return "anchor+1kb ctx"
    return text


def _short_candidate_family(value: object) -> str:
    text = humanize_display_text(value)
    normalized = text.casefold()
    if normalized == "intermediate block mean":
        return "Block"
    if normalized == "output-layer mean":
        return "Output"
    return text


def _candidate_tick_label(
    row: dict[str, object],
    *,
    fallback_column: str,
    include_family: bool = True,
    include_scope: bool = True,
    multiline: bool = True,
) -> str:
    model = str(row.get("candidate_model") or "").strip()
    scope = str(row.get("candidate_scope") or "").strip()
    family = str(row.get("candidate_family") or "").strip()
    if model:
        short_model = _short_candidate_model(model)
        parts = [short_model]
        if include_scope and scope:
            scope_label = _compact_candidate_scope(scope) if not multiline else _short_candidate_scope(scope)
            parts[-1] = f"{short_model} {scope_label}"
        if include_family and family:
            short_family = _short_candidate_family(family)
            parts.append(short_family)
        separator = "\n" if multiline else " "
        return separator.join(part for part in parts if part)
    base_label = _candidate_row_label(
        row,
        fallback_column=fallback_column,
        include_family=include_family,
    )
    if multiline:
        return _wrapped_tick_label(base_label, width=12, max_lines=4)
    return base_label


def _style_metric_tick_labels(
    ax: Any,
    *,
    label_count: int,
    axis: str = "x",
    rotation: float = 0.0,
    ha: str | None = None,
    va: str | None = None,
) -> None:
    if label_count >= 8:
        font_size = PLOT_TICK_FONT_SIZE - 3.1
    elif label_count >= 6:
        font_size = PLOT_TICK_FONT_SIZE - 1.4
    else:
        font_size = PLOT_TICK_FONT_SIZE - 0.6
    if axis == "x" and rotation:
        font_size -= 0.8
    tick_labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    default_ha = "right" if axis == "y" or rotation else "center"
    default_va = "center" if axis == "y" else "top"
    for label in tick_labels:
        label.set_fontsize(font_size)
        label.set_linespacing(0.95)
        label.set_multialignment("right" if rotation else "center")
        label.set_rotation(rotation)
        label.set_rotation_mode("anchor")
        label.set_ha(ha or default_ha)
        label.set_va(va or default_va)


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
    axis_styles: dict[str, AxisStyle] | None = None,
) -> dict[str, object]:
    finite_rows = [
        row
        for row in rows
        if _coerce_finite_float(row.get(resolved_x)) is not None
        and _coerce_finite_float(row.get(resolved_y)) is not None
    ]
    if not finite_rows:
        _render_placeholder_panel(
            ax,
            panel_title=compact_candidate_title(panel_title),
            message="Margins unavailable",
            detail="No finite values in this snapshot",
            square=True,
        )
        ax.set_xlabel(
            _resolved_axis_label(
                explicit_label=spec.x_axis_label,
                fallback_label=_scatter_axis_label(rows, resolved_column=resolved_x, display_column="x_display_name"),
                width=28,
                max_lines=2,
            )
        )
        ax.set_ylabel(
            _resolved_axis_label(
                explicit_label=spec.y_axis_label,
                fallback_label=_scatter_axis_label(rows, resolved_column=resolved_y, display_column="y_display_name"),
                width=28,
                max_lines=2,
            )
        )
        return {}

    x_values = [float(row[resolved_x]) for row in finite_rows]
    y_values = [float(row[resolved_y]) for row in finite_rows]
    x_span = float(np.ptp(np.asarray(x_values, dtype=np.float64))) if x_values else 0.0
    y_span = float(np.ptp(np.asarray(y_values, dtype=np.float64))) if y_values else 0.0
    collapsed_panel = x_span <= 1e-12 and y_span <= 1e-12
    render_mode = spec.render_mode or "points"
    colors, _ = _color_series(
        finite_rows,
        spec.color_column,
        color_map=color_map if color_map else None,
        axis_styles=axis_styles,
    )
    if collapsed_panel:
        centroid_x = x_values[0] if x_values else 0.0
        centroid_y = y_values[0] if y_values else 0.0
        point_style = scatter_style(len(rows))
        ax.scatter(
            [centroid_x],
            [centroid_y],
            c="#111111",
            s=max(point_style.point_size * 18.0, 90.0),
            alpha=0.92,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.set_xlim(centroid_x - 0.055, centroid_x + 0.055)
        ax.set_ylim(centroid_y - 0.055, centroid_y + 0.055)
        ax.text(
            0.5,
            0.93,
            "Collapsed to one point",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.0,
            color=SPINE_COLOR,
        )
    elif render_mode == "hexbin":
        ax.hexbin(
            x_values,
            y_values,
            gridsize=max(12, min(48, int(np.sqrt(len(finite_rows))) * 2)),
            cmap="cividis",
        )
    elif render_mode == "density_contour":
        bins = max(10, min(30, int(np.sqrt(len(finite_rows))) * 2))
        histogram, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=bins)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax.contour(x_centers, y_centers, histogram.T, levels=4, cmap="cividis")
        density_style = scatter_style(len(finite_rows))
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
            finite_rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            color_column=spec.color_column,
            color_map=color_map,
            shape_column=_effective_shape_column(spec),
            shape_map=shape_map,
            point_size=point_style.point_size,
            alpha=point_style.alpha,
            axis_styles=axis_styles,
            rasterized=point_style.rasterized,
            edgecolors=point_style.edgecolors,
            linewidths=point_style.linewidths,
        )
    _add_zero_reference_lines(ax, x_values=x_values, y_values=y_values)
    ax.set_xlabel(
        _resolved_axis_label(
            explicit_label=spec.x_axis_label,
            fallback_label=_scatter_axis_label(rows, resolved_column=resolved_x, display_column="x_display_name"),
            width=28,
            max_lines=2,
        )
    )
    ax.set_ylabel(
        _resolved_axis_label(
            explicit_label=spec.y_axis_label,
            fallback_label=_scatter_axis_label(rows, resolved_column=resolved_y, display_column="y_display_name"),
            width=28,
            max_lines=2,
        )
    )
    ax.set_title(wrap_plot_title(compact_candidate_title(panel_title), width=22, max_lines=3), pad=8)
    _apply_axes_style(ax, grid=True, square=True)
    selected_rows, resolved_label_column, _, annotation_state = _resolve_annotation_rows(
        context,
        finite_rows,
        spec=spec,
    )
    _draw_resolved_annotations(
        ax,
        context=context,
        spec=spec,
        rows=selected_rows,
        resolved_x=resolved_x,
        resolved_y=resolved_y,
        resolved_label_column=resolved_label_column,
        color_map=color_map,
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
    x_axis_label: str | None = None,
    y_axis_label: str | None = None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> None:
    style = _axis_style(axis_styles, color_column)
    if render_mode == "violin_box" and color_column is not None and style is not None and style.ordinal_subset:
        allowed = {str(value) for value in style.ordinal_subset}
        rows = [row for row in rows if normalize_axis_category(style, row.get(color_column), row=row) in allowed]
        if not rows:
            raise ContractViolationError("ordinal distribution requires at least one row in the configured subset")
    values = np.asarray([float(row[metric_column]) for row in rows], dtype=np.float32)
    bin_count = max(5, min(30, int(np.sqrt(values.size)) + 1))
    boxplot_kwargs = {
        "widths": 0.18,
        "boxprops": {"color": "#111111", "linewidth": 1.2},
        "whiskerprops": {"color": "#111111", "linewidth": 1.2},
        "capprops": {"color": "#111111", "linewidth": 1.2},
        "medianprops": {"color": "#111111", "linewidth": 1.35},
        "flierprops": {
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "#111111",
            "markeredgewidth": 1.25,
            "markersize": 7.0,
            "linestyle": "none",
        },
    }
    if render_mode == "ecdf":
        x_axis_fallback = metric_column
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
        x_axis_fallback = color_column or metric_column
        if color_column is None:
            violin = ax.violinplot([values], showmeans=False, showmedians=False)
            for body in violin["bodies"]:
                body.set_facecolor(PUBLICATION_PALETTE[0])
                body.set_alpha(0.5)
            ax.boxplot([values], **boxplot_kwargs)
            ax.set_xticks([1], [humanize_display_text(metric_column)])
        else:
            if rows and color_column not in rows[0]:
                raise ContractViolationError(f"distribution color column is missing: {color_column!r}")
            categories = _axis_categories(
                [_axis_category_value(row, color_column, axis_styles=axis_styles) for row in rows],
                column=color_column,
                axis_styles=axis_styles,
            )
            grouped_values = [
                np.asarray(
                    [
                        float(row[metric_column])
                        for row in rows
                        if _axis_category_value(row, color_column, axis_styles=axis_styles) == category
                    ],
                    dtype=np.float32,
                )
                for category in categories
            ]
            violin = ax.violinplot(grouped_values, showmeans=False, showmedians=False)
            for index, body in enumerate(violin["bodies"]):
                body.set_facecolor(PUBLICATION_PALETTE[index % len(PUBLICATION_PALETTE)])
                body.set_alpha(0.45)
            ax.boxplot(grouped_values, **boxplot_kwargs)
            ax.set_xticks(
                range(1, len(categories) + 1),
                [
                    _axis_category_label(category, column=color_column, axis_styles=axis_styles, compact=True)
                    for category in categories
                ],
                rotation=0 if style is not None and style.compact_display_labels else 25,
                ha="center" if style is not None and style.compact_display_labels else "right",
            )
        ax.set_ylabel(
            _resolved_axis_label(
                explicit_label=y_axis_label,
                fallback_label=humanize_display_text(metric_column),
                width=18,
            )
        )
    else:
        x_axis_fallback = metric_column
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
    ax.set_xlabel(
        _resolved_axis_label(
            explicit_label=x_axis_label,
            fallback_label=x_axis_fallback,
            width=20,
        )
    )
    ax.set_title(wrap_plot_title(panel_title, width=24), pad=8)
    _apply_axes_style(ax, grid=True, square=square)
    if render_mode == "violin_box" and style is not None and style.compact_display_labels:
        _style_compact_category_tick_labels(ax, axis="x")


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

    def _value_sort_key(row: dict[str, object], *, descending: bool) -> tuple[int, float, str]:
        value = _coerce_finite_float(row.get(spec.value_column))
        label = str(row.get(label_column) or "").casefold()
        if value is None:
            return (1, 0.0, label)
        sortable = -value if descending else value
        return (0, sortable, label)

    if sort_rule == "value_asc":
        return sorted(rows, key=lambda row: _value_sort_key(row, descending=False))
    if sort_rule == "value_desc":
        return sorted(rows, key=lambda row: _value_sort_key(row, descending=True))
    direction = ""
    if spec.direction_column is not None and rows:
        direction = str(rows[0].get(spec.direction_column) or "").strip().lower()
    descending = direction != "lower_is_better"
    return sorted(rows, key=lambda row: _value_sort_key(row, descending=descending))


def _render_metric_panel(
    ax: Any,
    *,
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
    panel_title: str,
    color_map: dict[str, str],
    square: bool = False,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> None:
    if spec.value_column is None:
        raise ContractViolationError("metric_panel_grid rendering requires value_column")
    label_column = spec.label_column or spec.column_column
    if label_column is None:
        raise ContractViolationError("metric_panel_grid rendering requires label_column")
    ordered_rows = _sorted_metric_rows(rows, spec=spec)
    grouped_family_bars = (
        spec.color_column == "candidate_family"
        and all(str(row.get("candidate_model") or "").strip() for row in ordered_rows)
        and all(str(row.get("candidate_scope") or "").strip() for row in ordered_rows)
    )
    horizontal_metric = spec.plot_id == "representation_health_summary" and not grouped_family_bars
    include_family = not (spec.color_column == "candidate_family")
    labels = [
        _candidate_tick_label(
            row,
            fallback_column=label_column,
            include_family=include_family,
            multiline=not horizontal_metric,
        )
        for row in ordered_rows
    ]
    if spec.color_column is not None:
        if ordered_rows and spec.color_column not in ordered_rows[0]:
            raise ContractViolationError(f"metric_panel_grid color column is missing: {spec.color_column!r}")
        bar_colors = [
            color_map[_axis_category_value(row, spec.color_column, axis_styles=axis_styles)] for row in ordered_rows
        ]
    else:
        bar_colors = [PUBLICATION_PALETTE[0]] * len(ordered_rows)
    ci_enabled = spec.ci_lower_column is not None and spec.ci_upper_column is not None and ordered_rows
    horizontal_grouped_metric = grouped_family_bars and spec.plot_id in _HORIZONTAL_GROUPED_METRIC_PLOT_IDS

    if horizontal_grouped_metric:
        family_order = ordered_categories_for_axis(None, [str(row["candidate_family"]) for row in ordered_rows])
        group_keys = list(
            dict.fromkeys(
                (
                    str(row["candidate_model"]),
                    str(row["candidate_scope"]),
                )
                for row in ordered_rows
            )
        )
        shared_scope = {
            str(row.get("candidate_scope") or "").strip()
            for row in ordered_rows
            if str(row.get("candidate_scope") or "").strip()
        }
        include_scope = len(shared_scope) > 1
        group_labels = [
            _candidate_tick_label(
                {
                    "candidate_model": model,
                    "candidate_scope": scope,
                },
                fallback_column=label_column,
                include_family=False,
                include_scope=include_scope,
                multiline=False,
            )
            for model, scope in group_keys
        ]
        group_positions = np.arange(len(group_keys), dtype=float)
        group_height = min(0.78, 0.32 * max(len(family_order), 1))
        bar_height = group_height / max(len(family_order), 1)
        offsets = np.linspace(
            -(group_height / 2.0) + (bar_height / 2.0),
            (group_height / 2.0) - (bar_height / 2.0),
            max(len(family_order), 1),
        )
        bar_value_pairs: list[tuple[Any, float]] = []
        errorbar_specs: list[tuple[float, float, float, float]] = []
        missing_positions: list[float] = []
        for family, offset in zip(family_order, offsets, strict=False):
            family_rows = {
                (str(row["candidate_model"]), str(row["candidate_scope"])): row
                for row in ordered_rows
                if str(row["candidate_family"]) == family
            }
            family_positions: list[float] = []
            family_values: list[float] = []
            family_ci_rows: list[dict[str, object]] = []
            for group_position, group_key in zip(group_positions, group_keys, strict=False):
                row = family_rows.get(group_key)
                if row is None:
                    continue
                y_position = float(group_position + offset)
                value = _coerce_finite_float(row.get(spec.value_column))
                if value is None:
                    missing_positions.append(y_position)
                    continue
                family_positions.append(y_position)
                family_values.append(value)
                family_ci_rows.append(row)
            if not family_positions:
                continue
            family_bars = ax.barh(
                family_positions,
                family_values,
                height=bar_height * 0.9,
                color=color_map[family],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
            )
            bar_value_pairs.extend(zip(family_bars, family_values, strict=True))
            if ci_enabled:
                for bar, row in zip(family_bars, family_ci_rows, strict=False):
                    lower = _coerce_finite_float(row.get(spec.ci_lower_column))
                    upper = _coerce_finite_float(row.get(spec.ci_upper_column))
                    if lower is None or upper is None:
                        continue
                    errorbar_specs.append(
                        (
                            float(bar.get_y() + (bar.get_height() / 2.0)),
                            float(row[spec.value_column]),
                            float(lower),
                            float(upper),
                        )
                    )
        ax.set_yticks(group_positions, group_labels)
        if group_positions.size:
            ax.set_ylim(float(group_positions.min()) - 0.55, float(group_positions.max()) + 0.55)
        ax.tick_params(axis="y", pad=6)
        _style_metric_tick_labels(ax, label_count=max(len(group_labels), len(bar_value_pairs)), axis="y")
        finite_values = [value for _, value in bar_value_pairs]
        finite_value_array = np.asarray(finite_values, dtype=np.float64)
        if spec.reference_line is not None:
            ax.axvline(float(spec.reference_line), color=SPINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
        if finite_value_array.size and float(finite_value_array.min()) < 0.0 < float(finite_value_array.max()):
            ax.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
        if errorbar_specs:
            ys = np.asarray([item[0] for item in errorbar_specs], dtype=np.float64)
            xs = np.asarray([item[1] for item in errorbar_specs], dtype=np.float64)
            lowers = np.asarray([max(item[1] - item[2], 0.0) for item in errorbar_specs], dtype=np.float64)
            uppers = np.asarray([max(item[3] - item[1], 0.0) for item in errorbar_specs], dtype=np.float64)
            ax.errorbar(
                xs,
                ys,
                xerr=np.vstack([lowers, uppers]),
                fmt="none",
                ecolor=SPINE_COLOR,
                elinewidth=0.9,
                capsize=2.0,
                alpha=0.85,
            )
        ax.set_ylabel("")
        ax.set_xlabel(_wrapped_axis_label(_metric_axis_label(rows=ordered_rows, spec=spec), width=28, max_lines=2))
        ax.set_title(wrap_plot_title(panel_title, width=24, max_lines=2), pad=8)
        _apply_axes_style(ax, grid=True, square=square)
        ax.margins(x=0.02, y=0.02)
        if not finite_value_array.size:
            _render_placeholder_panel(
                ax,
                panel_title=panel_title,
                message="Metric unavailable",
                detail="No finite values in this snapshot",
                square=square,
            )
            return
        span = float(finite_value_array.max() - finite_value_array.min())
        offset = max(span * 0.03, 0.018) if span > 0 else 0.018
        low = min(0.0, float(finite_value_array.min()))
        high = max(0.0, float(finite_value_array.max()))
        padding = max((high - low) * 0.1, 0.04)
        ax.set_xlim(low - padding, high + padding)
        missing_label_x = low + (padding * 0.6)
        ax.invert_yaxis()
        for bar, value in bar_value_pairs:
            x_text = value + offset if value >= 0 else value - offset
            ha = "left" if value >= 0 else "right"
            ax.text(
                x_text,
                bar.get_y() + (bar.get_height() / 2.0),
                f"{value:.3g}",
                va="center",
                ha=ha,
                fontsize=9,
                color=TEXT_COLOR,
            )
        for position in missing_positions:
            ax.text(
                missing_label_x,
                float(position),
                "NA",
                va="center",
                ha="left",
                fontsize=8.5,
                color=SPINE_COLOR,
            )
        return

    if horizontal_metric:
        positions = np.arange(len(ordered_rows), dtype=float)
        finite_positions: list[float] = []
        finite_values: list[float] = []
        finite_colors: list[str] = []
        finite_rows: list[dict[str, object]] = []
        missing_positions: list[float] = []
        for position, row, color in zip(positions, ordered_rows, bar_colors, strict=True):
            value = _coerce_finite_float(row.get(spec.value_column))
            if value is None:
                missing_positions.append(float(position))
                continue
            finite_positions.append(float(position))
            finite_values.append(value)
            finite_colors.append(color)
            finite_rows.append(row)

        bars = ax.barh(
            finite_positions,
            finite_values,
            color=finite_colors,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        if ci_enabled:
            errorbar_specs: list[tuple[float, float, float, float]] = []
            for position, row in zip(finite_positions, finite_rows, strict=True):
                lower = _coerce_finite_float(row.get(spec.ci_lower_column))
                upper = _coerce_finite_float(row.get(spec.ci_upper_column))
                if lower is None or upper is None:
                    continue
                errorbar_specs.append((float(position), float(row[spec.value_column]), lower, upper))
            if errorbar_specs:
                ys = np.asarray([item[0] for item in errorbar_specs], dtype=np.float64)
                xs = np.asarray([item[1] for item in errorbar_specs], dtype=np.float64)
                lowers = np.asarray([max(item[1] - item[2], 0.0) for item in errorbar_specs], dtype=np.float64)
                uppers = np.asarray([max(item[3] - item[1], 0.0) for item in errorbar_specs], dtype=np.float64)
                ax.errorbar(
                    xs,
                    ys,
                    xerr=np.vstack([lowers, uppers]),
                    fmt="none",
                    ecolor=SPINE_COLOR,
                    elinewidth=0.9,
                    capsize=2.0,
                    alpha=0.85,
                )

        ax.set_yticks(positions, labels)
        ax.tick_params(axis="y", pad=6)
        _style_metric_tick_labels(ax, label_count=len(labels), axis="y")
        finite_value_array = np.asarray(finite_values, dtype=np.float64)
        if spec.reference_line is not None:
            ax.axvline(float(spec.reference_line), color=SPINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
        if finite_value_array.size and float(finite_value_array.min()) < 0.0 < float(finite_value_array.max()):
            ax.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
        ax.set_ylabel("")
        ax.set_xlabel(_wrapped_axis_label(_metric_axis_label(rows=ordered_rows, spec=spec), width=28, max_lines=2))
        ax.set_title(wrap_plot_title(panel_title, width=24, max_lines=2), pad=8)
        _apply_axes_style(ax, grid=True, square=square)
        ax.margins(x=0.02, y=0.02)
        if not finite_value_array.size:
            _render_placeholder_panel(
                ax,
                panel_title=panel_title,
                message="Metric unavailable",
                detail="No finite values in this snapshot",
                square=square,
            )
            return
        if finite_value_array.size:
            span = float(finite_value_array.max() - finite_value_array.min())
            offset = max(span * 0.03, 0.018) if span > 0 else 0.018
            low = min(0.0, float(finite_value_array.min()))
            high = max(0.0, float(finite_value_array.max()))
            padding = max((high - low) * 0.1, 0.04)
            ax.set_xlim(low - padding, high + padding)
            missing_label_x = low + (padding * 0.6)
        else:
            offset = 0.018
            ax.set_xlim(-0.2, 0.2)
            missing_label_x = 0.04
        ax.invert_yaxis()
        for bar, value in zip(bars, finite_values, strict=True):
            x_text = value + offset if value >= 0 else value - offset
            ha = "left" if value >= 0 else "right"
            ax.text(
                x_text,
                bar.get_y() + (bar.get_height() / 2.0),
                f"{value:.3g}",
                va="center",
                ha=ha,
                fontsize=9,
                color=TEXT_COLOR,
            )
        for position in missing_positions:
            ax.text(
                missing_label_x,
                float(position),
                "NA",
                va="center",
                ha="left",
                fontsize=8.5,
                color=SPINE_COLOR,
            )
        return

    bar_value_pairs: list[tuple[Any, float]] = []
    errorbar_specs: list[tuple[float, float, float, float]] = []
    missing_positions: list[float] = []
    if grouped_family_bars:
        family_order = ordered_categories_for_axis(None, [str(row["candidate_family"]) for row in ordered_rows])
        group_keys = list(
            dict.fromkeys(
                (
                    str(row["candidate_model"]),
                    str(row["candidate_scope"]),
                )
                for row in ordered_rows
            )
        )
        shared_scope = {
            str(row.get("candidate_scope") or "").strip()
            for row in ordered_rows
            if str(row.get("candidate_scope") or "").strip()
        }
        include_scope = len(shared_scope) > 1
        group_labels = [
            _candidate_tick_label(
                {
                    "candidate_model": model,
                    "candidate_scope": scope,
                },
                fallback_column=label_column,
                include_family=False,
                include_scope=include_scope,
                multiline=False,
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
            family_ci_rows: list[dict[str, object]] = []
            for group_position, group_key in zip(group_positions, group_keys, strict=False):
                row = family_rows.get(group_key)
                if row is None:
                    continue
                x_position = float(group_position + offset)
                value = _coerce_finite_float(row.get(spec.value_column))
                if value is None:
                    missing_positions.append(x_position)
                    continue
                family_positions.append(x_position)
                family_values.append(value)
                family_ci_rows.append(row)
            if not family_positions:
                continue
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
                for bar, row in zip(family_bars, family_ci_rows, strict=False):
                    lower = _coerce_finite_float(row.get(spec.ci_lower_column))
                    upper = _coerce_finite_float(row.get(spec.ci_upper_column))
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
        if group_positions.size:
            ax.set_xlim(float(group_positions.min()) - 0.55, float(group_positions.max()) + 0.55)
    else:
        positions = np.arange(len(ordered_rows), dtype=float)
        finite_positions: list[float] = []
        finite_values: list[float] = []
        finite_colors: list[str] = []
        finite_rows: list[dict[str, object]] = []
        for position, row, color in zip(positions, ordered_rows, bar_colors, strict=True):
            value = _coerce_finite_float(row.get(spec.value_column))
            if value is None:
                missing_positions.append(float(position))
                continue
            finite_positions.append(float(position))
            finite_values.append(value)
            finite_colors.append(color)
            finite_rows.append(row)
        bars = ax.bar(
            finite_positions,
            finite_values,
            color=finite_colors,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
        )
        bar_value_pairs.extend(zip(bars, finite_values, strict=True))
        if ci_enabled:
            for position, row in zip(finite_positions, finite_rows, strict=True):
                lower = _coerce_finite_float(row.get(spec.ci_lower_column))
                upper = _coerce_finite_float(row.get(spec.ci_upper_column))
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
        if positions.size:
            ax.set_xlim(float(positions.min()) - 0.55, float(positions.max()) + 0.55)
    ax.tick_params(axis="x", pad=6)
    tick_labels_for_count = group_labels if grouped_family_bars else labels
    tick_rotation = 32.0 if grouped_family_bars else 0.0
    _style_metric_tick_labels(
        ax,
        label_count=max(len(tick_labels_for_count), len(bar_value_pairs)),
        rotation=tick_rotation,
        ha="right" if tick_rotation else None,
    )
    finite_values = [value for _, value in bar_value_pairs]
    finite_value_array = np.asarray(finite_values, dtype=np.float64)
    if spec.reference_line is not None:
        ax.axhline(float(spec.reference_line), color=SPINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9)
    if finite_value_array.size and float(finite_value_array.min()) < 0.0 < float(finite_value_array.max()):
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
    ax.set_title(wrap_plot_title(panel_title, width=24, max_lines=2), pad=8)
    _apply_axes_style(ax, grid=True, square=square)
    ax.margins(x=0.02, y=0.02)
    if not finite_value_array.size:
        _render_placeholder_panel(
            ax,
            panel_title=panel_title,
            message="Metric unavailable",
            detail="No finite values in this snapshot",
            square=square,
        )
        return
    if finite_value_array.size:
        span = float(finite_value_array.max() - finite_value_array.min())
        offset = max(span * 0.03, 0.018) if span > 0 else 0.018
        low = min(0.0, float(finite_value_array.min()))
        high = max(0.0, float(finite_value_array.max()))
        padding = max((high - low) * 0.1, 0.045)
        ax.set_ylim(low - padding, high + padding)
        missing_label_y = low + (padding * 0.55)
    else:
        offset = 0.018
        ax.set_ylim(-0.2, 0.2)
        missing_label_y = 0.03
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
    for position in missing_positions:
        ax.text(
            float(position),
            missing_label_y,
            "NA",
            va="bottom",
            ha="center",
            fontsize=8.5,
            color=SPINE_COLOR,
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
    axis_styles = axis_style_map_from_config(context.config)
    if spec.kind not in SUPPORTED_PLOT_KINDS:
        raise ContractViolationError(f"unsupported plot kind: {spec.kind}")
    if spec.kind in {"projection_scatter", "projection_grid"} and not spec.projection_ids:
        raise ContractViolationError("plot rendering requires at least one projection artifact")
    if spec.kind == "heatmap" and spec.enrichment_id is None and spec.scalar_id is None:
        raise ContractViolationError("heatmap rendering requires an enrichment or scalar artifact")
    if spec.kind == "heatmap_grid" and not spec.scalar_ids:
        raise ContractViolationError("heatmap_grid rendering requires at least one scalar artifact")
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
        grid, row_values, column_values = _heatmap_grid_from_rows(
            rows,
            row_column=row_column,
            column_column=column_column,
            value_column=metric_column,
            row_order=list(spec.row_order or []),
            column_order=list(spec.column_order or []),
        )
        cmap, norm = _heatmap_color_params([grid], color_scale=spec.color_scale)
        fig, ax = plt.subplots(figsize=(2.2 + 1.35 * len(column_values), 1.7 + 1.05 * len(row_values)))
        image = _render_heatmap_panel(
            ax,
            grid=grid,
            row_values=row_values,
            column_values=column_values,
            row_column=row_column,
            column_column=column_column,
            title=spec.plot_id,
            cmap=cmap,
            norm=norm,
            square_cells=bool(spec.square_panels),
            x_axis_label=spec.x_axis_label,
            y_axis_label=spec.y_axis_label,
            axis_styles=axis_styles,
        )
        colorbar = fig.colorbar(
            image,
            ax=ax,
            label=_explicit_axis_label(spec.colorbar_label, width=20) or humanize_display_text(metric_column),
        )
        colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
        colorbar.set_label(
            _explicit_axis_label(spec.colorbar_label, width=20) or humanize_display_text(metric_column),
            fontsize=11,
            color=TEXT_COLOR,
        )
    elif spec.kind == "heatmap_grid":
        heatmap_tables: list[tuple[str, np.ndarray, list[str], list[str]]] = []
        for scalar_id in spec.scalar_ids:
            table_path = context.output_root / "scalars" / scalar_id / "table.parquet"
            if not table_path.exists():
                raise MissingArtifactError(f"scalar artifact is missing for heatmap_grid rendering: {scalar_id}")
            rows = _table_rows(table_path)
            heatmap_tables.append(
                (
                    scalar_id,
                    *_heatmap_grid_from_rows(
                        rows,
                        row_column=str(spec.row_column),
                        column_column=str(spec.column_column),
                        value_column=str(spec.value_column or "metric_value"),
                        row_order=list(spec.row_order or []),
                        column_order=list(spec.column_order or []),
                    ),
                )
            )
        grids = [grid for _, grid, _, _ in heatmap_tables]
        cmap, norm = _heatmap_color_params(grids, color_scale=spec.color_scale)
        prefer_single_row = _prefer_single_row_panel_layout(
            spec.plot_id,
            len(heatmap_tables),
            configured=spec.single_row_panels,
        )
        rows_count, columns = _panel_grid_dimensions(len(heatmap_tables), prefer_single_row=prefer_single_row)
        max_row_count = max(len(row_values) for _, _, row_values, _ in heatmap_tables)
        max_column_count = max(len(column_values) for _, _, _, column_values in heatmap_tables)
        figure_size = (
            (
                max(9.6, (2.9 * columns) + 0.6),
                max(3.25, (2.55 * rows_count) + 0.12),
            )
            if spec.square_panels
            else (
                max(4.2, 1.9 + (1.15 * max_column_count)) * columns,
                max(4.1, 1.6 + (0.9 * max_row_count)) * rows_count,
            )
        )
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=figure_size,
            squeeze=False,
        )
        titles = spec.panel_titles or [scalar_id for scalar_id, _, _, _ in heatmap_tables]
        for axis in axes.ravel()[len(heatmap_tables) :]:
            axis.axis("off")
        image = None
        for panel_index, (axis, (_, grid, row_values, column_values), panel_title) in enumerate(
            zip(
                axes.ravel(),
                heatmap_tables,
                titles,
                strict=False,
            )
        ):
            image = _render_heatmap_panel(
                axis,
                grid=grid,
                row_values=row_values,
                column_values=column_values,
                row_column=str(spec.row_column),
                column_column=str(spec.column_column),
                title=panel_title,
                cmap=cmap,
                norm=norm,
                square_cells=bool(spec.square_panels),
                x_axis_label=spec.x_axis_label,
                y_axis_label=spec.y_axis_label,
                show_y_tick_labels=not (spec.hide_repeated_y_axis and panel_index > 0),
                show_y_axis_label=not (spec.hide_repeated_y_axis and panel_index > 0),
                axis_styles=axis_styles,
            )
        assert image is not None
        if spec.square_panels:
            fig.subplots_adjust(left=0.07, right=0.875, wspace=0.18, bottom=0.18, top=0.79)
            colorbar = fig.colorbar(
                image,
                cax=fig.add_axes([0.922, 0.19, 0.013, 0.58]),
                label=_explicit_axis_label(spec.colorbar_label, width=20)
                or humanize_display_text(str(spec.value_column or "metric_value")),
            )
        else:
            colorbar = fig.colorbar(
                image,
                ax=axes.ravel().tolist(),
                fraction=0.03,
                pad=0.03,
                label=_explicit_axis_label(spec.colorbar_label, width=20)
                or humanize_display_text(str(spec.value_column or "metric_value")),
            )
        colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
        colorbar.set_label(
            _explicit_axis_label(spec.colorbar_label, width=20)
            or humanize_display_text(str(spec.value_column or "metric_value")),
            fontsize=11,
            color=TEXT_COLOR,
        )
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
        color_encoding = _continuous_color_encoding(rows, spec, axis_styles=axis_styles)
        color_map, categories = (
            ({}, [])
            if color_encoding is not None
            else _category_color_map([rows], spec.color_column, axis_styles=axis_styles)
        )
        effective_shape_column = _effective_shape_column(spec)
        shape_map, shape_categories = _shape_marker_map([rows], effective_shape_column)
        finite_rows = [
            row
            for row in rows
            if _coerce_finite_float(row.get(resolved_x)) is not None
            and _coerce_finite_float(row.get(resolved_y)) is not None
        ]
        if not finite_rows:
            _render_placeholder_panel(
                ax,
                panel_title=spec.plot_id,
                message="Margins unavailable",
                detail="No finite values in this snapshot",
                square=True,
            )
            annotation_state = {}
        else:
            point_style = scatter_style(len(finite_rows))
            point_sizes = _scatter_point_sizes(
                finite_rows,
                size_column=spec.size_column,
                default_size=point_style.point_size,
                size_range=spec.size_range,
            )
            _scatter_points(
                ax,
                finite_rows,
                resolved_x=resolved_x,
                resolved_y=resolved_y,
                color_column=spec.color_column,
                color_map=color_map,
                shape_column=effective_shape_column,
                shape_map=shape_map,
                point_size=point_style.point_size,
                point_sizes=point_sizes,
                alpha=point_style.alpha,
                continuous_color=color_encoding,
                axis_styles=axis_styles,
                rasterized=point_style.rasterized,
                edgecolors=point_style.edgecolors,
                linewidths=point_style.linewidths,
            )
            x_values = [float(row[resolved_x]) for row in finite_rows]
            y_values = [float(row[resolved_y]) for row in finite_rows]
            _add_zero_reference_lines(ax, x_values=x_values, y_values=y_values)
            ax.set_xlabel(
                _resolved_axis_label(
                    explicit_label=spec.x_axis_label,
                    fallback_label=_scatter_axis_label(
                        rows,
                        resolved_column=resolved_x,
                        display_column="x_display_name",
                    ),
                    width=28,
                    max_lines=2,
                )
            )
            ax.set_ylabel(
                _resolved_axis_label(
                    explicit_label=spec.y_axis_label,
                    fallback_label=_scatter_axis_label(
                        rows,
                        resolved_column=resolved_y,
                        display_column="y_display_name",
                    ),
                    width=28,
                    max_lines=2,
                )
            )
            ax.set_title(wrap_plot_title(spec.plot_id, width=24), pad=8)
            if spec.size_column is not None:
                ax.text(
                    0.98,
                    0.02,
                    f"Point size: {humanize_display_text(spec.size_column)}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8.8,
                    color=SPINE_COLOR,
                )
            _apply_axes_style(ax, grid=True, square=True)
            selected_rows, resolved_label_column, _, annotation_state = _resolve_annotation_rows(
                context,
                finite_rows,
                spec=spec,
            )
            _draw_resolved_annotations(
                ax,
                context=context,
                spec=spec,
                rows=selected_rows,
                resolved_x=resolved_x,
                resolved_y=resolved_y,
                resolved_label_column=resolved_label_column,
                color_map=color_map,
                font_size=8.4 if spec.plot_id == "candidate_decision_frontier" else 9.5,
                marker_size=104.0 if spec.plot_id == "candidate_decision_frontier" else 128.0,
                marker="*",
            )
        plot_metadata["reference_panels"] = {
            spec.scalar_id or spec.distance_id or spec.plot_id: annotation_state,
        }
        if color_encoding is not None:
            from matplotlib.cm import ScalarMappable

            colorbar = fig.colorbar(
                ScalarMappable(norm=color_encoding["norm"], cmap=str(color_encoding["cmap"])),
                ax=ax,
                fraction=0.046,
                pad=0.04,
                label=_explicit_axis_label(_hue_display_label(spec, spec.color_column), width=24, max_lines=3)
                or "Value",
            )
            colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
            colorbar.set_label(
                _explicit_axis_label(_hue_display_label(spec, spec.color_column), width=24, max_lines=3) or "Value",
                fontsize=11,
                color=TEXT_COLOR,
            )
        elif (spec.render_mode or "points") == "points":
            _add_axis_legends(
                ax,
                plt,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_shape_column,
                axis_styles=axis_styles,
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
        prefer_single_row = _prefer_single_row_panel_layout(
            spec.plot_id,
            len(scalar_tables),
            configured=spec.single_row_panels,
        )
        rows_count, columns = _panel_grid_dimensions(len(scalar_tables), prefer_single_row=prefer_single_row)
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=_grid_figure_size(len(scalar_tables), square_panels=True, prefer_single_row=prefer_single_row),
            squeeze=False,
        )
        color_map, categories = _category_color_map(
            [rows for _, rows, _, _ in scalar_tables],
            spec.color_column,
            axis_styles=axis_styles,
        )
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
                axis_styles=axis_styles,
            )
            plot_metadata.setdefault("reference_panels", {})[scalar_id] = annotation_state
        grid_legend_bottom_margin = 0.0
        if (spec.render_mode or "points") == "points":
            grid_legend_bottom_margin = _add_figure_legends(
                fig,
                plt,
                plot_id=spec.plot_id,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_shape_column,
                axis_styles=axis_styles,
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
        square_count_panels = bool(getattr(spec, "square_panels", False))
        if square_count_panels and len(panel_values) <= 3:
            rows_count, columns = 1, len(panel_values)
        elif len(panel_values) <= 2:
            rows_count, columns = len(panel_values), 1
        else:
            rows_count, columns = _panel_grid_dimensions(len(panel_values))
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(
                ((4.0 * columns) + 0.35, 4.55)
                if square_count_panels and rows_count == 1
                else _grid_figure_size(len(panel_values), square_panels=True)
                if square_count_panels
                else (6.6 * columns, 5.8 * rows_count)
            ),
            squeeze=False,
        )
        color_map, categories = _category_color_map([rows], spec.color_column, axis_styles=axis_styles)
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
                bar_colors = [
                    color_map[_axis_category_value(row, spec.color_column, axis_styles=axis_styles)]
                    for row in panel_rows
                ]
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
            _apply_axes_style(axis, grid=True, square=square_count_panels)
        grid_legend_bottom_margin = 0.0
        if len(categories) > 1 and spec.color_column is not None:
            grid_legend_bottom_margin = _add_figure_legends(
                fig,
                plt,
                plot_id=spec.plot_id,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=[],
                shape_map={},
                shape_title=None,
                axis_styles=axis_styles,
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
        square_metric_panels = metric_panel_uses_square_axes(spec.plot_id)
        rows_count, columns, metric_figsize = metric_panel_grid_layout(spec.plot_id, len(panel_values))
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=metric_figsize,
            squeeze=False,
        )
        color_map, categories = _category_color_map([rows], spec.color_column, axis_styles=axis_styles)
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
                axis_styles=axis_styles,
            )
        plot_metadata["metric_columns"] = panel_values
        grid_legend_bottom_margin = 0.0
        if len(categories) > 1 and spec.color_column is not None:
            grid_legend_bottom_margin = _add_figure_legends(
                fig,
                plt,
                plot_id=spec.plot_id,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=[],
                shape_map={},
                shape_title=None,
                axis_styles=axis_styles,
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
        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        render_mode = spec.render_mode or "histogram"
        _render_distribution_panel(
            ax,
            rows=rows,
            metric_column=metric_column,
            color_column=spec.color_column,
            render_mode=render_mode,
            panel_title=artifact_id,
            square=False,
            x_axis_label=spec.x_axis_label,
            y_axis_label=spec.y_axis_label,
            axis_styles=axis_styles,
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
        prefer_single_row = _prefer_single_row_panel_layout(
            spec.plot_id,
            len(scalar_tables),
            configured=spec.single_row_panels,
        )
        rows_count, columns = _panel_grid_dimensions(len(scalar_tables), prefer_single_row=prefer_single_row)
        square_distribution_panels = bool(spec.square_panels)
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=_grid_figure_size(
                len(scalar_tables),
                square_panels=square_distribution_panels,
                prefer_single_row=prefer_single_row,
            ),
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
                x_axis_label=spec.x_axis_label,
                y_axis_label=spec.y_axis_label,
                axis_styles=axis_styles,
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
        prefer_single_row = _prefer_single_row_panel_layout(
            spec.plot_id,
            len(reducer_summaries),
            configured=spec.single_row_panels,
        )
        rows_count, columns = _panel_grid_dimensions(
            len(reducer_summaries),
            prefer_single_row=prefer_single_row,
        )
        square_curve_panels = spec.plot_id == "representation_scree_diagnostic"
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=_grid_figure_size(
                len(reducer_summaries),
                square_panels=square_curve_panels,
                prefer_single_row=prefer_single_row,
            ),
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
        legend_labels = ["Explained variance ratio", "Cumulative variance ratio"]
        layout = legend_layout(
            legend_labels,
            plot_id=spec.plot_id,
            default_anchor_y=0.02,
            default_base_margin=0.11,
            row_step=0.038,
            single_row=True,
        )
        legend = fig.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linewidth=1.8,
                    color=PUBLICATION_PALETTE[0],
                    label=legend_labels[0],
                ),
                plt.Line2D(
                    [],
                    [],
                    marker="s",
                    linewidth=1.8,
                    color=PUBLICATION_PALETTE[2],
                    label=legend_labels[1],
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, layout.anchor_y),
            ncol=layout.columns,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.1,
            handletextpad=0.5,
        )
        _style_legend(legend)
        grid_legend_bottom_margin = layout.bottom_margin
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
            color_encoding = _continuous_color_encoding(rows, spec, axis_styles=axis_styles)
            color_map, categories = (
                ({}, [])
                if color_encoding is not None
                else _category_color_map([rows], spec.color_column, axis_styles=axis_styles)
            )
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
                continuous_color=color_encoding,
                axis_styles=axis_styles,
                rasterized=point_style.rasterized,
                edgecolors=point_style.edgecolors,
                linewidths=point_style.linewidths,
            )
            ax.set_xlabel("Projection 1")
            ax.set_ylabel("Projection 2")
            panel_title = spec.panel_titles[0] if spec.panel_titles else spec.projection_ids[0]
            ax.set_title(wrap_plot_title(compact_candidate_title(panel_title), width=28), pad=8)
            _apply_axes_style(ax, grid=True, square=True)
            selected_rows, resolved_label_column, _, annotation_state = _resolve_annotation_rows(
                context,
                rows,
                spec=spec,
            )
            _draw_resolved_annotations(
                ax,
                context=context,
                spec=spec,
                rows=selected_rows,
                resolved_x="x",
                resolved_y="y",
                resolved_label_column=resolved_label_column,
                color_map=color_map,
            )
            plot_metadata["reference_panels"] = {spec.projection_ids[0]: annotation_state}
            if color_encoding is not None:
                _add_continuous_colorbar(fig, ax, spec=spec, color_encoding=color_encoding)
            else:
                if len(categories) + len(shape_categories) > 8:
                    grid_legend_right_margin = _add_side_figure_legends(
                        fig,
                        plt,
                        color_categories=categories,
                        color_map=color_map,
                        color_title=spec.color_column,
                        shape_categories=shape_categories,
                        shape_map=shape_map,
                        shape_title=effective_shape_column,
                        axis_styles=axis_styles,
                    )
                else:
                    grid_legend_bottom_margin = _add_figure_legends(
                        fig,
                        plt,
                        plot_id=spec.plot_id,
                        color_categories=categories,
                        color_map=color_map,
                        color_title=spec.color_column,
                        shape_categories=shape_categories,
                        shape_map=shape_map,
                        shape_title=effective_shape_column,
                        single_row=False,
                        axis_styles=axis_styles,
                    )
        else:
            prefer_single_row = _prefer_single_row_panel_layout(
                spec.plot_id,
                len(projection_tables),
                configured=spec.single_row_panels,
            )
            rows_count, columns = _panel_grid_dimensions(
                len(projection_tables),
                prefer_single_row=prefer_single_row,
            )
            fig, axes = plt.subplots(
                rows_count,
                columns,
                figsize=_grid_figure_size(
                    len(projection_tables),
                    square_panels=True,
                    prefer_single_row=prefer_single_row,
                ),
                squeeze=False,
            )
            color_map, categories = _category_color_map(projection_tables, spec.color_column, axis_styles=axis_styles)
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
                    axis_styles=axis_styles,
                    rasterized=point_style.rasterized,
                    edgecolors=point_style.edgecolors,
                    linewidths=point_style.linewidths,
                )
                axis.set_title(
                    wrap_plot_title(compact_candidate_title(panel_title), width=22, max_lines=3),
                    pad=8,
                )
                axis.set_xlabel("Projection 1")
                axis.set_ylabel("Projection 2")
                _apply_axes_style(axis, grid=True, square=True)
                selected_rows, resolved_label_column, _, annotation_state = _resolve_annotation_rows(
                    context,
                    projection_rows,
                    spec=spec,
                )
                _draw_resolved_annotations(
                    axis,
                    context=context,
                    spec=spec,
                    rows=selected_rows,
                    resolved_x="x",
                    resolved_y="y",
                    resolved_label_column=resolved_label_column,
                    color_map=color_map,
                )
                plot_metadata.setdefault("reference_panels", {})[projection_id] = annotation_state
            grid_legend_bottom_margin = _add_figure_legends(
                fig,
                plt,
                plot_id=spec.plot_id,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_shape_column,
                axis_styles=axis_styles,
            )
    grid_legend_bottom_margin = float(locals().get("grid_legend_bottom_margin", 0.0))
    grid_legend_right_margin = float(locals().get("grid_legend_right_margin", 0.0))
    if spec.kind == "heatmap_grid":
        fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.08, wspace=0.30, hspace=0.48)
    elif spec.kind in {
        "projection_scatter",
        "projection_grid",
        "xy_scatter_grid",
        "paired_xy_scatter_grid",
        "categorical_count",
        "metric_panel_grid",
        "curve_grid",
    } and (grid_legend_bottom_margin > 0.0 or grid_legend_right_margin > 0.0):
        fig.tight_layout(
            **_tight_layout_kwargs(
                spec,
                legend_bottom=grid_legend_bottom_margin,
                legend_right=grid_legend_right_margin,
            )
        )
    else:
        fig.tight_layout(**_tight_layout_kwargs(spec, legend_bottom=0.0))

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
