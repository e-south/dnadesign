"""Shared scatter encoding and drawing helpers for static plot renderers."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from ...contracts.plot import ResolvedPlotSpec
from ...metadata_axes import (
    AxisStyle,
    axis_color_map,
    legend_categories,
    normalize_axis_categories,
    normalize_axis_category,
    ordered_categories_for_axis,
)
from ...visual_style import PUBLICATION_PALETTE, TEXT_COLOR, ZERO_LINE_COLOR, humanize_display_text
from ..axes import explicit_axis_label, resolved_axis_label
from ..tables import require_row_columns

SHAPE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]


def compact_repeated_alpha_prefix(parts: list[str]) -> str | None:
    """Compact repeated alpha prefixes in list-valued categories."""

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


def category_key(value: object) -> str:
    """Normalize categorical plot values into stable legend/category keys."""

    if value is None:
        return "None"
    if isinstance(value, list | tuple | set):
        values = sorted(value, key=lambda part: str(part)) if isinstance(value, set) else value
        parts = [" ".join(str(part or "").split()) for part in values]
        parts = [part for part in parts if part]
        if not parts:
            return "None"
        compact = compact_repeated_alpha_prefix(parts)
        return compact or "+".join(parts)
    return " ".join(str(value).split()) or "None"


def coerce_finite_float(value: object) -> float | None:
    """Coerce finite scalar values for continuous encodings."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def axis_style(axis_styles: dict[str, AxisStyle] | None, column: str | None) -> AxisStyle | None:
    if column is None:
        return None
    return (axis_styles or {}).get(str(column))


def axis_category_value(
    row: dict[str, object],
    column: str,
    *,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> str:
    """Resolve a row value through configured metadata-axis normalization."""

    style = axis_style(axis_styles, column)
    if style is not None:
        return normalize_axis_category(style, row[column], row=row)
    return category_key(row[column])


def axis_categories(
    values: list[str],
    *,
    column: str | None,
    axis_styles: dict[str, AxisStyle] | None = None,
    legend_only: bool = False,
) -> list[str]:
    """Return stable category order for plot axes and legends."""

    style = axis_style(axis_styles, column)
    if style is not None:
        return legend_categories(style, values) if legend_only else ordered_categories_for_axis(style, values)
    return ordered_categories_for_axis(None, values)


def hue_option_type(spec: ResolvedPlotSpec, column: str | None) -> str | None:
    if column is None:
        return None
    for option in spec.hue_options:
        if option.column == column:
            return option.type
    return None


def hue_display_label(spec: ResolvedPlotSpec, column: str | None) -> str:
    if spec.colorbar_label:
        return str(spec.colorbar_label)
    if column is None:
        return "Value"
    for option in spec.hue_options:
        if option.column == column:
            return option.label
    return humanize_display_text(column)


def continuous_scatter_encoding(rows: list[dict], column: str | None) -> dict[str, object] | None:
    """Build continuous color encoding, or return None when not informative."""

    if column is None:
        return None
    require_row_columns(rows, [column], context="plot continuous color encoding")
    numeric = np.asarray(
        [
            coerce_finite_float(row.get(column)) if coerce_finite_float(row.get(column)) is not None else np.nan
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


def continuous_color_encoding(
    rows: list[dict],
    spec: ResolvedPlotSpec,
    *,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> dict[str, object] | None:
    """Resolve whether the selected hue should be rendered as continuous color."""

    column = spec.color_column
    hue_type = hue_option_type(spec, column)
    if hue_type in {"categorical", "binary", "ordinal"}:
        return None
    if hue_type == "continuous":
        return continuous_scatter_encoding(rows, column)
    style = axis_style(axis_styles, column)
    if style is not None and style.kind in {"categorical", "binary", "ordinal"}:
        return None
    return continuous_scatter_encoding(rows, column)


def add_continuous_colorbar(
    figure: Any,
    axis: Any,
    *,
    spec: ResolvedPlotSpec,
    color_encoding: dict[str, object],
) -> None:
    """Attach a continuous colorbar for a scatter plot."""

    from matplotlib.cm import ScalarMappable

    label = explicit_axis_label(hue_display_label(spec, spec.color_column), width=24, max_lines=3) or "Value"
    colorbar = figure.colorbar(
        ScalarMappable(norm=color_encoding["norm"], cmap=str(color_encoding["cmap"])),
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label=label,
    )
    colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
    colorbar.set_label(label, fontsize=11, color=TEXT_COLOR)


def category_color_map(
    row_groups: list[list[dict]],
    column: str | None,
    *,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build a categorical color map with fail-fast column validation."""

    if column is None:
        return {}, []
    flattened = [row for rows in row_groups for row in rows]
    require_row_columns(flattened, [column], context="plot color map")
    style = axis_style(axis_styles, column)
    if style is not None:
        values = normalize_axis_categories(
            style,
            [row[column] for row in flattened],
            rows=flattened,
        )
        categories = axis_categories(values, column=column, axis_styles=axis_styles, legend_only=True)
        color_map = axis_color_map(style, categories, fallback_palette=PUBLICATION_PALETTE)
        return color_map, categories
    categories = ordered_categories_for_axis(None, [category_key(row[column]) for row in flattened])
    color_map = axis_color_map(None, categories, fallback_palette=PUBLICATION_PALETTE)
    return color_map, categories


def color_series(
    rows: list[dict],
    column: str | None,
    *,
    color_map: dict[str, str] | None = None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> tuple[list[str], list[str]]:
    """Return one color per row for a categorical scatter encoding."""

    if column is None:
        return [PUBLICATION_PALETTE[0]] * len(rows), []
    require_row_columns(rows, [column], context="plot color encoding")
    resolved_map = color_map or category_color_map([rows], column, axis_styles=axis_styles)[0]
    style = axis_style(axis_styles, column)
    if style is not None:
        values = normalize_axis_categories(
            style,
            [row[column] for row in rows],
            rows=rows,
        )
        categories = axis_categories(list(resolved_map), column=column, axis_styles=axis_styles)
        return [resolved_map.get(value, "#9AA5B1") for value in values], categories
    categories = axis_categories(list(resolved_map), column=column, axis_styles=axis_styles)
    return [
        resolved_map.get(axis_category_value(row, column, axis_styles=axis_styles), "#9AA5B1") for row in rows
    ], categories


def scatter_point_sizes(
    rows: list[dict],
    *,
    size_column: str | None,
    default_size: float,
    size_range: tuple[float, float] | None,
) -> np.ndarray:
    """Return one point size per row for optional continuous size encoding."""

    base = np.full(len(rows), float(default_size), dtype=np.float64)
    if size_column is None:
        return base
    require_row_columns(rows, [size_column], context="plot size encoding")
    values = np.asarray(
        [
            coerce_finite_float(row.get(size_column))
            if coerce_finite_float(row.get(size_column)) is not None
            else np.nan
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


def shape_marker_map(row_groups: list[list[dict]], column: str | None) -> tuple[dict[str, str], list[str]]:
    """Build marker mapping for optional shape encoding."""

    if column is None:
        return {}, []
    flattened = [row for rows in row_groups for row in rows]
    require_row_columns(flattened, [column], context="plot shape encoding")
    categories = list(dict.fromkeys(category_key(row[column]) for row in flattened if category_key(row[column])))
    shape_map = {name: SHAPE_MARKERS[index % len(SHAPE_MARKERS)] for index, name in enumerate(categories)}
    return shape_map, categories


def effective_shape_column(spec: ResolvedPlotSpec) -> str | None:
    """Resolve shape encoding while avoiding redundant controls for hue-option plots."""

    if spec.hue_options and spec.kind in {
        "projection_scatter",
        "projection_grid",
        "xy_scatter",
        "xy_scatter_grid",
        "paired_xy_scatter_grid",
    }:
        return None
    return spec.shape_column


def scatter_points(
    axis: Any,
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
    """Draw scatter points while preserving missing continuous-hue rows."""

    sizes = point_sizes if point_sizes is not None else np.full(len(rows), float(point_size), dtype=np.float64)
    if shape_column is None:
        if continuous_color is not None:
            values = np.asarray(continuous_color["values"], dtype=np.float64)
            valid = np.isfinite(values)
            invalid = ~valid
            if np.any(valid):
                axis.scatter(
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
                axis.scatter(
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
            colors, _ = color_series(
                rows,
                color_column,
                color_map=color_map if color_map else None,
                axis_styles=axis_styles,
            )
            axis.scatter(
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
    require_row_columns(rows, [shape_column], context="plot shape encoding")
    for shape_category, marker in shape_map.items():
        group_indices = [index for index, row in enumerate(rows) if category_key(row[shape_column]) == shape_category]
        group_rows = [rows[index] for index in group_indices]
        if not group_rows:
            continue
        if continuous_color is not None:
            values = np.asarray(continuous_color["values"], dtype=np.float64)[group_indices]
            valid = np.isfinite(values)
            invalid = ~valid
            group_sizes = sizes[np.asarray(group_indices, dtype=np.int64)]
            if np.any(valid):
                axis.scatter(
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
                axis.scatter(
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
            colors, _ = color_series(
                group_rows,
                color_column,
                color_map=color_map if color_map else None,
                axis_styles=axis_styles,
            )
            axis.scatter(
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


def add_zero_reference_lines(axis: Any, *, x_values: list[float], y_values: list[float]) -> None:
    """Draw zero reference lines only when zero is within the observed range."""

    if x_values and min(x_values) < 0.0 < max(x_values):
        axis.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9, zorder=0)
    if y_values and min(y_values) < 0.0 < max(y_values):
        axis.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, linestyle="--", alpha=0.9, zorder=0)


def scatter_axis_label(
    rows: list[dict[str, object]],
    *,
    resolved_column: str,
    display_column: str,
) -> str:
    """Resolve a scatter axis label from display metadata when unambiguous."""

    labels = {str(row.get(display_column) or "").strip() for row in rows if str(row.get(display_column) or "").strip()}
    if len(labels) == 1:
        return humanize_display_text(next(iter(labels)))
    return humanize_display_text(resolved_column)


def resolved_scatter_axis_label(
    rows: list[dict[str, object]],
    *,
    explicit_label: object | None,
    resolved_column: str,
    display_column: str,
    width: int = 22,
) -> str:
    """Resolve an explicit-or-derived scatter axis label."""

    return resolved_axis_label(
        explicit_label=explicit_label,
        fallback_label=scatter_axis_label(
            rows,
            resolved_column=resolved_column,
            display_column=display_column,
        ),
        width=width,
    )
