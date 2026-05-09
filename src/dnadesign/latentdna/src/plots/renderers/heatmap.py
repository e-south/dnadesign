"""Heatmap renderers for static plot artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import ResolvedPlotSpec
from ...metadata_axes import AxisStyle
from ...visual_style import TEXT_COLOR, humanize_display_text, normalize_category_key, wrap_plot_title
from ...workspaces.loader import WorkspaceContext
from ..axes import (
    apply_axes_style,
    axis_category_label,
    explicit_axis_label,
    resolved_axis_label,
    style_compact_category_tick_labels,
)
from ..layout import _panel_grid_dimensions, _prefer_single_row_panel_layout
from ..tables import read_table_rows, require_row_columns, require_unique_grid_cell


@dataclass(frozen=True, slots=True)
class HeatmapRenderResult:
    """Rendered heatmap figure and any plot metadata emitted by the renderer."""

    figure: Any
    metadata: dict[str, object] = field(default_factory=dict)


def _ordered_heatmap_axis_values(rows: list[dict[str, object]], column: str, configured_order: list[str]) -> list[str]:
    observed = list(dict.fromkeys(str(row[column]) for row in rows))
    if not configured_order:
        return observed
    ordered = [value for value in configured_order if value in set(observed)]
    return ordered or observed


def heatmap_grid_from_rows(
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
    require_row_columns(
        rows,
        [row_column, column_column, value_column],
        context="heatmap grid",
    )
    row_values = _ordered_heatmap_axis_values(rows, row_column, row_order)
    column_values = _ordered_heatmap_axis_values(rows, column_column, column_order)
    row_index = {row_value: index for index, row_value in enumerate(row_values)}
    column_index = {column_value: index for index, column_value in enumerate(column_values)}
    grid = np.full((len(row_values), len(column_values)), np.nan, dtype=np.float32)
    seen_cells: dict[tuple[str, str], int] = {}
    for row_number, row in enumerate(rows, start=1):
        row_key = str(row[row_column])
        column_key = str(row[column_column])
        if row_key not in row_index or column_key not in column_index:
            continue
        require_unique_grid_cell(
            seen_cells,
            row_key=row_key,
            column_key=column_key,
            row_number=row_number,
            context="heatmap grid",
        )
        try:
            value = float(row[value_column])
        except (TypeError, ValueError) as exc:
            raise ContractViolationError(
                f"heatmap grid value column {value_column!r} must contain numeric values; "
                f"row {row_number} has {row[value_column]!r}"
            ) from exc
        grid[
            row_index[row_key],
            column_index[column_key],
        ] = value
    return grid, row_values, column_values


def _compact_sigma_variant_letter(value: object) -> str | None:
    match = re.search(r"\(([A-Za-z])\)\s*$", str(value or ""))
    if match is None:
        return None
    letter = match.group(1)
    if letter.casefold() not in {"b", "c", "d", "e", "f"}:
        return None
    return letter.upper()


def _compact_centroid_label(value: object) -> str | None:
    normalized = normalize_category_key(value)
    return {
        "background": "Bg",
        "background_only": "Bg",
        "ethanol": "Et",
        "cipro": "Cp",
        "ciprofloxacin": "Cp",
        "dual": "Du",
        "ethanol_ciprofloxacin": "Du",
    }.get(normalized)


def heatmap_color_params(
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


def render_heatmap_panel(
    axis: Any,
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
) -> Any:
    grid = np.asarray(grid, dtype=np.float32)
    image = axis.imshow(grid, cmap=cmap, norm=norm, aspect="equal" if square_cells else "auto")
    x_tick_labels = [
        _compact_sigma_variant_letter(value)
        if square_cells and str(column_column).endswith("variant") and _compact_sigma_variant_letter(value) is not None
        else _compact_centroid_label(value)
        if square_cells and str(column_column) == "centroid_label" and _compact_centroid_label(value) is not None
        else axis_category_label(value, column=column_column, axis_styles=axis_styles, compact=square_cells)
        for value in column_values
    ]
    y_tick_labels = [
        axis_category_label(value, column=row_column, axis_styles=axis_styles, compact=square_cells)
        for value in row_values
    ]
    axis.set_xticks(
        range(len(column_values)),
        x_tick_labels,
        rotation=0 if square_cells else 30,
        ha="center" if square_cells else "right",
    )
    if show_y_tick_labels:
        axis.set_yticks(range(len(row_values)), y_tick_labels)
    else:
        axis.set_yticks(range(len(row_values)), [])
        axis.tick_params(axis="y", length=0)
    axis.set_xlabel(resolved_axis_label(explicit_label=x_axis_label, fallback_label=column_column, width=20))
    axis.set_ylabel(
        resolved_axis_label(explicit_label=y_axis_label, fallback_label=row_column, width=20)
        if show_y_axis_label
        else ""
    )
    axis.set_title(wrap_plot_title(title, width=24), pad=8)
    finite = np.asarray(grid[np.isfinite(grid)], dtype=np.float32)
    contrast_midpoint = float(np.mean(finite)) if finite.size else 0.0
    cell_label_font_size = 6.8 if square_cells and str(column_column) == "centroid_label" else 9.2
    for row_index_value in range(len(row_values)):
        for column_index_value in range(len(column_values)):
            value = grid[row_index_value, column_index_value]
            if not np.isfinite(value):
                label = "NA"
                text_color = TEXT_COLOR
            else:
                label = f"{value:.2f}"
                text_color = "white" if float(value) >= contrast_midpoint else TEXT_COLOR
            axis.text(
                column_index_value,
                row_index_value,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=cell_label_font_size,
            )
    apply_axes_style(axis, grid=False)
    if square_cells:
        style_compact_category_tick_labels(axis, axis_name="x")
    if square_cells and show_y_tick_labels:
        style_compact_category_tick_labels(axis, axis_name="y")
    return image


def _heatmap_colorbar_label(spec: ResolvedPlotSpec, value_column: str) -> str:
    return explicit_axis_label(spec.colorbar_label, width=20) or humanize_display_text(value_column)


def render_heatmap_plot(
    context: WorkspaceContext,
    *,
    spec: ResolvedPlotSpec,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle],
) -> HeatmapRenderResult:
    if spec.enrichment_id is not None:
        table_path = context.output_root / "enrichments" / spec.enrichment_id / "table.parquet"
        if not table_path.exists():
            raise MissingArtifactError(f"enrichment artifact is missing for heatmap rendering: {spec.enrichment_id}")
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
    rows = read_table_rows(
        table_path,
        required_columns=(row_column, column_column, metric_column),
        artifact_label=f"{spec.kind} input table",
    )
    grid, row_values, column_values = heatmap_grid_from_rows(
        rows,
        row_column=row_column,
        column_column=column_column,
        value_column=metric_column,
        row_order=list(spec.row_order or []),
        column_order=list(spec.column_order or []),
    )
    cmap, norm = heatmap_color_params([grid], color_scale=spec.color_scale)
    fig, axis = pyplot.subplots(figsize=(2.2 + 1.35 * len(column_values), 1.7 + 1.05 * len(row_values)))
    image = render_heatmap_panel(
        axis,
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
    colorbar = fig.colorbar(image, ax=axis, label=_heatmap_colorbar_label(spec, metric_column))
    colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
    colorbar.set_label(_heatmap_colorbar_label(spec, metric_column), fontsize=11, color=TEXT_COLOR)
    return HeatmapRenderResult(figure=fig)


def _heatmap_grid_figure_size(
    spec: ResolvedPlotSpec,
    *,
    rows_count: int,
    columns: int,
    max_row_count: int,
    max_column_count: int,
) -> tuple[float, float]:
    if spec.square_panels:
        cell_size = 0.26 if spec.plot_id == "reference_to_plan_centroid_heatmap" else 0.46
        panel_width = max(1.72, 0.95 + (cell_size * max_column_count))
        panel_height = max(2.12, 1.0 + (cell_size * max_row_count))
        return (
            max(
                7.3 if spec.plot_id == "reference_to_plan_centroid_heatmap" else 6.0,
                (panel_width * columns) + (2.2 if spec.plot_id == "reference_to_plan_centroid_heatmap" else 0.95),
            ),
            max(3.05, (panel_height * rows_count) + 0.35),
        )
    return (
        max(4.2, 1.9 + (1.15 * max_column_count)) * columns,
        max(4.1, 1.6 + (0.9 * max_row_count)) * rows_count,
    )


def render_heatmap_grid_plot(
    context: WorkspaceContext,
    *,
    spec: ResolvedPlotSpec,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle],
) -> HeatmapRenderResult:
    heatmap_tables: list[tuple[str, np.ndarray, list[str], list[str]]] = []
    for scalar_id in spec.scalar_ids:
        table_path = context.output_root / "scalars" / scalar_id / "table.parquet"
        if not table_path.exists():
            raise MissingArtifactError(f"scalar artifact is missing for heatmap_grid rendering: {scalar_id}")
        rows = read_table_rows(
            table_path,
            required_columns=(spec.row_column, spec.column_column, spec.value_column or "metric_value"),
            artifact_label=f"heatmap_grid scalar {scalar_id!r}",
        )
        heatmap_tables.append(
            (
                scalar_id,
                *heatmap_grid_from_rows(
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
    cmap, norm = heatmap_color_params(grids, color_scale=spec.color_scale)
    prefer_single_row = _prefer_single_row_panel_layout(
        spec.plot_id,
        len(heatmap_tables),
        configured=spec.single_row_panels,
    )
    rows_count, columns = _panel_grid_dimensions(len(heatmap_tables), prefer_single_row=prefer_single_row)
    max_row_count = max(len(row_values) for _, _, row_values, _ in heatmap_tables)
    max_column_count = max(len(column_values) for _, _, _, column_values in heatmap_tables)
    fig, axes = pyplot.subplots(
        rows_count,
        columns,
        figsize=_heatmap_grid_figure_size(
            spec,
            rows_count=rows_count,
            columns=columns,
            max_row_count=max_row_count,
            max_column_count=max_column_count,
        ),
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
        image = render_heatmap_panel(
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
    colorbar_label = _heatmap_colorbar_label(spec, str(spec.value_column or "metric_value"))
    if spec.square_panels:
        if spec.plot_id == "reference_to_plan_centroid_heatmap":
            fig.subplots_adjust(left=0.115, right=0.96, wspace=0.16, bottom=0.28, top=0.82)
            from matplotlib.colorbar import ColorbarBase

            colorbar = ColorbarBase(
                fig.add_axes([0.36, 0.095, 0.38, 0.024]),
                cmap=cmap,
                norm=norm,
                orientation="horizontal",
            )
        else:
            fig.subplots_adjust(left=0.07, right=0.875, wspace=0.18, bottom=0.18, top=0.79)
            colorbar = fig.colorbar(
                image,
                cax=fig.add_axes([0.922, 0.19, 0.013, 0.58]),
                label=colorbar_label,
            )
    else:
        colorbar = fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            fraction=0.03,
            pad=0.03,
            label=colorbar_label,
        )
    colorbar.ax.tick_params(labelsize=10, colors=TEXT_COLOR)
    colorbar.set_label(colorbar_label, fontsize=11, color=TEXT_COLOR)
    return HeatmapRenderResult(figure=fig)
