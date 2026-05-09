"""XY scatter renderers for table-backed static plot artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import ResolvedPlotSpec
from ...metadata_axes import AxisStyle
from ...visual_style import SPINE_COLOR, compact_candidate_title, humanize_display_text, scatter_style, wrap_plot_title
from ...workspaces.loader import WorkspaceContext
from ..annotation_rendering import draw_resolved_annotations
from ..annotations import empty_annotation_state, resolve_annotation_rows
from ..axes import apply_axes_style, resolved_axis_label
from ..layout import _grid_figure_size, _panel_grid_dimensions, _prefer_single_row_panel_layout
from ..legends import add_axis_legends, add_figure_legends, add_side_figure_legends
from ..panels import render_placeholder_panel
from ..render_state import LayoutReservation
from ..tables import numeric_table_columns, read_table_rows, secondary_numeric_column, table_artifact_path
from .scatter import (
    add_continuous_colorbar,
    add_zero_reference_lines,
    category_color_map,
    coerce_finite_float,
    color_series,
    continuous_color_encoding,
    effective_shape_column,
    scatter_axis_label,
    scatter_point_sizes,
    scatter_points,
    shape_marker_map,
)


@dataclass(frozen=True, slots=True)
class XYRenderResult:
    """Rendered XY figure, metadata, and explicit layout reservations."""

    figure: Any
    metadata: dict[str, object] = field(default_factory=dict)
    layout_reservation: LayoutReservation = field(default_factory=LayoutReservation)


def ordered_numeric_axes(
    table: pa.Table,
    *,
    x_column: str | None,
    y_column: str | None,
    value_column: str | None,
) -> tuple[str, str]:
    """Resolve XY axes from numeric schema columns, failing before row conversion."""

    numeric_columns = numeric_table_columns(table)
    if len(numeric_columns) < 2:
        raise ContractViolationError("scatter rendering requires at least two numeric columns")
    resolved_x = x_column or value_column or numeric_columns[0]
    if resolved_x not in numeric_columns:
        raise ContractViolationError(f"scatter x column is missing or non-numeric: {resolved_x!r}")
    resolved_y = y_column or secondary_numeric_column(table, primary=resolved_x)
    if resolved_y not in numeric_columns:
        raise ContractViolationError(f"scatter y column is missing or non-numeric: {resolved_y!r}")
    return resolved_x, resolved_y


def _xy_axis_labels(
    axis: Any,
    rows: list[dict[str, object]],
    *,
    spec: ResolvedPlotSpec,
    resolved_x: str,
    resolved_y: str,
) -> None:
    axis.set_xlabel(
        resolved_axis_label(
            explicit_label=spec.x_axis_label,
            fallback_label=scatter_axis_label(
                rows,
                resolved_column=resolved_x,
                display_column="x_display_name",
            ),
            width=28,
            max_lines=2,
        )
    )
    axis.set_ylabel(
        resolved_axis_label(
            explicit_label=spec.y_axis_label,
            fallback_label=scatter_axis_label(
                rows,
                resolved_column=resolved_y,
                display_column="y_display_name",
            ),
            width=28,
            max_lines=2,
        )
    )


def _finite_xy_rows(
    rows: list[dict[str, object]],
    *,
    resolved_x: str,
    resolved_y: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if coerce_finite_float(row.get(resolved_x)) is not None and coerce_finite_float(row.get(resolved_y)) is not None
    ]


def render_xy_panel(
    axis: Any,
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
    """Render one XY panel and return reference-annotation state."""

    finite_rows = _finite_xy_rows(rows, resolved_x=resolved_x, resolved_y=resolved_y)
    if not finite_rows:
        render_placeholder_panel(
            axis,
            panel_title=compact_candidate_title(panel_title),
            message="Margins unavailable",
            detail="No finite values in this snapshot",
            square=True,
        )
        _xy_axis_labels(axis, rows, spec=spec, resolved_x=resolved_x, resolved_y=resolved_y)
        return {}

    x_values = [float(row[resolved_x]) for row in finite_rows]
    y_values = [float(row[resolved_y]) for row in finite_rows]
    x_span = float(np.ptp(np.asarray(x_values, dtype=np.float64))) if x_values else 0.0
    y_span = float(np.ptp(np.asarray(y_values, dtype=np.float64))) if y_values else 0.0
    collapsed_panel = x_span <= 1e-12 and y_span <= 1e-12
    render_mode = spec.render_mode or "points"
    if collapsed_panel:
        centroid_x = x_values[0] if x_values else 0.0
        centroid_y = y_values[0] if y_values else 0.0
        point_style = scatter_style(len(rows))
        axis.scatter(
            [centroid_x],
            [centroid_y],
            c="#111111",
            s=max(point_style.point_size * 18.0, 90.0),
            alpha=0.92,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        axis.set_xlim(centroid_x - 0.055, centroid_x + 0.055)
        axis.set_ylim(centroid_y - 0.055, centroid_y + 0.055)
        axis.text(
            0.5,
            0.93,
            "Collapsed to one point",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9.0,
            color=SPINE_COLOR,
        )
    elif render_mode == "hexbin":
        axis.hexbin(
            x_values,
            y_values,
            gridsize=max(12, min(48, int(np.sqrt(len(finite_rows))) * 2)),
            cmap="cividis",
        )
    elif render_mode == "density_contour":
        colors, _ = color_series(
            finite_rows,
            spec.color_column,
            color_map=color_map if color_map else None,
            axis_styles=axis_styles,
        )
        bins = max(10, min(30, int(np.sqrt(len(finite_rows))) * 2))
        histogram, x_edges, y_edges = np.histogram2d(x_values, y_values, bins=bins)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        axis.contour(x_centers, y_centers, histogram.T, levels=4, cmap="cividis")
        density_style = scatter_style(len(finite_rows))
        axis.scatter(
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
        scatter_points(
            axis,
            finite_rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            color_column=spec.color_column,
            color_map=color_map,
            shape_column=effective_shape_column(spec),
            shape_map=shape_map,
            point_size=point_style.point_size,
            alpha=point_style.alpha,
            axis_styles=axis_styles,
            rasterized=point_style.rasterized,
            edgecolors=point_style.edgecolors,
            linewidths=point_style.linewidths,
        )
    add_zero_reference_lines(axis, x_values=x_values, y_values=y_values)
    _xy_axis_labels(axis, rows, spec=spec, resolved_x=resolved_x, resolved_y=resolved_y)
    axis.set_title(wrap_plot_title(compact_candidate_title(panel_title), width=22, max_lines=3), pad=8)
    apply_axes_style(axis, grid=True, square=True)
    annotation_rows = resolve_annotation_rows(context, finite_rows, spec=spec)
    draw_resolved_annotations(
        axis,
        context=context,
        spec=spec,
        rows=annotation_rows.selected_rows,
        resolved_x=resolved_x,
        resolved_y=resolved_y,
        resolved_label_column=annotation_rows.label_column,
        color_map=color_map,
    )
    return annotation_rows.state


def _render_single_xy(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> XYRenderResult:
    _, artifact_id, table_path = table_artifact_path(context, spec)
    table = pq.read_table(table_path)
    resolved_x, resolved_y = ordered_numeric_axes(
        table,
        x_column=spec.x_column,
        y_column=spec.y_column,
        value_column=spec.value_column,
    )
    rows = read_table_rows(table_path)
    figure, axis = pyplot.subplots(figsize=_grid_figure_size(1, square_panels=True))
    color_encoding = continuous_color_encoding(rows, spec, axis_styles=axis_styles)
    color_map, categories = (
        ({}, [])
        if color_encoding is not None
        else category_color_map([rows], spec.color_column, axis_styles=axis_styles)
    )
    effective_marker_column = effective_shape_column(spec)
    shape_map, shape_categories = shape_marker_map([rows], effective_marker_column)
    finite_rows = _finite_xy_rows(rows, resolved_x=resolved_x, resolved_y=resolved_y)
    if not finite_rows:
        render_placeholder_panel(
            axis,
            panel_title=spec.plot_id,
            message="Margins unavailable",
            detail="No finite values in this snapshot",
            square=True,
        )
        annotation_state = empty_annotation_state(context, spec=spec, error="no_finite_rows")
    else:
        point_style = scatter_style(len(finite_rows))
        point_sizes = scatter_point_sizes(
            finite_rows,
            size_column=spec.size_column,
            default_size=point_style.point_size,
            size_range=spec.size_range,
        )
        scatter_points(
            axis,
            finite_rows,
            resolved_x=resolved_x,
            resolved_y=resolved_y,
            color_column=spec.color_column,
            color_map=color_map,
            shape_column=effective_marker_column,
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
        add_zero_reference_lines(axis, x_values=x_values, y_values=y_values)
        _xy_axis_labels(axis, rows, spec=spec, resolved_x=resolved_x, resolved_y=resolved_y)
        axis.set_title(wrap_plot_title(spec.plot_id, width=24), pad=8)
        if spec.size_column is not None:
            axis.text(
                0.98,
                0.02,
                f"Point size: {humanize_display_text(spec.size_column)}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.8,
                color=SPINE_COLOR,
            )
        apply_axes_style(axis, grid=True, square=True)
        annotation_rows = resolve_annotation_rows(context, finite_rows, spec=spec)
        if spec.plot_id != "candidate_decision_frontier":
            draw_resolved_annotations(
                axis,
                context=context,
                spec=spec,
                rows=annotation_rows.selected_rows,
                resolved_x=resolved_x,
                resolved_y=resolved_y,
                resolved_label_column=annotation_rows.label_column,
                color_map=color_map,
                font_size=9.5,
                marker_size=128.0,
                marker="*",
            )
        annotation_state = annotation_rows.state

    metadata = {"reference_panels": {artifact_id or spec.plot_id: annotation_state}}
    layout_reservation = LayoutReservation()
    if color_encoding is not None:
        add_continuous_colorbar(figure, axis, spec=spec, color_encoding=color_encoding)
    elif (spec.render_mode or "points") == "points":
        if spec.plot_id == "candidate_decision_frontier" and shape_categories:
            layout_reservation.reserve_right(
                add_side_figure_legends(
                    figure,
                    pyplot,
                    color_categories=categories,
                    color_map=color_map,
                    color_title=spec.color_column,
                    shape_categories=shape_categories,
                    shape_map=shape_map,
                    shape_title=effective_marker_column,
                    axis_styles=axis_styles,
                )
            )
        else:
            add_axis_legends(
                axis,
                pyplot,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_marker_column,
                axis_styles=axis_styles,
            )
    return XYRenderResult(figure=figure, metadata=metadata, layout_reservation=layout_reservation)


def _render_xy_grid(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> XYRenderResult:
    scalar_tables: list[tuple[str, list[dict[str, object]], str, str]] = []
    for scalar_id in spec.scalar_ids:
        table_path = context.output_root / "scalars" / scalar_id / "table.parquet"
        if not table_path.exists():
            raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {scalar_id}")
        table = pq.read_table(table_path)
        resolved_x, resolved_y = ordered_numeric_axes(
            table,
            x_column=spec.x_column,
            y_column=spec.y_column,
            value_column=spec.value_column,
        )
        scalar_tables.append((scalar_id, read_table_rows(table_path), resolved_x, resolved_y))

    prefer_single_row = _prefer_single_row_panel_layout(
        spec.plot_id,
        len(scalar_tables),
        configured=spec.single_row_panels,
    )
    rows_count, columns = _panel_grid_dimensions(len(scalar_tables), prefer_single_row=prefer_single_row)
    figure, axes = pyplot.subplots(
        rows_count,
        columns,
        figsize=_grid_figure_size(len(scalar_tables), square_panels=True, prefer_single_row=prefer_single_row),
        squeeze=False,
    )
    color_map, categories = category_color_map(
        [rows for _, rows, _, _ in scalar_tables],
        spec.color_column,
        axis_styles=axis_styles,
    )
    effective_marker_column = effective_shape_column(spec)
    shape_map, shape_categories = shape_marker_map(
        [rows for _, rows, _, _ in scalar_tables],
        effective_marker_column,
    )
    titles = spec.panel_titles or [scalar_id for scalar_id, _, _, _ in scalar_tables]
    for axis in axes.ravel()[len(scalar_tables) :]:
        axis.axis("off")
    metadata: dict[str, object] = {"reference_panels": {}}
    for axis, (scalar_id, rows, resolved_x, resolved_y), panel_title in zip(
        axes.ravel(),
        scalar_tables,
        titles,
        strict=False,
    ):
        annotation_state = render_xy_panel(
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
        metadata["reference_panels"][scalar_id] = annotation_state

    layout_reservation = LayoutReservation()
    if (spec.render_mode or "points") == "points":
        layout_reservation.reserve_bottom(
            add_figure_legends(
                figure,
                pyplot,
                plot_id=spec.plot_id,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=effective_marker_column,
                axis_styles=axis_styles,
            )
        )
    return XYRenderResult(figure=figure, metadata=metadata, layout_reservation=layout_reservation)


def render_xy_plot(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> XYRenderResult:
    """Render static XY scatter plot kinds with explicit table contracts."""

    if spec.kind in {"distance_scatter", "xy_scatter"}:
        return _render_single_xy(context, spec, pyplot=pyplot, axis_styles=axis_styles)
    if spec.kind in {"xy_scatter_grid", "paired_xy_scatter_grid"}:
        return _render_xy_grid(context, spec, pyplot=pyplot, axis_styles=axis_styles)
    raise ContractViolationError(f"xy renderer does not support plot kind: {spec.kind}")
