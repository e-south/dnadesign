"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/plots/renderers/projection.py

Projection/UMAP renderers for static plot artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import ResolvedPlotSpec
from ...metadata.axes import AxisStyle
from ...presentation.visual_style import PLOT_TITLE_FONT_SIZE, compact_candidate_title, scatter_style, wrap_plot_title
from ...workspaces.loader import WorkspaceContext
from ..annotation_rendering import (
    add_annotation_colorbar,
    annotation_continuous_color_encoding,
    draw_resolved_annotations,
)
from ..annotations import resolve_annotation_rows
from ..axes import apply_axes_style
from ..layout import _grid_figure_size, _panel_grid_dimensions, _prefer_single_row_panel_layout
from ..legends import add_figure_legends, add_side_figure_legends
from ..render_state import LayoutReservation
from ..tables import read_table_rows
from .scatter import (
    add_continuous_colorbar,
    category_color_map,
    continuous_color_encoding,
    effective_shape_column,
    scatter_points,
    shape_marker_map,
)


@dataclass(frozen=True, slots=True)
class ProjectionRenderResult:
    """Rendered projection figure, metadata, and explicit layout reservations."""

    figure: Any
    metadata: dict[str, object] = field(default_factory=dict)
    layout_reservation: LayoutReservation = field(default_factory=LayoutReservation)


def _projection_rows(context: WorkspaceContext, projection_id: str) -> list[dict[str, object]]:
    projection_path = context.output_root / "projections" / projection_id / "coords.parquet"
    if not projection_path.exists():
        raise MissingArtifactError(f"projection artifact is missing for plot rendering: {projection_id}")
    return read_table_rows(
        projection_path,
        required_columns=("x", "y"),
        artifact_label=f"projection artifact {projection_id}",
    )


def _annotation_hue_column(spec: ResolvedPlotSpec) -> str | None:
    if spec.annotation is None or spec.annotation.hue_column is None:
        return None
    return str(spec.annotation.hue_column)


def _annotation_color_encoding(
    rows: list[dict[str, object]],
    spec: ResolvedPlotSpec,
    *,
    base_color_encoding: dict[str, object] | None = None,
    template: dict[str, object] | None = None,
) -> dict[str, object] | None:
    annotation_hue = _annotation_hue_column(spec)
    if annotation_hue is None:
        return None
    if base_color_encoding is not None:
        if spec.color_column != annotation_hue:
            raise ContractViolationError(
                "projection plots cannot render separate continuous colorbars for base scatter hue "
                f"{spec.color_column!r} and annotation hue {annotation_hue!r}; use the same column or split plots"
            )
        return annotation_continuous_color_encoding(rows, spec, template=base_color_encoding)
    return annotation_continuous_color_encoding(rows, spec, template=template)


def _render_projection_scatter(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> ProjectionRenderResult:
    if not spec.projection_ids:
        raise ContractViolationError("projection_scatter rendering requires one projection artifact")
    rows = _projection_rows(context, spec.projection_ids[0])
    color_encoding = continuous_color_encoding(rows, spec, axis_styles=axis_styles)
    color_map, categories = (
        ({}, [])
        if color_encoding is not None
        else category_color_map([rows], spec.color_column, axis_styles=axis_styles)
    )
    resolved_shape_column = effective_shape_column(spec)
    shape_map, shape_categories = shape_marker_map([rows], resolved_shape_column)
    figure, axis = pyplot.subplots(figsize=_grid_figure_size(1, square_panels=True))
    point_style = scatter_style(len(rows))
    scatter_points(
        axis,
        rows,
        resolved_x="x",
        resolved_y="y",
        color_column=spec.color_column,
        color_map=color_map,
        shape_column=resolved_shape_column,
        shape_map=shape_map,
        point_size=point_style.point_size,
        alpha=point_style.alpha,
        continuous_color=color_encoding,
        axis_styles=axis_styles,
        rasterized=point_style.rasterized,
        edgecolors=point_style.edgecolors,
        linewidths=point_style.linewidths,
    )
    axis.set_xlabel("Projection 1")
    axis.set_ylabel("Projection 2")
    panel_title = spec.panel_titles[0] if spec.panel_titles else spec.projection_ids[0]
    axis.set_title(wrap_plot_title(compact_candidate_title(panel_title), width=28), pad=8)
    apply_axes_style(axis, grid=True, square=True)

    annotation_rows = resolve_annotation_rows(context, rows, spec=spec)
    annotation_color_encoding = _annotation_color_encoding(
        annotation_rows.selected_rows,
        spec,
        base_color_encoding=color_encoding,
    )
    draw_resolved_annotations(
        axis,
        context=context,
        spec=spec,
        rows=annotation_rows.selected_rows,
        resolved_x="x",
        resolved_y="y",
        resolved_label_column=annotation_rows.label_column,
        color_map=color_map,
        annotation_color_encoding=annotation_color_encoding,
    )

    metadata = {"reference_panels": {spec.projection_ids[0]: annotation_rows.state}}
    layout_reservation = LayoutReservation()
    if color_encoding is not None:
        add_continuous_colorbar(figure, axis, spec=spec, color_encoding=color_encoding)
    elif annotation_color_encoding is not None:
        add_annotation_colorbar(figure, axis, spec=spec, color_encoding=annotation_color_encoding)
    elif len(categories) + len(shape_categories) > 8:
        layout_reservation.reserve_right(
            add_side_figure_legends(
                figure,
                pyplot,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=shape_categories,
                shape_map=shape_map,
                shape_title=resolved_shape_column,
                axis_styles=axis_styles,
            )
        )
    else:
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
                shape_title=resolved_shape_column,
                single_row=False,
                axis_styles=axis_styles,
            )
        )
    return ProjectionRenderResult(
        figure=figure,
        metadata=metadata,
        layout_reservation=layout_reservation,
    )


def _render_projection_grid(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> ProjectionRenderResult:
    projection_tables = [_projection_rows(context, projection_id) for projection_id in spec.projection_ids]
    prefer_single_row = _prefer_single_row_panel_layout(
        spec.plot_id,
        len(projection_tables),
        configured=spec.single_row_panels,
    )
    rows_count, columns = _panel_grid_dimensions(
        len(projection_tables),
        prefer_single_row=prefer_single_row,
    )
    figure, axes = pyplot.subplots(
        rows_count,
        columns,
        figsize=_grid_figure_size(
            len(projection_tables),
            square_panels=True,
            prefer_single_row=prefer_single_row,
        ),
        squeeze=False,
    )
    color_map, categories = category_color_map(projection_tables, spec.color_column, axis_styles=axis_styles)
    resolved_shape_column = effective_shape_column(spec)
    shape_map, shape_categories = shape_marker_map(projection_tables, resolved_shape_column)
    titles = spec.panel_titles or list(spec.projection_ids)
    for axis in axes.ravel()[len(projection_tables) :]:
        axis.axis("off")

    metadata: dict[str, object] = {"reference_panels": {}}
    annotation_rows_by_projection = [resolve_annotation_rows(context, rows, spec=spec) for rows in projection_tables]
    global_annotation_color_encoding = _annotation_color_encoding(
        [row for annotation_rows in annotation_rows_by_projection for row in annotation_rows.selected_rows],
        spec,
    )
    for axis_index, (axis, projection_rows, projection_id, panel_title, annotation_rows) in enumerate(
        zip(
            axes.ravel(),
            projection_tables,
            spec.projection_ids,
            titles,
            annotation_rows_by_projection,
            strict=False,
        )
    ):
        point_style = scatter_style(len(projection_rows))
        scatter_points(
            axis,
            projection_rows,
            resolved_x="x",
            resolved_y="y",
            color_column=spec.color_column,
            color_map=color_map,
            shape_column=resolved_shape_column,
            shape_map=shape_map,
            point_size=point_style.point_size,
            alpha=point_style.alpha,
            axis_styles=axis_styles,
            rasterized=point_style.rasterized,
            edgecolors=point_style.edgecolors,
            linewidths=point_style.linewidths,
        )
        is_appendix_umap = spec.plot_id == "appendix_umap_gallery"
        title_width = 28 if is_appendix_umap else 22
        axis.set_title(
            wrap_plot_title(compact_candidate_title(panel_title), width=title_width, max_lines=3),
            pad=5 if is_appendix_umap else 8,
        )
        axis.set_xlabel("Projection 1")
        axis.set_ylabel("Projection 2")
        apply_axes_style(axis, grid=True, square=True)
        if is_appendix_umap:
            axis.title.set_fontsize(PLOT_TITLE_FONT_SIZE - 1.25)
            axis.title.set_linespacing(1.0)
        panel_annotation_color_encoding = (
            _annotation_color_encoding(
                annotation_rows.selected_rows,
                spec,
                template=global_annotation_color_encoding,
            )
            if global_annotation_color_encoding is not None
            else None
        )
        draw_resolved_annotations(
            axis,
            context=context,
            spec=spec,
            rows=annotation_rows.selected_rows,
            resolved_x="x",
            resolved_y="y",
            resolved_label_column=annotation_rows.label_column,
            color_map=color_map,
            annotation_color_encoding=panel_annotation_color_encoding,
        )
        metadata["reference_panels"][projection_id] = annotation_rows.state

    layout_reservation = LayoutReservation()
    if global_annotation_color_encoding is not None:
        add_annotation_colorbar(
            figure,
            axes.ravel()[: len(projection_tables)].tolist(),
            spec=spec,
            color_encoding=global_annotation_color_encoding,
        )
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
            shape_title=resolved_shape_column,
            axis_styles=axis_styles,
        )
    )
    return ProjectionRenderResult(
        figure=figure,
        metadata=metadata,
        layout_reservation=layout_reservation,
    )


def render_projection_plot(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> ProjectionRenderResult:
    """Render projection scatter/grid plots with explicit x/y table contracts."""

    if spec.kind == "projection_scatter":
        return _render_projection_scatter(context, spec, pyplot=pyplot, axis_styles=axis_styles)
    if spec.kind == "projection_grid":
        return _render_projection_grid(context, spec, pyplot=pyplot, axis_styles=axis_styles)
    raise ContractViolationError(f"projection renderer received unsupported plot kind: {spec.kind}")
