"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/plots/renderers/categorical.py

Categorical count renderer for static plot artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import ResolvedPlotSpec
from ...metadata.axes import AxisStyle
from ...presentation.visual_style import PUBLICATION_PALETTE, TEXT_COLOR, humanize_display_text, wrap_plot_title
from ...workspaces.loader import WorkspaceContext
from ..axes import apply_axes_style
from ..layout import _grid_figure_size, _panel_grid_dimensions
from ..legends import add_figure_legends
from ..render_state import LayoutReservation
from ..tables import read_table_rows
from .scatter import axis_category_value, category_color_map, coerce_finite_float


@dataclass(frozen=True, slots=True)
class CategoricalCountRenderResult:
    """Rendered categorical-count figure and explicit layout reservations."""

    figure: Any
    metadata: dict[str, object] = field(default_factory=dict)
    layout_reservation: LayoutReservation = field(default_factory=LayoutReservation)


def _wrapped_tick_label(value: object, *, width: int = 16, max_lines: int | None = None) -> str:
    return wrap_plot_title(humanize_display_text(str(value)), width=width, max_lines=max_lines)


def _finite_count_value(row: dict[str, object], *, column: str, row_index: int) -> float:
    value = coerce_finite_float(row.get(column))
    if value is None:
        raise ContractViolationError(
            f"categorical_count value column {column!r} must be finite numeric; row {row_index} has {row.get(column)!r}"
        )
    return value


def _panel_values(rows: list[dict[str, object]], spec: ResolvedPlotSpec) -> list[str | None]:
    if spec.panel_column is None:
        return [None]
    return list(dict.fromkeys(str(row[spec.panel_column]) for row in rows))


def _count_grid_shape(panel_count: int, *, square_count_panels: bool, single_row_panels: bool) -> tuple[int, int]:
    if single_row_panels and panel_count <= 6:
        return 1, panel_count
    if square_count_panels and panel_count <= 3:
        return 1, panel_count
    if panel_count <= 2:
        return panel_count, 1
    return _panel_grid_dimensions(panel_count)


def _count_grid_size(
    panel_count: int,
    rows_count: int,
    columns: int,
    *,
    square_count_panels: bool,
) -> tuple[float, float]:
    if square_count_panels and rows_count == 1:
        return (4.8 * columns) + 0.45, 4.9
    if square_count_panels:
        width, height = _grid_figure_size(panel_count, square_panels=True)
        return width + (0.55 * columns), height
    return 6.6 * columns, 5.8 * rows_count


def _render_count_panel(
    axis: Any,
    panel_rows: list[dict[str, object]],
    *,
    spec: ResolvedPlotSpec,
    panel_value: str | None,
    color_map: dict[str, str],
    square_count_panels: bool,
    axis_styles: dict[str, AxisStyle] | None,
) -> None:
    if panel_rows and "order" in panel_rows[0]:
        panel_rows = sorted(panel_rows, key=lambda row: float(row.get("order", 0)))
    if spec.color_column is not None:
        bar_colors = [
            color_map[axis_category_value(row, spec.color_column, axis_styles=axis_styles)] for row in panel_rows
        ]
    else:
        bar_colors = [PUBLICATION_PALETTE[0]] * len(panel_rows)

    assert spec.value_column is not None
    values = [
        _finite_count_value(row, column=spec.value_column, row_index=row_index)
        for row_index, row in enumerate(panel_rows)
    ]
    y_positions = np.arange(len(panel_rows), dtype=float)
    axis.barh(
        y_positions,
        values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.92,
    )
    tick_width = 18 if len(panel_rows) >= 4 else 22
    axis.set_yticks(
        y_positions,
        [_wrapped_tick_label(row[spec.column_column], width=tick_width, max_lines=3) for row in panel_rows],
    )
    axis.tick_params(axis="y", labelsize=9.4 if len(panel_rows) >= 4 else 10.2, pad=4)
    axis.invert_yaxis()
    show_as_percent = spec.value_column in {"fraction", "percent"} or all(0.0 <= value <= 1.0 for value in values)
    axis.set_xlabel("Percent of N" if show_as_percent else humanize_display_text(spec.value_column or "row_count"))
    axis.set_ylabel("")
    denominator_values = {int(float(row["denominator"])) for row in panel_rows if row.get("denominator") is not None}
    panel_title = wrap_plot_title(str(panel_value) if panel_value is not None else spec.plot_id, width=24)
    if len(denominator_values) == 1:
        panel_title = f"{panel_title}\nN = {next(iter(denominator_values)):,}"
    axis.set_title(panel_title, pad=8)
    max_value = max(values, default=0.0)
    axis.set_xlim(0, max_value * 1.12 if max_value > 0 else 1.0)
    if show_as_percent:
        from matplotlib.ticker import PercentFormatter

        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0 if max_value <= 1.0 else 100.0))
    for row_index, (y_position, value) in enumerate(zip(y_positions, values, strict=False)):
        count_text = None
        if panel_rows[row_index].get("count") is not None:
            count_text = f"{int(float(panel_rows[row_index]['count'])):,}"
        percent_text = None
        if panel_rows[row_index].get("percent") is not None:
            percent_text = f"{float(panel_rows[row_index]['percent']):.1f}%"
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
    apply_axes_style(axis, grid=True, square=square_count_panels)


def render_categorical_count_plot(
    context: WorkspaceContext,
    spec: ResolvedPlotSpec,
    *,
    pyplot: Any,
    axis_styles: dict[str, AxisStyle] | None,
) -> CategoricalCountRenderResult:
    """Render a categorical-count plot with schema-first contracts."""

    if spec.kind != "categorical_count":
        raise ContractViolationError(f"categorical-count renderer does not support plot kind: {spec.kind}")
    assert spec.scalar_id is not None
    table_path = context.output_root / "scalars" / spec.scalar_id / "table.parquet"
    if not table_path.exists():
        raise MissingArtifactError(f"scalar artifact is missing for plot rendering: {spec.scalar_id}")
    required_columns = [spec.row_column, spec.column_column, spec.value_column, spec.color_column]
    if spec.panel_column is not None:
        required_columns.append(spec.panel_column)
    rows = read_table_rows(
        table_path,
        required_columns=required_columns,
        artifact_label=f"categorical_count scalar {spec.scalar_id}",
    )
    if not rows:
        raise ContractViolationError("categorical_count rendering requires at least one row")

    panel_values = _panel_values(rows, spec)
    square_count_panels = bool(spec.square_panels)
    rows_count, columns = _count_grid_shape(
        len(panel_values),
        square_count_panels=square_count_panels,
        single_row_panels=bool(spec.single_row_panels),
    )
    figure, axes = pyplot.subplots(
        rows_count,
        columns,
        figsize=_count_grid_size(
            len(panel_values),
            rows_count,
            columns,
            square_count_panels=square_count_panels,
        ),
        squeeze=False,
    )
    color_map, categories = category_color_map([rows], spec.color_column, axis_styles=axis_styles)
    for axis in axes.ravel()[len(panel_values) :]:
        axis.axis("off")
    for axis, panel_value in zip(axes.ravel(), panel_values, strict=False):
        panel_rows = (
            [row for row in rows if str(row[spec.panel_column]) == panel_value]
            if panel_value is not None and spec.panel_column is not None
            else rows
        )
        _render_count_panel(
            axis,
            panel_rows,
            spec=spec,
            panel_value=panel_value,
            color_map=color_map,
            square_count_panels=square_count_panels,
            axis_styles=axis_styles,
        )

    layout_reservation = LayoutReservation()
    if len(categories) > 1 and spec.color_column is not None:
        layout_reservation.reserve_bottom(
            add_figure_legends(
                figure,
                pyplot,
                plot_id=spec.plot_id,
                color_categories=categories,
                color_map=color_map,
                color_title=spec.color_column,
                shape_categories=[],
                shape_map={},
                shape_title=None,
                axis_styles=axis_styles,
            )
        )
    return CategoricalCountRenderResult(figure=figure, layout_reservation=layout_reservation)
