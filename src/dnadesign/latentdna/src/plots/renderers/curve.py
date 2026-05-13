"""Curve renderers for static and notebook plot surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...contracts.plot import ResolvedPlotSpec
from ...visual_style import PUBLICATION_PALETTE, legend_layout, wrap_plot_title
from ...workspaces.loader import WorkspaceContext
from ..axes import apply_axes_style
from ..layout import _grid_figure_size, _panel_grid_dimensions, _prefer_single_row_panel_layout
from ..legends import style_legend
from ..render_state import LayoutReservation


@dataclass(frozen=True, slots=True)
class CurveRenderResult:
    """Rendered curve figure plus explicit layout reservations."""

    figure: Any
    metadata: dict[str, object] = field(default_factory=dict)
    layout_reservation: LayoutReservation = field(default_factory=LayoutReservation)


def render_curve_panel(
    ax: Any,
    *,
    reducer_id: str,
    summary: dict[str, object],
    panel_title: str,
    square: bool = False,
    show_legend: bool = True,
) -> None:
    """Render one explained-variance curve panel from a reducer summary."""

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
        style_legend(legend)
    apply_axes_style(ax, grid=True, square=square)


def _load_reducer_summary(context: WorkspaceContext, reducer_id: str) -> dict[str, object]:
    reducer_path = context.output_root / "reducers" / reducer_id / "summary.json"
    if not reducer_path.exists():
        raise MissingArtifactError(f"reducer artifact is missing for curve rendering: {reducer_id}")
    return json.loads(reducer_path.read_text(encoding="utf-8"))


def _render_curve(context: WorkspaceContext, spec: ResolvedPlotSpec, *, pyplot: Any) -> CurveRenderResult:
    assert spec.reducer_id is not None
    summary = _load_reducer_summary(context, spec.reducer_id)
    square_curve_panel = spec.plot_id == "representation_scree_diagnostic"
    figure, axis = pyplot.subplots(figsize=(5.5, 5.3 if square_curve_panel else 4.7))
    render_curve_panel(
        axis,
        reducer_id=str(spec.reducer_id),
        summary=summary,
        panel_title=spec.plot_id,
        square=square_curve_panel,
    )
    return CurveRenderResult(figure=figure)


def _render_curve_grid(context: WorkspaceContext, spec: ResolvedPlotSpec, *, pyplot: Any) -> CurveRenderResult:
    reducer_summaries = [(reducer_id, _load_reducer_summary(context, reducer_id)) for reducer_id in spec.reducer_ids]
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
    figure, axes = pyplot.subplots(
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
        render_curve_panel(
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
    legend = figure.legend(
        handles=[
            pyplot.Line2D(
                [],
                [],
                marker="o",
                linewidth=1.8,
                color=PUBLICATION_PALETTE[0],
                label=legend_labels[0],
            ),
            pyplot.Line2D(
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
    style_legend(legend)
    layout_reservation = LayoutReservation()
    layout_reservation.reserve_bottom(layout.bottom_margin)
    return CurveRenderResult(figure=figure, layout_reservation=layout_reservation)


def render_curve_plot(context: WorkspaceContext, spec: ResolvedPlotSpec, *, pyplot: Any) -> CurveRenderResult:
    """Render reducer-backed curve plot kinds."""

    if spec.kind == "curve":
        return _render_curve(context, spec, pyplot=pyplot)
    if spec.kind == "curve_grid":
        return _render_curve_grid(context, spec, pyplot=pyplot)
    raise ContractViolationError(f"curve renderer does not support plot kind: {spec.kind}")
