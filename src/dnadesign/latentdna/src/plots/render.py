"""
Artifact-driven plotting helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts.errors import ContractViolationError
from ..contracts.plot import SUPPORTED_PLOT_KINDS, ResolvedPlotSpec, metric_panel_uses_square_axes
from ..contracts.plot_semantics import PlotSemantics
from ..metadata_axes import axis_style_map_from_config
from ..visual_style import (
    DEFAULT_PLOT_PNG_DPI,
    PLOT_FONT_FAMILY,
    TEXT_COLOR,
)
from ..workspaces.loader import WorkspaceContext
from .layout import (
    metric_panel_grid_layout,
    plot_tight_layout_kwargs,
)
from .legends import (
    add_figure_legends as _add_figure_legends,
)
from .render_state import LayoutReservation
from .renderers.agreement import render_agreement_summary_plot, render_correspondence_heatmap_plot
from .renderers.categorical import render_categorical_count_plot
from .renderers.curve import render_curve_plot
from .renderers.distribution import render_distribution_plot
from .renderers.heatmap import render_heatmap_grid_plot, render_heatmap_plot
from .renderers.metric import (
    load_metric_panel_grid_input,
    render_metric_panel,
)
from .renderers.projection import render_projection_plot
from .renderers.scatter import (
    category_color_map as _category_color_map,
)
from .renderers.xy import render_xy_plot


def _pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = PLOT_FONT_FAMILY
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    return plt


def _tight_layout_kwargs(
    spec: ResolvedPlotSpec,
    *,
    legend_bottom: float,
    legend_right: float = 0.0,
) -> dict[str, object]:
    return plot_tight_layout_kwargs(spec.plot_id, legend_bottom=legend_bottom, legend_right=legend_right)


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
    layout_reservation = LayoutReservation()

    if spec.kind == "heatmap":
        result = render_heatmap_plot(context, spec=spec, pyplot=plt, axis_styles=axis_styles)
        fig = result.figure
        plot_metadata.update(result.metadata)
    elif spec.kind == "heatmap_grid":
        result = render_heatmap_grid_plot(context, spec=spec, pyplot=plt, axis_styles=axis_styles)
        fig = result.figure
        plot_metadata.update(result.metadata)
    elif spec.kind in {"distance_scatter", "xy_scatter", "xy_scatter_grid", "paired_xy_scatter_grid"}:
        xy_result = render_xy_plot(context, spec, pyplot=plt, axis_styles=axis_styles)
        fig = xy_result.figure
        plot_metadata.update(xy_result.metadata)
        layout_reservation.reserve_bottom(xy_result.layout_reservation.legend_bottom)
        layout_reservation.reserve_right(xy_result.layout_reservation.legend_right)
    elif spec.kind == "categorical_count":
        count_result = render_categorical_count_plot(context, spec, pyplot=plt, axis_styles=axis_styles)
        fig = count_result.figure
        plot_metadata.update(count_result.metadata)
        layout_reservation.reserve_bottom(count_result.layout_reservation.legend_bottom)
        layout_reservation.reserve_right(count_result.layout_reservation.legend_right)
    elif spec.kind == "metric_panel_grid":
        metric_input = load_metric_panel_grid_input(context, spec)
        rows = metric_input.rows
        resolved_spec = metric_input.resolved_spec
        panel_groups = metric_input.groups
        square_metric_panels = bool(spec.square_panels) or metric_panel_uses_square_axes(spec.plot_id)
        rows_count, columns, metric_figsize = metric_panel_grid_layout(
            spec.plot_id,
            len(panel_groups),
            prefer_single_row=bool(spec.single_row_panels),
            square_panels=square_metric_panels,
        )
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=metric_figsize,
            squeeze=False,
        )
        color_map, categories = _category_color_map([rows], spec.color_column, axis_styles=axis_styles)
        for axis in axes.ravel()[len(panel_groups) :]:
            axis.axis("off")
        for axis, panel_group in zip(axes.ravel(), panel_groups, strict=False):
            render_metric_panel(
                axis,
                rows=panel_group.rows,
                spec=resolved_spec,
                panel_title=panel_group.title,
                color_map=color_map,
                square=square_metric_panels,
                axis_styles=axis_styles,
            )
        plot_metadata["metric_columns"] = [panel_group.title for panel_group in panel_groups]
        if len(categories) > 1 and spec.color_column is not None:
            layout_reservation.reserve_bottom(
                _add_figure_legends(
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
            )
    elif spec.kind in {"distribution", "distribution_grid"}:
        distribution_result = render_distribution_plot(context, spec, pyplot=plt, axis_styles=axis_styles)
        fig = distribution_result.figure
        plot_metadata.update(distribution_result.metadata)
    elif spec.kind in {"curve", "curve_grid"}:
        curve_result = render_curve_plot(context, spec, pyplot=plt)
        fig = curve_result.figure
        plot_metadata.update(curve_result.metadata)
        layout_reservation.reserve_bottom(curve_result.layout_reservation.legend_bottom)
        layout_reservation.reserve_right(curve_result.layout_reservation.legend_right)
    elif spec.kind == "correspondence_heatmap":
        agreement_result = render_correspondence_heatmap_plot(context, spec, pyplot=plt)
        fig = agreement_result.figure
        plot_metadata.update(agreement_result.metadata)
    elif spec.kind in {"agreement_summary", "agreement_summary_grid"}:
        agreement_result = render_agreement_summary_plot(context, spec, pyplot=plt)
        fig = agreement_result.figure
        plot_metadata.update(agreement_result.metadata)
    elif spec.kind in {"projection_scatter", "projection_grid"}:
        projection_result = render_projection_plot(context, spec, pyplot=plt, axis_styles=axis_styles)
        fig = projection_result.figure
        plot_metadata.update(projection_result.metadata)
        layout_reservation.reserve_bottom(projection_result.layout_reservation.legend_bottom)
        layout_reservation.reserve_right(projection_result.layout_reservation.legend_right)
    else:
        raise ContractViolationError(f"plot renderer reached unsupported plot kind: {spec.kind}")
    if spec.kind == "heatmap_grid":
        if spec.square_panels:
            pass
        else:
            fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.08, wspace=0.30, hspace=0.48)
    elif (
        spec.kind
        in {
            "projection_scatter",
            "projection_grid",
            "xy_scatter_grid",
            "paired_xy_scatter_grid",
            "categorical_count",
            "metric_panel_grid",
            "curve_grid",
        }
        and layout_reservation.has_reservation
    ):
        fig.tight_layout(
            **_tight_layout_kwargs(
                spec,
                legend_bottom=layout_reservation.legend_bottom,
                legend_right=layout_reservation.legend_right,
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
