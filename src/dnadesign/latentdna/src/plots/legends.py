"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/plots/legends.py

Shared legend helpers for static plot renderers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from ..metadata.axes import AxisStyle
from ..presentation.visual_style import (
    PLOT_FONT_FAMILY,
    PLOT_LEGEND_FONT_SIZE,
    TEXT_COLOR,
    humanize_display_text,
    legend_layout,
)
from .axes import axis_category_label


def style_legend(legend: Any) -> None:
    """Apply the LatentDNA static-plot legend style."""

    if legend is None:
        return
    title = legend.get_title()
    if title is not None:
        title.set_visible(False)
    for text in legend.get_texts():
        text.set_fontsize(PLOT_LEGEND_FONT_SIZE)
        text.set_color(TEXT_COLOR)
        text.set_fontfamily(PLOT_FONT_FAMILY)


def legend_handles(
    pyplot: Any,
    categories: list[str],
    color_map: dict[str, str],
    *,
    column: str | None = None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> list[Any]:
    """Build category legend handles using the same labels as plotted axes."""

    return [
        pyplot.Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=7.5,
            color=color_map[category],
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=axis_category_label(category, column=column, axis_styles=axis_styles),
        )
        for category in categories
    ]


def shape_legend_handles(pyplot: Any, categories: list[str], shape_map: dict[str, str]) -> list[Any]:
    """Build marker-shape legend handles for categorical shape encodings."""

    return [
        pyplot.Line2D(
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


def add_axis_legends(
    axis: Any,
    pyplot: Any,
    *,
    color_categories: list[str],
    color_map: dict[str, str],
    color_title: str | None,
    shape_categories: list[str],
    shape_map: dict[str, str],
    shape_title: str | None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> None:
    """Attach per-axis legends for small single-panel plots."""

    color_legend = None
    if color_categories and color_title is not None:
        color_legend = axis.legend(
            handles=legend_handles(
                pyplot,
                color_categories,
                color_map,
                column=color_title,
                axis_styles=axis_styles,
            ),
            frameon=False,
            loc="upper left",
        )
        style_legend(color_legend)
    if shape_categories and shape_title is not None:
        if color_legend is not None:
            axis.add_artist(color_legend)
        shape_legend = axis.legend(
            handles=shape_legend_handles(pyplot, shape_categories, shape_map),
            frameon=False,
            loc="lower right",
        )
        style_legend(shape_legend)


def add_figure_legends(
    figure: Any,
    pyplot: Any,
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
    """Attach bottom figure legends and return the required bottom margin."""

    legend_specs: list[list[Any]] = []
    if len(color_categories) > 1 and color_title is not None:
        legend_specs.append(
            legend_handles(pyplot, color_categories, color_map, column=color_title, axis_styles=axis_styles)
        )
    if len(shape_categories) > 1 and shape_title is not None:
        legend_specs.append(shape_legend_handles(pyplot, shape_categories, shape_map))
    if not legend_specs:
        return 0.0

    compact_bottom_legend_plot_ids = {
        "balanced_design_family_margin_gallery",
        "design_centroid_margin_gallery",
    }
    is_appendix_umap = plot_id == "appendix_umap_gallery"
    uses_compact_bottom_legend = plot_id in compact_bottom_legend_plot_ids
    legend_y = 0.036 if is_appendix_umap else 0.055 if uses_compact_bottom_legend else 0.012
    base_margin = 0.088 if is_appendix_umap else 0.095 if uses_compact_bottom_legend else 0.055
    reserved_bottom = 0.0
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
        legend = figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, layout.anchor_y),
            ncol=layout.columns,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.05,
            handletextpad=0.5,
        )
        style_legend(legend)
        reserved_bottom = max(reserved_bottom, layout.bottom_margin)
        legend_y = layout.anchor_y + layout.bottom_margin
    if is_appendix_umap:
        return min(max(reserved_bottom, 0.088), 0.16)
    if uses_compact_bottom_legend:
        return min(max(reserved_bottom, 0.11), 0.18)
    return min(max(legend_y + 0.014, 0.1), 0.40)


def add_side_figure_legends(
    figure: Any,
    pyplot: Any,
    *,
    color_categories: list[str],
    color_map: dict[str, str],
    color_title: str | None,
    shape_categories: list[str],
    shape_map: dict[str, str],
    shape_title: str | None,
    axis_styles: dict[str, AxisStyle] | None = None,
) -> float:
    """Attach right-side figure legends and return the required right margin."""

    legend_specs: list[list[Any]] = []
    if len(color_categories) > 1 and color_title is not None:
        legend_specs.append(
            legend_handles(pyplot, color_categories, color_map, column=color_title, axis_styles=axis_styles)
        )
    if len(shape_categories) > 1 and shape_title is not None:
        legend_specs.append(shape_legend_handles(pyplot, shape_categories, shape_map))
    if not legend_specs:
        return 0.0

    width, height = figure.get_size_inches()
    figure.set_size_inches(max(width + 2.6, 7.35), height, forward=True)
    for index, handles in enumerate(legend_specs):
        legend = figure.legend(
            handles=handles,
            loc="center right",
            bbox_to_anchor=(0.985, 0.5 - (index * 0.22)),
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.0,
            handletextpad=0.5,
        )
        style_legend(legend)
    return 0.30
