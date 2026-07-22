"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plot_style_primitives.py

Axis and legend styling primitives for response-metastudy figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.legend import Legend

FIGURE_TITLE_SIZE = 18
PANEL_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_TEXT_SIZE = 11


def style_axis(axis: plt.Axes, *, is_data_axis: bool, panel_count: int) -> None:
    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    if is_data_axis and not _is_matrix_axis(axis):
        axis.grid(True, color="#e5e7eb", linewidth=0.65, alpha=0.7, zorder=0)
    for line in (*axis.get_xgridlines(), *axis.get_ygridlines()):
        line.set_zorder(0)
    axis.title.set_color("#111827")
    if is_data_axis and panel_count > 1:
        axis.title.set_fontsize(max(PANEL_TITLE_SIZE, axis.title.get_fontsize()))
    axis.xaxis.label.set_color("#111827")
    axis.yaxis.label.set_color("#111827")
    axis.xaxis.label.set_fontsize(max(AXIS_LABEL_SIZE, axis.xaxis.label.get_fontsize()))
    axis.yaxis.label.set_fontsize(max(AXIS_LABEL_SIZE, axis.yaxis.label.get_fontsize()))
    axis.tick_params(colors="#111827", labelsize=TICK_LABEL_SIZE)
    for location, spine in axis.spines.items():
        spine.set_color("#6b7280")
        if is_data_axis and location in {"top", "right"}:
            spine.set_visible(False)
    legend = axis.get_legend()
    if legend is not None:
        style_legend(legend)


def style_legend(legend: Legend) -> None:
    legend.set_frame_on(False)
    for text in legend.get_texts():
        text.set_color("#111827")
        text.set_fontsize(max(LEGEND_TEXT_SIZE, text.get_fontsize()))
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(max(LEGEND_TEXT_SIZE, legend.get_title().get_fontsize()))


def center_panel_title(axis: plt.Axes) -> None:
    titles = [axis.get_title(loc=location) for location in ("left", "center", "right")]
    populated = [title for title in titles if title]
    if len(populated) > 1:
        raise ValueError("response-metastudy panel axis declares multiple titles.")
    if not populated or titles[1]:
        return
    fontsize = axis.title.get_fontsize()
    for location in ("left", "center", "right"):
        axis.set_title("", loc=location)
    axis.set_title(populated[0], loc="center", fontsize=fontsize)


def _is_matrix_axis(axis: plt.Axes) -> bool:
    return bool(axis.images) or any(isinstance(collection, QuadMesh) for collection in axis.collections)


__all__ = [
    "AXIS_LABEL_SIZE",
    "FIGURE_TITLE_SIZE",
    "LEGEND_TEXT_SIZE",
    "PANEL_TITLE_SIZE",
    "TICK_LABEL_SIZE",
    "center_panel_title",
    "style_axis",
    "style_legend",
]
