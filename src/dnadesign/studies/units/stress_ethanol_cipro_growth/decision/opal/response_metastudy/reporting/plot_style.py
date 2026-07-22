"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/plot_style.py

Publication styling shared by response-metastudy plot writers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt

from .plot_definitions import PLOT_SPECS
from .plot_style_primitives import (
    AXIS_LABEL_SIZE,
    FIGURE_TITLE_SIZE,
    LEGEND_TEXT_SIZE,
    PANEL_TITLE_SIZE,
    TICK_LABEL_SIZE,
    center_panel_title,
    style_axis,
    style_legend,
)

_TITLES = {spec.plot_id: spec.title for spec in PLOT_SPECS}


def save_metastudy_figure(figure: plt.Figure, path: Path) -> None:
    plot_id = path.stem
    if plot_id not in _TITLES:
        raise ValueError(f"unregistered response-metastudy plot path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor("white")
    data_axes = [axis for axis in figure.axes if axis.get_label() != "<colorbar>"]
    has_group_labels = any(
        str(text.get_gid() or "").startswith("column-group-label:") for axis in data_axes for text in axis.texts
    )
    title = fill(_TITLES[plot_id], width=max(48, int(figure.get_figwidth() * 7)))
    declared_title_size = figure._suptitle.get_fontsize() if figure._suptitle is not None else FIGURE_TITLE_SIZE
    title_size = max(FIGURE_TITLE_SIZE, float(declared_title_size))
    if len(data_axes) == 1:
        for location in ("left", "center", "right"):
            data_axes[0].set_title("", loc=location)
        data_axes[0].set_title(
            title,
            loc="center",
            fontweight="semibold",
            fontsize=title_size,
            pad=52 if has_group_labels else 12,
        )
    else:
        layout_engine = figure.get_layout_engine()
        if figure.get_constrained_layout() and not has_group_labels and layout_engine is not None:
            layout_parameters = layout_engine.get()
            if tuple(layout_parameters.get("rect", (0.0, 0.0, 1.0, 1.0))) == (0.0, 0.0, 1.0, 1.0):
                layout_engine.set(rect=(0.0, 0.0, 1.0, 0.90))
        constrained_title_y = 0.995 if figure.get_constrained_layout() and not has_group_labels else None
        figure.suptitle(
            title,
            x=0.5,
            y=constrained_title_y if constrained_title_y is not None else (1.065 if has_group_labels else 1.04),
            ha="center",
            fontweight="semibold",
            fontsize=title_size,
        )
        for axis in data_axes:
            center_panel_title(axis)
    for axis in figure.axes:
        is_data_axis = axis.get_label() != "<colorbar>"
        style_axis(axis, is_data_axis=is_data_axis, panel_count=len(data_axes))
    for legend in figure.legends:
        style_legend(legend)
    if len(data_axes) > 1 and not figure.get_constrained_layout():
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(figure)


__all__ = [
    "AXIS_LABEL_SIZE",
    "FIGURE_TITLE_SIZE",
    "LEGEND_TEXT_SIZE",
    "PANEL_TITLE_SIZE",
    "TICK_LABEL_SIZE",
    "save_metastudy_figure",
]
