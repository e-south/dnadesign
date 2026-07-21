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
from matplotlib.collections import QuadMesh

from .plot_definitions import PLOT_SPECS

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
    declared_title_size = figure._suptitle.get_fontsize() if figure._suptitle is not None else 13
    title_size = max(13, float(declared_title_size))
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
        figure.suptitle(
            title,
            x=0.5,
            y=1.11 if has_group_labels else 1.04,
            ha="center",
            fontweight="semibold",
            fontsize=title_size,
        )
        for axis in data_axes:
            _center_panel_title(axis)
    for axis in figure.axes:
        axis.set_facecolor("white")
        axis.set_axisbelow(True)
        is_data_axis = axis.get_label() != "<colorbar>"
        if is_data_axis and not _is_matrix_axis(axis):
            axis.grid(True, color="#e5e7eb", linewidth=0.65, alpha=0.7, zorder=0)
        for line in (*axis.get_xgridlines(), *axis.get_ygridlines()):
            line.set_zorder(0)
        axis.title.set_color("#111827")
        axis.xaxis.label.set_color("#111827")
        axis.yaxis.label.set_color("#111827")
        axis.tick_params(colors="#111827")
        for location, spine in axis.spines.items():
            spine.set_color("#6b7280")
            if is_data_axis and location in {"top", "right"}:
                spine.set_visible(False)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_color("#111827")
    for legend in figure.legends:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_color("#111827")
    if len(data_axes) > 1 and not figure.get_constrained_layout():
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(path, dpi=300, facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(figure)


def _is_matrix_axis(axis: plt.Axes) -> bool:
    return bool(axis.images) or any(isinstance(collection, QuadMesh) for collection in axis.collections)


def _center_panel_title(axis: plt.Axes) -> None:
    titles = [axis.get_title(loc=location) for location in ("left", "center", "right")]
    populated = [title for title in titles if title]
    if len(populated) > 1:
        raise ValueError("response-metastudy panel axis declares multiple titles.")
    if not populated:
        return
    fontsize = axis.title.get_fontsize()
    for location in ("left", "center", "right"):
        axis.set_title("", loc=location)
    axis.set_title(populated[0], loc="center", fontsize=fontsize)


__all__ = ["save_metastudy_figure"]
