"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/_style.py

Shared Matplotlib styling helpers for analysis plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure


def apply_axes_style(
    ax: Axes,
    *,
    ygrid: bool = True,
    xgrid: bool = False,
    tick_labelsize: int = 10,
    title_size: int = 13,
    label_size: int = 12,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_labelsize)
    ax.title.set_fontsize(title_size)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.set_axisbelow(True)
    if ygrid or xgrid:
        if ygrid and xgrid:
            grid_axis = "both"
        elif ygrid:
            grid_axis = "y"
        else:
            grid_axis = "x"
        ax.grid(axis=grid_axis, linestyle="-", linewidth=0.7, color="#d7d7d7", alpha=0.9)
    else:
        ax.grid(False)


def place_figure_caption(fig: Figure, text: str | None) -> None:
    caption = str(text or "").strip()
    if not caption:
        return
    fig.text(
        0.01,
        0.01,
        caption,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
