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


def apply_axes_style(ax: Axes, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    if ygrid:
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
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
