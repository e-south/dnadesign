"""Publication typography and export for MSRB shadow figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIGURE_TITLE_SIZE = 18
PANEL_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 13
TICK_SIZE = 11
LEGEND_SIZE = 11


def style_axis(axis: plt.Axes, *, grid_axis: str) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=TICK_SIZE, width=0.8)
    axis.grid(axis=grid_axis, color="#D9DDE2", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(
        path,
        dpi=300,
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


__all__ = [
    "AXIS_LABEL_SIZE",
    "FIGURE_TITLE_SIZE",
    "LEGEND_SIZE",
    "PANEL_TITLE_SIZE",
    "TICK_SIZE",
    "save_figure",
    "style_axis",
]
