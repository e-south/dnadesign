"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/plots/panels.py

Shared panel-level plot helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from ..presentation.visual_style import SPINE_COLOR, TEXT_COLOR, wrap_plot_title
from .axes import apply_axes_style


def render_placeholder_panel(
    ax: Any,
    *,
    panel_title: str,
    message: str,
    detail: str | None = None,
    square: bool = False,
) -> None:
    """Render a contract-visible placeholder panel for empty plot data."""

    ax.cla()
    ax.set_title(wrap_plot_title(panel_title, width=24, max_lines=2), pad=8)
    apply_axes_style(ax, grid=False, square=square)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.text(
        0.5,
        0.58,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.0,
        color=TEXT_COLOR,
        fontweight="semibold",
    )
    if detail:
        ax.text(
            0.5,
            0.42,
            detail,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.0,
            color=SPINE_COLOR,
        )
