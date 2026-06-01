"""Shared plotting style for DenseGen TFBS learnability review surfaces."""

from __future__ import annotations

from typing import Any

ROLE_PALETTE = {
    "positive": "#446A8C",
    "matched_null": "#8C4E4A",
}
DEFAULT_LINE_COLOR = "#6B6B6B"


def role_color(role: object) -> str:
    """Return the stable review color for an oracle role."""

    return ROLE_PALETTE.get(str(role), DEFAULT_LINE_COLOR)


def style_review_axis(ax: Any, *, x_grid: bool = True) -> None:
    """Apply the common peer-review plot style to a matplotlib axis."""

    ax.set_facecolor("#F5F6F7")
    ax.grid(axis="y", color="#D7DBDF", linewidth=0.8, alpha=0.9)
    if x_grid:
        ax.grid(axis="x", color="#ECEFF2", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA1A8")
    ax.spines["bottom"].set_color("#9AA1A8")
    ax.tick_params(axis="both", colors="#444B52", length=3, width=0.8, labelsize=8)
