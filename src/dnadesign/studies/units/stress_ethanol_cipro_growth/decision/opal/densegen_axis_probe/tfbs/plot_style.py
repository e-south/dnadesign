"""Shared plotting style for DenseGen TFBS learnability review surfaces."""

from __future__ import annotations

from typing import Any

REVIEW_SQUARE_FIGSIZE = (7.2, 7.2)
REVIEW_WIDE_FIGSIZE = (9.2, 5.2)
REVIEW_STACKED_FIGSIZE = (7.2, 8.8)
REVIEW_MATRIX_FIGSIZE = (7.4, 5.8)

ROLE_PALETTE = {
    "positive": "#446A8C",
    "matched_null": "#8C4E4A",
}
DEFAULT_LINE_COLOR = "#6B6B6B"


def role_color(role: object) -> str:
    """Return the stable review color for an oracle role."""

    return ROLE_PALETTE.get(str(role), DEFAULT_LINE_COLOR)


def style_review_axis(
    ax: Any,
    *,
    grid: bool = True,
    x_grid: bool = True,
    square: bool = False,
) -> None:
    """Apply the common peer-review plot style to a matplotlib axis."""

    ax.set_facecolor("#F5F6F7")
    if grid:
        ax.grid(axis="y", color="#D7DBDF", linewidth=0.8, alpha=0.9)
    else:
        ax.grid(False)
    if grid and x_grid:
        ax.grid(axis="x", color="#ECEFF2", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA1A8")
    ax.spines["bottom"].set_color("#9AA1A8")
    ax.tick_params(axis="both", colors="#444B52", direction="out", length=3, width=0.8, labelsize=8)
    if square:
        try:
            ax.set_box_aspect(1.0)
        except (AttributeError, TypeError):
            ax.set_aspect("equal", adjustable="box")
