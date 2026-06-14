"""Shared plotting style for DenseGen TFBS learnability review surfaces."""

from __future__ import annotations

from typing import Any

REVIEW_SQUARE_FIGSIZE = (7.2, 7.2)
REVIEW_WIDE_FIGSIZE = (9.2, 5.2)
REVIEW_STACKED_FIGSIZE = (7.2, 8.8)
REVIEW_MATRIX_FIGSIZE = (7.4, 5.8)
REVIEW_AXIS_LABEL_FONTSIZE = 15
REVIEW_TITLE_FONTSIZE = 20
REVIEW_TICK_LABEL_FONTSIZE = 15
REVIEW_LEGEND_FONTSIZE = 15

ROLE_PALETTE = {
    "positive": "#446A8C",
    "matched_null": "#8C4E4A",
}
DEFAULT_LINE_COLOR = "#6B6B6B"


def role_color(role: object) -> str:
    """Return the stable review color for a label-source role."""

    return ROLE_PALETTE.get(str(role), DEFAULT_LINE_COLOR)


def style_review_axis(
    ax: Any,
    *,
    grid: bool = True,
    x_grid: bool = True,
    square: bool = False,
) -> None:
    """Apply the common peer-review plot style to a matplotlib axis."""

    ax.set_facecolor("white")
    if grid:
        ax.grid(axis="y", color="#DDE2E6", linewidth=0.8, alpha=0.95)
    else:
        ax.grid(False)
    if grid and x_grid:
        ax.grid(axis="x", color="#EBEEF1", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA1A8")
    ax.spines["bottom"].set_color("#9AA1A8")
    ax.tick_params(
        axis="both",
        colors="#444B52",
        direction="out",
        length=4,
        width=0.8,
        labelsize=REVIEW_TICK_LABEL_FONTSIZE,
    )
    if square:
        try:
            ax.set_box_aspect(1.0)
        except (AttributeError, TypeError):
            ax.set_aspect("equal", adjustable="box")
