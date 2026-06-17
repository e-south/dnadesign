"""Shared plotting style for DenseGen TFBS learnability review surfaces."""

from __future__ import annotations

from typing import Any

REVIEW_SQUARE_FIGSIZE = (7.2, 7.2)
REVIEW_WIDE_FIGSIZE = (9.2, 5.2)
REVIEW_STACKED_FIGSIZE = (7.2, 8.8)
REVIEW_MATRIX_FIGSIZE = (7.4, 5.8)
REVIEW_AXIS_LABEL_FONTSIZE = 15
REVIEW_TITLE_FONTSIZE = 15
REVIEW_TICK_LABEL_FONTSIZE = 15
REVIEW_LEGEND_FONTSIZE = 15

TARGET_METADATA_COLOR = "#0072B2"
SHUFFLED_CONTROL_COLOR = "#D55E00"
KNOWN_LABEL_REFERENCE_COLOR = "#009E73"
POOL_AVERAGE_COLOR = "#4D4D4D"
DEFAULT_LINE_COLOR = "#6B6B6B"

ROLE_PALETTE = {
    "positive": TARGET_METADATA_COLOR,
    "matched_null": SHUFFLED_CONTROL_COLOR,
    "null": SHUFFLED_CONTROL_COLOR,
}

ROLE_MARKERS = {
    "positive": "o",
    "matched_null": "s",
    "null": "s",
}
DEFAULT_LINE_MARKER = "o"


def role_color(role: object) -> str:
    """Return the stable review color for a label-source role."""

    return ROLE_PALETTE.get(str(role), DEFAULT_LINE_COLOR)


def role_marker(role: object) -> str:
    """Return the stable review marker for a label-source role."""

    return ROLE_MARKERS.get(str(role), DEFAULT_LINE_MARKER)


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
