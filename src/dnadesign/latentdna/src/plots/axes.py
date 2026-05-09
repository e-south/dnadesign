"""Shared axis styling and label contracts for static plot renderers."""

from __future__ import annotations

from typing import Any

from ..metadata_axes import AxisStyle, axis_display_text
from ..visual_style import (
    GRID_COLOR,
    PANEL_BACKGROUND_COLOR,
    PLOT_FONT_FAMILY,
    PLOT_LABEL_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    SPINE_COLOR,
    TEXT_COLOR,
    humanize_display_text,
    wrap_plot_title,
)


def _axis_style(axis_styles: dict[str, AxisStyle] | None, column: str | None) -> AxisStyle | None:
    if column is None:
        return None
    return (axis_styles or {}).get(str(column))


def axis_category_label(
    value: object,
    *,
    column: str | None,
    axis_styles: dict[str, AxisStyle] | None = None,
    compact: bool = False,
) -> str:
    style = _axis_style(axis_styles, column)
    if style is not None:
        return axis_display_text(style, value, compact=compact)
    return humanize_display_text(value)


def apply_axes_style(axis: Any, *, grid: bool, square: bool = False) -> None:
    axis.set_facecolor(PANEL_BACKGROUND_COLOR)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)
    axis.spines["left"].set_linewidth(0.85)
    axis.spines["bottom"].set_linewidth(0.85)
    axis.tick_params(colors=TEXT_COLOR, labelsize=PLOT_TICK_FONT_SIZE, length=4.5, width=0.8, direction="out")
    axis.xaxis.label.set_color(TEXT_COLOR)
    axis.yaxis.label.set_color(TEXT_COLOR)
    axis.xaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    axis.yaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    axis.title.set_color(TEXT_COLOR)
    axis.title.set_fontsize(PLOT_TITLE_FONT_SIZE)
    axis.title.set_fontweight("semibold")
    axis.title.set_fontfamily(PLOT_FONT_FAMILY)
    axis.margins(x=0.04, y=0.05)
    if square:
        axis.set_box_aspect(1)
    if grid:
        axis.grid(True, color=GRID_COLOR, linewidth=0.75, alpha=0.58)
        axis.set_axisbelow(True)


def style_compact_category_tick_labels(axis: Any, *, axis_name: str = "x", font_size: float = 9.2) -> None:
    tick_labels = axis.get_xticklabels() if axis_name == "x" else axis.get_yticklabels()
    for label in tick_labels:
        label.set_fontsize(font_size)
        label.set_linespacing(0.92)
        label.set_rotation(0)
        label.set_rotation_mode("default")
        label.set_ha("center" if axis_name == "x" else "right")
        label.set_va("top" if axis_name == "x" else "center")


def wrapped_axis_label(value: object, *, width: int = 22, max_lines: int | None = 4) -> str:
    return wrap_plot_title(humanize_display_text(str(value)), width=width, max_lines=max_lines)


def _contains_math_text(value: object) -> bool:
    text = str(value or "")
    return "$" in text or "\\(" in text or "\\[" in text


def explicit_axis_label(value: object | None, *, width: int = 22, max_lines: int | None = 4) -> str | None:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return None
    if _contains_math_text(text):
        return text
    return wrap_plot_title(text, width=width, max_lines=max_lines)


def resolved_axis_label(
    *,
    explicit_label: object | None,
    fallback_label: object,
    width: int = 22,
    max_lines: int | None = 4,
) -> str:
    return explicit_axis_label(explicit_label, width=width, max_lines=max_lines) or wrapped_axis_label(
        fallback_label,
        width=width,
        max_lines=max_lines,
    )
