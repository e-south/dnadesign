"""
Shared visual style primitives for latentdna plots and notebooks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from textwrap import shorten, wrap

TEXT_COLOR = "#16202A"
GRID_COLOR = "#D5DCE4"
SPINE_COLOR = "#5C6874"
ZERO_LINE_COLOR = "#94A3B8"
PANEL_BACKGROUND_COLOR = "#FCFDFE"

PLOT_FONT_FAMILY = "DejaVu Sans"
NOTEBOOK_FONT_STACK = '"Avenir Next", Avenir, "Segoe UI", "Helvetica Neue", sans-serif'
NOTEBOOK_MONO_STACK = '"SFMono-Regular", Menlo, Consolas, "DejaVu Sans Mono", monospace'

DEFAULT_PLOT_PNG_DPI = 300
DEFAULT_NOTEBOOK_FIG_DPI = 220

PLOT_SUPTITLE_FONT_SIZE = 15.5
PLOT_TITLE_FONT_SIZE = 13.0
PLOT_LABEL_FONT_SIZE = 12.0
PLOT_TICK_FONT_SIZE = 11.0
PLOT_LEGEND_FONT_SIZE = 9.5
PLOT_LEGEND_TITLE_SIZE = 11.0

PUBLICATION_PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#111111",
]

_SEMANTIC_CATEGORY_COLORS = {
    "background": "#56B4E9",
    "background_only": "#56B4E9",
    "ethanol": "#E69F00",
    "ethanol_responsive": "#E69F00",
    "cipro": "#009E73",
    "cipro_responsive": "#009E73",
    "ciprofloxacin": "#009E73",
    "dual": "#CC79A7",
    "dual_and_responsive": "#CC79A7",
    "ethanol_ciprofloxacin": "#CC79A7",
    "control": "#111111",
}

_SEMANTIC_CATEGORY_PRIORITY = {
    "control": 0,
    "background": 1,
    "background_only": 1,
    "ethanol": 2,
    "ethanol_responsive": 2,
    "cipro": 3,
    "cipro_responsive": 3,
    "ciprofloxacin": 3,
    "dual": 4,
    "dual_and_responsive": 4,
    "ethanol_ciprofloxacin": 4,
}


@dataclass(frozen=True)
class ScatterStyle:
    point_size: float
    alpha: float
    edgecolors: str
    linewidths: float
    rasterized: bool


def normalize_category_key(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def ordered_categories(values: Iterable[str]) -> list[str]:
    unique = sorted({str(value) for value in values})
    return sorted(
        unique,
        key=lambda value: (_SEMANTIC_CATEGORY_PRIORITY.get(normalize_category_key(value), 99), value.casefold()),
    )


def categorical_color_map(categories: Iterable[str]) -> dict[str, str]:
    ordered = ordered_categories(categories)
    color_map: dict[str, str] = {}
    fallback_index = 0
    for category in ordered:
        semantic_color = _SEMANTIC_CATEGORY_COLORS.get(normalize_category_key(category))
        if semantic_color is not None:
            color_map[category] = semantic_color
            continue
        color_map[category] = PUBLICATION_PALETTE[fallback_index % len(PUBLICATION_PALETTE)]
        fallback_index += 1
    return color_map


def scatter_style(row_count: int) -> ScatterStyle:
    if row_count <= 250:
        return ScatterStyle(point_size=30.0, alpha=0.84, edgecolors="white", linewidths=0.32, rasterized=False)
    if row_count <= 1_000:
        return ScatterStyle(point_size=16.0, alpha=0.66, edgecolors="white", linewidths=0.16, rasterized=False)
    if row_count <= 5_000:
        return ScatterStyle(point_size=6.6, alpha=0.34, edgecolors="none", linewidths=0.0, rasterized=True)
    if row_count <= 20_000:
        return ScatterStyle(point_size=3.4, alpha=0.22, edgecolors="none", linewidths=0.0, rasterized=True)
    return ScatterStyle(point_size=1.7, alpha=0.15, edgecolors="none", linewidths=0.0, rasterized=True)


def wrap_plot_title(title: object, *, width: int = 28, max_lines: int = 2) -> str:
    text = " ".join(str(title or "").split())
    if not text:
        return ""
    lines = wrap(text, width=max(width, 10), break_long_words=False, break_on_hyphens=False)
    if max_lines <= 0 or len(lines) <= max_lines:
        return "\n".join(lines)
    visible = lines[:max_lines]
    trailing = " ".join(lines[max_lines - 1 :])
    visible[-1] = shorten(trailing, width=max(width, 10), placeholder="...")
    return "\n".join(visible)
