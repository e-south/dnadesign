"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/notebooks/browser_runtime_ui.py

Marimo UI helpers for generated LatentDNA browser notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable

import marimo as mo
import pandas as pd

from ..presentation.visual_style import (
    GRID_COLOR,
    NOTEBOOK_FONT_STACK,
    PANEL_BACKGROUND_COLOR,
    PLOT_FONT_FAMILY,
    PLOT_LABEL_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    PLOT_TICK_FONT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    SPINE_COLOR,
    TEXT_COLOR,
)


def notebook_theme():
    return mo.Html(
        f"""
        <style>
          .latentdna-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 5.5rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 650;
            letter-spacing: 0.02em;
            font-family: {NOTEBOOK_FONT_STACK};
            color: {TEXT_COLOR};
            background: rgba(226, 232, 240, 0.82);
          }}

          .latentdna-badge--primary {{
            background: rgba(59, 130, 246, 0.12);
          }}

          .latentdna-badge--appendix {{
            background: rgba(236, 201, 75, 0.22);
          }}
        </style>
        """
    )


def unique_in_order(values):
    seen = set()
    ordered = []
    for value in values:
        key = str(value or "Unsectioned").strip() or "Unsectioned"
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def option_key_for_value(options: dict[str, object], target_value: object) -> str | None:
    target_text = str(target_value)
    for key, value in options.items():
        if str(value) == target_text:
            return key
    return None


def labeled_options(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    normalized: list[tuple[str, object]] = []
    counts: dict[str, int] = {}
    for label, value in pairs:
        value_text = str(value).strip()
        base_label = str(label).strip() or value_text
        normalized.append((base_label, value))
        counts[base_label] = counts.get(base_label, 0) + 1

    options: dict[str, object] = {}
    for base_label, value in normalized:
        value_text = str(value).strip()
        if not value_text:
            continue
        label = base_label if counts[base_label] == 1 else f"{base_label} [{value_text}]"
        if label in options:
            if str(options[label]).strip() == value_text:
                continue
            suffix = 2
            while f"{label} #{suffix}" in options:
                suffix += 1
            label = f"{label} #{suffix}"
        options[label] = value
    return options


def resolve_labeled_option_card(
    cards: Iterable[dict[str, object]],
    selected_value: object,
    *,
    id_column: str = "plot_id",
    title_column: str = "title",
) -> dict[str, object] | None:
    """Resolve a Marimo dropdown selection against stable IDs and labels."""
    ordered_cards = [dict(card) for card in cards]
    if not ordered_cards:
        return None
    selected_text = str(selected_value or "").strip()
    if not selected_text:
        return ordered_cards[0]
    for card in ordered_cards:
        card_id = str(card.get(id_column) or "").strip()
        card_title = str(card.get(title_column) or "").strip()
        accepted = {value for value in (card_id, card_title) if value}
        if card_id and card_title:
            accepted.add(f"{card_title} [{card_id}]")
        if selected_text in accepted:
            return card
    return ordered_cards[0]


def table_from_records(
    records: pd.DataFrame | list[dict[str, object]],
    *,
    columns: list[str] | None = None,
    page_size: int | None = None,
):
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    if columns is not None:
        frame = frame.reindex(columns=columns)
    if page_size is None:
        return mo.ui.table(frame)
    return mo.ui.table(frame, page_size=page_size)


def key_value_table(
    rows: list[tuple[str, object]],
    *,
    field_name: str = "Field",
    value_name: str = "Value",
):
    normalized_rows = [{field_name: str(field), value_name: value} for field, value in rows]
    return table_from_records(
        normalized_rows,
        columns=[field_name, value_name],
        page_size=min(max(len(normalized_rows), 1), 12),
    )


def style_notebook_axes(ax, *, grid: bool = True, square: bool = False) -> None:
    ax.set_facecolor(PANEL_BACKGROUND_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(colors=TEXT_COLOR, labelsize=PLOT_TICK_FONT_SIZE, length=4.5, width=0.8, direction="out")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontsize(PLOT_LABEL_FONT_SIZE)
    ax.title.set_color(TEXT_COLOR)
    ax.title.set_fontsize(PLOT_TITLE_FONT_SIZE)
    ax.title.set_fontweight("semibold")
    ax.title.set_fontfamily(PLOT_FONT_FAMILY)
    ax.margins(x=0.04, y=0.05)
    if square:
        ax.set_box_aspect(1)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.75, alpha=0.58)
        ax.set_axisbelow(True)


def style_notebook_legend(legend) -> None:
    if legend is None:
        return
    title = legend.get_title()
    if title is not None:
        title.set_visible(False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
        text.set_fontsize(PLOT_LEGEND_FONT_SIZE)
        text.set_fontfamily(PLOT_FONT_FAMILY)
