"""Plot-review layout, axis-label, and hue helper primitives."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..visual_style import humanize_display_text, wrap_plot_title
from .browser_runtime_support import (
    category_values_for_legend,
    continuous_hue_render_params,
    normalize_categorical_hue_series,
)

SINGLE_ROW_PANEL_PLOT_IDS = frozenset(
    {
        "balanced_design_family_margin_gallery",
        "design_centroid_margin_gallery",
        "representation_scree_diagnostic",
        "appendix_umap_gallery",
    }
)


def prefer_single_row_panel_layout(plot_id: str | None, panel_count: int, *, configured: object = None) -> bool:
    """Return whether a compact one-row panel layout is valid for the plot."""

    if configured is not None:
        return bool(configured) and 1 < panel_count <= 6
    return bool(plot_id in SINGLE_ROW_PANEL_PLOT_IDS and 1 < panel_count <= 6)


def panel_grid_dimensions(panel_count: int, *, prefer_single_row: bool = False) -> tuple[int, int]:
    """Return notebook grid dimensions for live plot-review panels."""

    if panel_count <= 1:
        return 1, 1
    if prefer_single_row and panel_count <= 6:
        return 1, panel_count
    if panel_count == 12:
        return 2, 6
    if panel_count == 5:
        return 2, 3
    if panel_count == 6:
        return 2, 3
    if panel_count in {7, 8}:
        return 2, 4
    if panel_count == 4:
        return 2, 2
    columns = min(4, panel_count)
    rows = int(math.ceil(panel_count / columns))
    return rows, columns


def panel_figure_size(panel_count: int, *, prefer_single_row: bool = False) -> tuple[float, float]:
    """Return a stable figure size for live plot-review panel grids."""

    rows, columns = panel_grid_dimensions(panel_count, prefer_single_row=prefer_single_row)
    if panel_count == 12:
        return ((4.15 * columns) + 0.35, (5.3 * rows) + 0.2)
    panel_width = 3.55 if prefer_single_row and columns >= 4 else 4.15
    panel_height = 4.2 if prefer_single_row and panel_count > 1 else 4.35
    return ((panel_width * columns) + 0.35, (panel_height * rows) + 0.2)


def configured_hue_kinds(plot_spec: dict[str, object]) -> dict[str, str]:
    """Return configured hue-column kinds for a plot spec."""

    options = plot_spec.get("hue_options", [])
    return {
        str(option.get("column")): str(option.get("type"))
        for option in options
        if isinstance(option, dict) and option.get("column") and option.get("type")
    }


def scatter_axis_label(frame: pd.DataFrame, *, value_column: str, label_column: str) -> str:
    """Resolve a compact display label for one scatter axis."""

    if label_column in frame.columns:
        labels = {
            str(value).strip() for value in frame[label_column].dropna().astype(str).tolist() if str(value).strip()
        }
        if len(labels) == 1:
            return humanize_display_text(next(iter(labels)))
    return humanize_display_text(value_column)


def contains_math_text(value: object) -> bool:
    """Return whether a label already contains math markup."""

    text = str(value or "")
    return "$" in text or "\\(" in text or "\\[" in text


def resolved_axis_label(
    *,
    explicit_label: object | None,
    fallback_label: object,
    width: int = 28,
    max_lines: int | None = 2,
) -> str:
    """Resolve configured or fallback axis labels without breaking math markup."""

    text = " ".join(str(explicit_label or "").split()).strip()
    if text:
        if contains_math_text(text):
            return text
        return wrap_plot_title(text, width=width, max_lines=max_lines)
    return wrap_plot_title(humanize_display_text(str(fallback_label)), width=width, max_lines=max_lines)


def scatter_grid_axis_label_texts(
    plot_spec: dict[str, object],
    *,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> tuple[str, str]:
    """Resolve x/y labels for a live scatter grid panel."""

    return (
        resolved_axis_label(
            explicit_label=plot_spec.get("x_axis_label"),
            fallback_label=scatter_axis_label(frame, value_column=x_column, label_column="x_display_name"),
            width=28,
            max_lines=2,
        ),
        resolved_axis_label(
            explicit_label=plot_spec.get("y_axis_label"),
            fallback_label=scatter_axis_label(frame, value_column=y_column, label_column="y_display_name"),
            width=28,
            max_lines=2,
        ),
    )


def shared_numeric_bounds(frames: list[pd.DataFrame], hue_column: str) -> tuple[float | None, float | None]:
    """Return shared numeric bounds for a continuous hue column across panels."""

    values = [
        pd.to_numeric(frame[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        for frame in frames
        if hue_column in frame.columns
    ]
    if not values:
        return None, None
    combined = pd.concat(values, ignore_index=True)
    if combined.empty or combined.nunique() < 2:
        return None, None
    return float(combined.min()), float(combined.max())


def continuous_hue_params(frames: list[pd.DataFrame], hue_column: str) -> dict[str, object]:
    """Return color scale params for a continuous hue across panels."""

    values = [
        pd.to_numeric(frame[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        for frame in frames
        if hue_column in frame.columns
    ]
    combined = pd.concat(values, ignore_index=True) if values else pd.Series(dtype=float)
    if combined.empty or combined.nunique() < 2:
        return {"cmap": "viridis", "norm": None, "vmin": None, "vmax": None}
    return continuous_hue_render_params(hue_column, combined)


def continuous_hue_params_for_frame(
    frame: pd.DataFrame,
    hue_column: str,
    fallback: dict[str, object],
) -> dict[str, object]:
    """Return panel-specific continuous hue params when a panel has variation."""

    if hue_column not in frame.columns:
        return fallback
    values = pd.to_numeric(frame[hue_column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty or values.nunique() < 2:
        return fallback
    return continuous_hue_render_params(hue_column, values)


def categorical_hue_values(
    frames: list[pd.DataFrame],
    hue_column: str,
    *,
    axis_styles: dict[str, object] | None = None,
) -> list[str]:
    """Return ordered legend values for a categorical hue across panels."""

    categories = [
        str(value)
        for frame in frames
        if hue_column in frame.columns
        for value in categorical_hue_series(frame, hue_column, axis_styles=axis_styles).unique()
    ]
    return category_values_for_legend(categories, column=hue_column, axis_styles=axis_styles)


def categorical_hue_series(
    frame: pd.DataFrame,
    hue_column: str,
    *,
    axis_styles: dict[str, object] | None = None,
) -> pd.Series:
    """Return normalized categorical hue values for one frame."""

    return normalize_categorical_hue_series(
        hue_column,
        frame[hue_column],
        axis_styles=axis_styles,
        frame=frame,
    )


def plot_panel_title(plot_spec: dict[str, object], index: int, fallback: str) -> str:
    """Return a configured panel title or a fallback title."""

    titles = plot_spec.get("panel_titles", [])
    if isinstance(titles, list) and index < len(titles):
        return str(titles[index])
    return fallback
