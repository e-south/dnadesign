"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/plots/layout.py

Shared subplot layout policy for LatentDNA plot renderers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math


def _prefer_single_row_panel_layout(plot_id: str | None, panel_count: int, *, configured: object = None) -> bool:
    return bool(configured) and 1 < panel_count <= 6


def _panel_grid_dimensions(panel_count: int, *, prefer_single_row: bool = False) -> tuple[int, int]:
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
    columns = min(4, max(1, int(math.ceil(math.sqrt(panel_count)))))
    rows = int(math.ceil(panel_count / columns))
    return rows, columns


def _grid_figure_size(panel_count: int, *, square_panels: bool, prefer_single_row: bool = False) -> tuple[float, float]:
    if panel_count <= 1:
        return (5.15, 5.0 if square_panels else 4.7)
    rows, columns = _panel_grid_dimensions(panel_count, prefer_single_row=prefer_single_row)
    if panel_count == 12 and square_panels:
        return (4.15 * columns, 4.7 * rows)
    panel_width = 3.55 if prefer_single_row and columns >= 4 else 4.15 if columns >= 4 else 4.3
    panel_height = 4.2 if square_panels and prefer_single_row else 4.35 if square_panels else 4.05
    return (panel_width * columns, panel_height * rows)


def metric_panel_grid_layout(
    plot_id: str | None,
    panel_count: int,
    *,
    prefer_single_row: bool = False,
    square_panels: bool | None = None,
    panel_width: float | None = None,
    panel_height: float | None = None,
    extra_width_per_column: float | None = None,
) -> tuple[int, int, tuple[float, float]]:
    resolved_square_panels = bool(square_panels)

    def _configured_size(
        rows: int,
        columns: int,
        fallback: tuple[float, float],
        *,
        default_extra_width_per_column: float | None = None,
    ) -> tuple[float, float]:
        width, height = fallback
        if panel_width is not None:
            width = float(panel_width) * columns
        if panel_height is not None:
            height = float(panel_height) * rows
        resolved_extra_width_per_column = extra_width_per_column
        if (
            resolved_extra_width_per_column is None
            and panel_width is None
            and panel_height is None
            and default_extra_width_per_column is not None
        ):
            resolved_extra_width_per_column = default_extra_width_per_column
        if resolved_extra_width_per_column is not None:
            width += float(resolved_extra_width_per_column) * columns
        return width, height

    if prefer_single_row and panel_count > 1:
        rows, columns = _panel_grid_dimensions(panel_count, prefer_single_row=True)
        figsize = _grid_figure_size(panel_count, square_panels=resolved_square_panels, prefer_single_row=True)
        return rows, columns, _configured_size(rows, columns, figsize)
    rows, columns = _panel_grid_dimensions(panel_count)
    figsize = _grid_figure_size(panel_count, square_panels=resolved_square_panels)
    return rows, columns, _configured_size(rows, columns, figsize)


def plot_tight_layout_kwargs(
    plot_id: str | None,
    *,
    legend_bottom: float,
    legend_right: float = 0.0,
    pad: float | None = None,
    h_pad: float | None = None,
    w_pad: float | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "pad": 0.95,
        "h_pad": 1.4,
        "w_pad": 0.95,
    }
    if pad is not None:
        kwargs["pad"] = float(pad)
    if h_pad is not None:
        kwargs["h_pad"] = float(h_pad)
    if w_pad is not None:
        kwargs["w_pad"] = float(w_pad)
    if legend_bottom > 0.0 or legend_right > 0.0:
        kwargs["rect"] = (0.0, legend_bottom, max(0.58, 1.0 - legend_right), 0.995)
    return kwargs
