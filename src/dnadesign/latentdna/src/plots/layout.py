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

from ..contracts.plot import metric_panel_uses_square_axes

_SINGLE_ROW_PANEL_PLOT_IDS = frozenset(
    {
        "balanced_design_family_margin_gallery",
        "design_centroid_margin_gallery",
        "representation_scree_diagnostic",
        "appendix_umap_gallery",
    }
)

_HORIZONTAL_GROUPED_METRIC_PLOT_IDS: frozenset[str] = frozenset(
    {
        "candidate_x_selection_scorecard",
        "design_structure_summary",
        "reference_alignment_summary",
        "reference_standard_strength_audit",
        "representation_health_summary",
    }
)

_FULL_FORMULA_XY_GRID_PLOT_IDS: frozenset[str] = frozenset(
    {
        "balanced_design_family_margin_gallery",
        "design_centroid_margin_gallery",
        "sigma35_stress_margin_gallery",
    }
)


def _prefer_single_row_panel_layout(plot_id: str | None, panel_count: int, *, configured: object = None) -> bool:
    if configured is not None:
        return bool(configured) and 1 < panel_count <= 6
    return bool(plot_id in _SINGLE_ROW_PANEL_PLOT_IDS and 1 < panel_count <= 6)


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
) -> tuple[int, int, tuple[float, float]]:
    resolved_square_panels = metric_panel_uses_square_axes(plot_id) if square_panels is None else bool(square_panels)
    if plot_id == "reference_alignment_summary" and panel_count > 8:
        columns = min(8, panel_count)
        rows = int(math.ceil(panel_count / columns))
        return rows, columns, (3.45 * columns, 3.55 * rows)
    if plot_id == "rt_lnrna_overlay_ordinal_audit":
        rows, columns = _panel_grid_dimensions(panel_count, prefer_single_row=False)
        return rows, columns, (5.2 * columns, 5.25 * rows)
    if (prefer_single_row or plot_id == "representation_health_summary") and panel_count > 1:
        rows, columns = _panel_grid_dimensions(panel_count, prefer_single_row=True)
        figsize = _grid_figure_size(panel_count, square_panels=resolved_square_panels, prefer_single_row=True)
        extra_width_per_column = 0.34 if plot_id == "sigma35_ordinal_audit" else 0.72
        if plot_id == "representation_health_summary":
            extra_width_per_column = 1.25
        return rows, columns, (figsize[0] + (extra_width_per_column * columns), figsize[1])
    rows, columns = _panel_grid_dimensions(panel_count)
    figsize = _grid_figure_size(panel_count, square_panels=resolved_square_panels)
    if plot_id == "representation_health_summary":
        figsize = (figsize[0] + (1.45 * columns), figsize[1])
    return rows, columns, figsize


def plot_tight_layout_kwargs(
    plot_id: str | None,
    *,
    legend_bottom: float,
    legend_right: float = 0.0,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "pad": 0.95,
        "h_pad": 1.4,
        "w_pad": 0.95,
    }
    if plot_id in _FULL_FORMULA_XY_GRID_PLOT_IDS:
        kwargs["w_pad"] = 2.05
    if plot_id == "representation_health_summary":
        kwargs["w_pad"] = 1.85
    if plot_id == "dataset_overview" or str(plot_id or "").endswith("_dataset_overview"):
        kwargs["w_pad"] = 1.8
    if plot_id == "rt_lnrna_overlay_ordinal_audit":
        kwargs["w_pad"] = 1.75
        kwargs["h_pad"] = 1.6
    if plot_id == "sigma35_ordinal_audit":
        kwargs["w_pad"] = 0.32
    if plot_id == "appendix_umap_gallery":
        kwargs["pad"] = 0.65
        kwargs["h_pad"] = 0.24
    if legend_bottom > 0.0 or legend_right > 0.0:
        kwargs["rect"] = (0.0, legend_bottom, max(0.58, 1.0 - legend_right), 0.995)
    return kwargs
