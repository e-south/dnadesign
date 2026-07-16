"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/layered_scatter_rendering.py

Render manifest-backed layered-scatter notebook figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ...plots._mpl_utils import NOTEBOOK_ANNOTATION_FONTSIZE


def render_layered_scatter_figure(rows: pd.DataFrame, *, contract: Mapping[str, Any]):
    """Render already-filtered prediction, selection, and observation layers."""

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm

    from ...plots._mpl_utils import (
        NOTEBOOK_AXIS_LABEL_FONTSIZE,
        NOTEBOOK_COLORBAR_LABEL_FONTSIZE,
        NOTEBOOK_LEGEND_FONTSIZE,
        NOTEBOOK_TICK_FONTSIZE,
        NOTEBOOK_TITLE_FONTSIZE,
        SIGNED_MARGIN_CMAP,
        add_flush_colorbar,
        apply_notebook_axes_style,
        apply_plot_style,
        observed_batch_marker_map,
        scatter_smart,
    )
    from ...plots.response_magnitude_feasibility_aliases import annotate_candidate_aliases

    apply_plot_style()
    view = _mapping(contract["view"])
    runtime = _mapping(contract["runtime"])
    record_column = str(view["record_kind_column"])
    selected_column = str(view["selection_column"])
    batch_column = str(view["batch_column"])
    label_column = str(view["label_column"])
    x_column = str(view["x_column"])
    y_column = str(view["y_column"])
    color_column = str(view["color_column"])
    prediction_value = str(view["prediction_value"])
    observed_value = str(view["observed_value"])
    numeric = rows[[x_column, y_column, color_column]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Layered-scatter visible coordinates and colors must be finite.")

    color_extent = float(runtime["color_extent"])
    if not np.isfinite(color_extent) or color_extent <= 0.0:
        raise ValueError("Layered-scatter color_extent must be finite and positive.")
    norm = TwoSlopeNorm(vmin=-color_extent, vcenter=0.0, vmax=color_extent)
    cmap = SIGNED_MARGIN_CMAP
    fig, ax = plt.subplots(figsize=(7.2, 7.2), layout="constrained")
    apply_notebook_axes_style(ax, square=True)
    kinds = rows[record_column].astype(str)
    selected_flags = rows[selected_column].fillna(False).astype(bool)
    show_pool = bool(rows.attrs.get("show_prediction_pool", True))
    show_selected = bool(rows.attrs.get("show_selected", True))
    pool = rows.loc[kinds.eq(prediction_value)] if show_pool else rows.iloc[0:0]
    selected = rows.loc[kinds.eq(prediction_value) & selected_flags] if show_selected else rows.iloc[0:0]
    observed = rows.loc[kinds.eq(observed_value)]
    if not pool.empty:
        scatter_smart(
            ax,
            pool[x_column],
            pool[y_column],
            c=pool[color_column],
            cmap=cmap,
            norm=norm,
            s=10,
            alpha=0.32,
            rasterize_at=10_000,
            label=f"Predicted pool (n={len(pool):,})",
            zorder=2,
        )
    if not selected.empty:
        ax.scatter(
            selected[x_column],
            selected[y_column],
            c=selected[color_column],
            cmap=cmap,
            norm=norm,
            marker="D",
            s=42,
            edgecolors="#111111",
            linewidths=1.1,
            label=f"Selected (n={len(selected)})",
            zorder=4,
        )
    batch_labels = {str(item["id"]): str(item["label"]) for item in contract["observed_batches"]}
    observed_batch_ids = observed[batch_column].astype(str)
    visible_batch_ids = tuple(batch_id for batch_id in batch_labels if observed_batch_ids.eq(batch_id).any())
    batch_markers = observed_batch_marker_map(
        visible_batch_ids,
        universe_batch_ids=tuple(batch_labels),
    )
    for batch_id in batch_labels:
        batch = observed.loc[observed_batch_ids.eq(batch_id)]
        if batch.empty:
            continue
        ax.scatter(
            batch[x_column],
            batch[y_column],
            c=batch[color_column],
            cmap=cmap,
            norm=norm,
            marker=batch_markers[batch_id],
            s=34,
            edgecolors="#111111",
            linewidths=0.8,
            label=f"Observed · {batch_labels[batch_id]} (n={len(batch)})",
            zorder=3,
        )
    ax.axvline(float(runtime["x_boundary"]), color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(float(runtime["y_boundary"]), color="#555555", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlim(_limits(runtime["x_limits"], field="x_limits"))
    ax.set_ylim(_limits(runtime["y_limits"], field="y_limits"))
    ax.set_xlabel(str(runtime["x_label"]), fontsize=NOTEBOOK_AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.set_ylabel(str(runtime["y_label"]), fontsize=NOTEBOOK_AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.set_title(
        f"{runtime['title']}\n{runtime['context']}",
        loc="center",
        fontweight="semibold",
        fontsize=NOTEBOOK_TITLE_FONTSIZE,
        pad=10,
        linespacing=1.25,
    )
    ax.tick_params(axis="both", labelsize=NOTEBOOK_TICK_FONTSIZE)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=NOTEBOOK_LEGEND_FONTSIZE,
        ncol=min(3, max(1, len(handles))),
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.7,
    )
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = add_flush_colorbar(
        fig,
        ax,
        mappable,
        label=f"{runtime['color_label']}\nred = greater clearance; 0 = boundary",
        pad=0.065,
        ticklabelsize=NOTEBOOK_TICK_FONTSIZE,
    )
    colorbar.ax.yaxis.label.set_size(NOTEBOOK_COLORBAR_LABEL_FONTSIZE)
    _annotate_visible_rows(
        ax,
        rows,
        x_column=x_column,
        y_column=y_column,
        label_column=label_column,
        batch_column=batch_column,
        batch_labels=batch_labels,
        annotate_candidate_aliases=annotate_candidate_aliases,
    )
    return fig


def _annotate_visible_rows(
    ax: Any,
    rows: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    label_column: str,
    batch_column: str,
    batch_labels: Mapping[str, str],
    annotate_candidate_aliases: Any,
) -> None:
    annotation_positions = tuple(rows.attrs.get("annotate_row_positions") or ())
    annotation_rows = rows.iloc[list(annotation_positions)].copy()
    if annotation_rows.empty:
        return
    repeats = annotation_rows["id"].astype(str).value_counts()
    synthetic_aliases: dict[str, str] = {}
    synthetic_ids: list[str] = []
    for index, row in enumerate(annotation_rows.itertuples(index=False)):
        source_id = str(getattr(row, "id"))
        synthetic_id = f"annotation-{index}"
        label = str(getattr(row, label_column))
        if repeats[source_id] > 1:
            batch_id = getattr(row, batch_column)
            suffix = batch_labels.get(str(batch_id), "Selected") if pd.notna(batch_id) else "Selected"
            label = f"{label} · {suffix}"
        synthetic_ids.append(synthetic_id)
        synthetic_aliases[synthetic_id] = label
    annotation_rows["id"] = synthetic_ids
    annotate_candidate_aliases(
        ax,
        annotation_rows,
        synthetic_aliases,
        x_column=x_column,
        y_column=y_column,
        font_size=NOTEBOOK_ANNOTATION_FONTSIZE,
    )


def _limits(value: object, *, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Layered-scatter {field} must contain two values.")
    parsed = tuple(float(item) for item in value)
    if parsed[0] >= parsed[1]:
        raise ValueError(f"Layered-scatter {field} must be strictly increasing.")
    return parsed


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["render_layered_scatter_figure"]
