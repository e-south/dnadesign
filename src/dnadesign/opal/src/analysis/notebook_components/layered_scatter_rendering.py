"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/layered_scatter_rendering.py

Render manifest-backed layered-scatter notebook figures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...plots._mpl_utils import NOTEBOOK_ANNOTATION_FONTSIZE


def render_layered_scatter_figure(rows: pd.DataFrame, *, contract: Mapping[str, Any]):
    """Render already-filtered prediction, selection, and observation layers."""

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

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
        compact_batch_label,
        observed_batch_marker_map,
        scatter_smart,
        set_notebook_title,
        wrap_plot_title,
    )
    from ...plots.candidate_annotations import annotate_candidate_aliases

    apply_plot_style()
    view = _mapping(contract["view"])
    runtime = _mapping(contract["runtime"])
    record_column = str(view["record_kind_column"])
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

    color_scale = _mapping(runtime["color_scale"])
    color_center = float(color_scale["center"])
    color_extent = float(color_scale["extent"])
    colorbar_extend = str(color_scale.get("extend") or "neither")
    if not np.isfinite(color_extent) or color_extent <= 0.0:
        raise ValueError("Layered-scatter color_extent must be finite and positive.")
    if not np.isfinite(color_center):
        raise ValueError("Layered-scatter color center must be finite.")
    norm = TwoSlopeNorm(
        vmin=color_center - color_extent,
        vcenter=color_center,
        vmax=color_center + color_extent,
    )
    cmap = SIGNED_MARGIN_CMAP
    fig, ax = plt.subplots(figsize=(7.2, 7.2), layout="constrained")
    apply_notebook_axes_style(ax, square=True)
    kinds = rows[record_column].astype(str)
    selection_round_column = "__notebook_selection_round"
    if selection_round_column not in rows:
        raise ValueError("Layered-scatter rows are missing categorical selection-round provenance.")
    selection_rounds = rows[selection_round_column]
    show_pool = bool(rows.attrs.get("show_prediction_pool", True))
    show_selected = bool(rows.attrs.get("show_selected", True))
    pool = rows.loc[kinds.eq(prediction_value) & selection_rounds.isna()] if show_pool else rows.iloc[0:0]
    selected = rows.loc[kinds.eq(prediction_value) & selection_rounds.notna()] if show_selected else rows.iloc[0:0]
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
            label=_legend_label(f"Predicted pool (n={len(pool):,})"),
            zorder=2,
        )
    selection_markers = ("D", "P", "X", "s", "^", "v", "<", ">")
    for index, round_k in enumerate(sorted(selected[selection_round_column].astype(int).unique())):
        round_selected = selected.loc[selected[selection_round_column].astype(int).eq(round_k)]
        ax.scatter(
            round_selected[x_column],
            round_selected[y_column],
            c=round_selected[color_column],
            cmap=cmap,
            norm=norm,
            marker=selection_markers[index % len(selection_markers)],
            s=42,
            edgecolors="#111111",
            linewidths=1.1,
            label=_legend_label(f"Selected for Round {round_k} (n={len(round_selected)})"),
            zorder=7,
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
            label=_legend_label(f"Observed · {compact_batch_label(batch_id)} (n={len(batch)})"),
            zorder=7,
        )
    reference_lines = _mapping(runtime["reference_lines"])
    _draw_reference_lines(ax, reference_lines.get("x"), axis="x")
    _draw_reference_lines(ax, reference_lines.get("y"), axis="y")
    ax.set_xlim(_limits(runtime["x_limits"], field="x_limits"))
    ax.set_ylim(_limits(runtime["y_limits"], field="y_limits"))
    ax.set_xlabel(str(runtime["x_label"]), fontsize=NOTEBOOK_AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.set_ylabel(str(runtime["y_label"]), fontsize=NOTEBOOK_AXIS_LABEL_FONTSIZE, labelpad=8)
    set_notebook_title(
        ax,
        wrap_plot_title(runtime["title"], width=50),
        subtitle=wrap_plot_title(runtime["context"], width=56),
        location="center",
        title_fontsize=NOTEBOOK_TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=NOTEBOOK_TICK_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels[0].startswith("Predicted pool"):
        handles[0] = Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor="#56B4E9",
            markeredgecolor="none",
            alpha=0.8,
        )
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=NOTEBOOK_LEGEND_FONTSIZE,
        ncol=min(2, max(1, len(handles))),
        frameon=False,
        handletextpad=0.4,
        columnspacing=0.7,
        alignment="left",
    )
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = add_flush_colorbar(
        fig,
        ax,
        mappable,
        label=str(runtime["color_label"]),
        pad=0.065,
        ticklabelsize=NOTEBOOK_TICK_FONTSIZE,
        extend=colorbar_extend,
        extendrect=True,
    )
    colorbar.ax.yaxis.label.set_size(NOTEBOOK_COLORBAR_LABEL_FONTSIZE)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.labelpad = 8
    _annotate_visible_rows(
        ax,
        rows,
        x_column=x_column,
        y_column=y_column,
        label_column=label_column,
        batch_column=batch_column,
        batch_labels=batch_labels,
        selection_round_column=selection_round_column,
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
    selection_round_column: str,
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
            round_k = getattr(row, selection_round_column)
            if pd.notna(round_k):
                suffix = f"Round {int(round_k)}"
            else:
                suffix = batch_labels.get(str(batch_id), "Observed")
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
        max_lanes=2,
    )


def _limits(value: object, *, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Layered-scatter {field} must contain two values.")
    parsed = tuple(float(item) for item in value)
    if parsed[0] >= parsed[1]:
        raise ValueError(f"Layered-scatter {field} must be strictly increasing.")
    return parsed


def _draw_reference_lines(ax: Any, raw: object, *, axis: str) -> None:
    if not isinstance(raw, list):
        raise ValueError(f"Layered-scatter {axis} reference lines must be a list.")
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"value", "label"}:
            raise ValueError(f"Layered-scatter {axis} reference lines require exactly value and label.")
        value = float(item["value"])
        label = str(item["label"]).strip()
        if not np.isfinite(value) or not label:
            raise ValueError(f"Layered-scatter {axis} reference-line values must be finite and labels non-empty.")
        draw = ax.axvline if axis == "x" else ax.axhline
        line = draw(value, color="#555555", linestyle="--", linewidth=1.0, zorder=1)
        line.set_gid(f"reference-line:{axis}:{label}")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _legend_label(value: str) -> str:
    return textwrap.fill(
        value,
        width=32,
        break_long_words=False,
        break_on_hyphens=False,
    )


__all__ = ["render_layered_scatter_figure"]
