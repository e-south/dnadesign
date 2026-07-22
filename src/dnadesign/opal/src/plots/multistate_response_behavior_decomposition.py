"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/multistate_response_behavior_decomposition.py

State and family decomposition for the Multistate Response Behavior objective.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..registries.plots import PlotMeta, register_plot
from ._mpl_utils import (
    NOTEBOOK_ANNOTATION_FONTSIZE,
    NOTEBOOK_AXIS_LABEL_FONTSIZE,
    NOTEBOOK_COLORBAR_LABEL_FONTSIZE,
    NOTEBOOK_TICK_FONTSIZE,
    SIGNED_MARGIN_CMAP,
    add_flush_colorbar,
    apply_notebook_axes_style,
    apply_plot_style,
    ensure_mpl_config_dir,
    set_notebook_title,
    wrap_plot_title,
)
from .candidate_annotations import resolve_candidate_display_aliases, short_candidate_id
from .multistate_response_behavior_data import (
    BEHAVIOR_SCORE_REF,
    HARD_BOTTLENECK_REF,
    OFF_SIGNAL_SUPPRESSION_FAMILY_REF,
    ON_SIGNAL_FAMILY_REF,
    RESPONSE_FAMILY_REF,
    SELECTED_COORDINATE_DETAIL_SCOPE,
    load_multistate_response_behavior_plot_data,
)
from .multistate_response_behavior_support import (
    figsize,
    nonempty,
    nonnegative_int,
    positive_float,
    save_figure,
    selection_view_title,
    state_display_labels,
    target_context,
)

KIND = "multistate_response_behavior_selected_decomposition"


@register_plot(
    KIND,
    meta=PlotMeta(
        summary="State-level clearances and family summaries for candidates allocated to one view.",
        premise="Every desired state-level change contributes to the smooth behavior score.",
        decision_value=(
            "Identifies the hard bottleneck and the family evidence behind each allocated candidate before handoff."
        ),
        rationale=(
            "A K-state component matrix reveals whether the scalar is supported broadly or by compensating behavior."
        ),
        alt_text=(
            "Heatmap with allocated candidates as rows. Columns contain every raw response-ordering, ON-signal, "
            "and OFF-signal-suppression coordinate, followed by three family scores and the smooth behavior score. "
            "All values retain the objective input units. An outline marks the lowest state-level coordinate, so "
            "the same hard-bottleneck value is not repeated as a summary column. Zero is a reference direction, "
            "not feasibility."
        ),
        non_claim_boundary=(
            "The decomposition is based on model predictions until measured; the behavior score is not a pass/fail "
            "label."
        ),
        tier="decision",
        params={
            "title": "Optional complete-sentence title.",
            "target_name": "Optional display name for the target behavior.",
            "state_labels": "Optional exact state-id to display-label mapping.",
            "figsize_in": "Optional two-item figure size in inches.",
            "max_selected": "Maximum allocated rows permitted in one heatmap (default 24).",
            "max_coordinates": "Maximum state-level coordinates permitted in one heatmap (default 48).",
            "candidate_label_mode": "Candidate row labels: short_id (default) or alias.",
            "value_label": "Optional display label for the raw objective input units.",
            "color_extent": (
                "Optional positive symmetric color extent in objective input units. Values outside the extent "
                "remain exact in labels and exported data while their colors saturate at rectangular endpoints."
            ),
        },
        requires=[
            "as_of_round",
            "run_id",
            "id",
            "pred__y_hat_model",
            "pred__score_channels",
            "view__rank_competition",
            "view__is_selected",
        ],
        notes=[
            "All columns are replayed from the public K-state objective math; only ledger-allocated candidates are "
            "shown."
        ],
        data_shape="allocated candidate by behavior coordinate and family matrix",
        tidy_schema=["id", "rank", "component_kind", "component_id", "value", "limiting"],
        failure_modes=[
            "ambiguous round or run",
            "no allocated candidates",
            "selected or coordinate count exceeds declared review limit",
            "persisted behavior score disagrees with canonical objective math",
        ],
        objective_family="multistate_response_behavior",
        data_layer="predictions_objective",
        round_scope="single_round",
    ),
)
def render_selected_decomposition(context: Any, params: dict) -> None:
    """Render complete K-state behavior evidence for allocated candidates."""

    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    apply_plot_style()
    data = load_multistate_response_behavior_plot_data(
        context,
        detail_scope=SELECTED_COORDINATE_DETAIL_SCOPE,
    )
    selected = data.selected_coordinate_frame
    if selected is None:
        raise ValueError("Behavior decomposition requires selected-coordinate detail.")
    if selected.empty:
        raise ValueError("Behavior decomposition has no allocated candidates.")
    max_selected = nonnegative_int(params.get("max_selected", 24), name="max_selected")
    max_coordinates = nonnegative_int(params.get("max_coordinates", 48), name="max_coordinates")
    value_label = nonempty(
        params.get("value_label", "Behavior evidence (input units)"),
        field="value_label",
    )
    coordinate_count = len(data.coordinate_labels)
    if max_selected == 0 or len(selected) > max_selected:
        raise ValueError(f"Allocated-candidate heatmap has {len(selected)} rows; max_selected={max_selected}.")
    if max_coordinates == 0 or coordinate_count > max_coordinates:
        raise ValueError(
            f"Behavior heatmap has {coordinate_count} state-level coordinates; max_coordinates={max_coordinates}."
        )

    coordinate_matrix = np.asarray(selected["coordinate_clearances"].tolist(), dtype=float)
    summary_columns = (
        RESPONSE_FAMILY_REF,
        ON_SIGNAL_FAMILY_REF,
        OFF_SIGNAL_SUPPRESSION_FAMILY_REF,
        BEHAVIOR_SCORE_REF,
    )
    matrix = np.concatenate((coordinate_matrix, selected.loc[:, summary_columns].to_numpy(dtype=float)), axis=1)
    full_extent = max(float(np.max(np.abs(matrix))), 1.0e-9)
    configured_extent = params.get("color_extent")
    extent = full_extent if configured_extent is None else positive_float(configured_extent, name="color_extent")
    saturated_below = bool(np.any(matrix < -extent))
    saturated_above = bool(np.any(matrix > extent))
    colorbar_extend = (
        "both"
        if saturated_below and saturated_above
        else "min"
        if saturated_below
        else "max"
        if saturated_above
        else "neither"
    )
    saturated_count = int(np.count_nonzero(np.abs(matrix) > extent))
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    default_width = min(15.0, max(8.2, 3.8 + 0.62 * matrix.shape[1]))
    default_height = max(4.8, min(10.0, 2.5 + 0.55 * len(selected)))
    figure_size = figsize(params.get("figsize_in", (default_width, default_height)))
    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
    image = ax.imshow(matrix, cmap=SIGNED_MARGIN_CMAP, norm=norm, aspect="auto")
    apply_notebook_axes_style(ax, grid=False, square=False)
    display_states = state_display_labels(data.state_ids, params.get("state_labels"))
    coordinate_tick_labels = [
        _coordinate_display_label(label, states=display_states) for label in data.coordinate_labels
    ]
    summary_tick_labels = [
        r"$S_R$",
        r"$S_{\mathrm{ON}}$",
        r"$S_{\mathrm{OFF}}$",
        r"$S_{\mathrm{MSRB}}$",
    ]
    ax.set_xticks(np.arange(matrix.shape[1]), [*coordinate_tick_labels, *summary_tick_labels], rotation=45, ha="right")
    row_labels = _candidate_row_labels(context, selected=selected, mode=params.get("candidate_label_mode", "short_id"))
    ax.set_yticks(
        np.arange(len(selected)),
        [f"#{int(row.view__rank_competition)}  {row_labels[str(row.id)]}" for row in selected.itertuples(index=False)],
    )
    ax.tick_params(axis="both", labelsize=NOTEBOOK_TICK_FONTSIZE)
    ax.set_xlabel(
        "Behavior coordinate (0 = reference direction; outline = hard bottleneck)",
        fontsize=NOTEBOOK_AXIS_LABEL_FONTSIZE,
        labelpad=8,
    )
    ax.set_ylabel("Competition rank · candidate", fontsize=NOTEBOOK_AXIS_LABEL_FONTSIZE, labelpad=8)
    title = selection_view_title(params.get("title", "Behavior evidence for allocated candidates"), context=context)
    set_notebook_title(
        ax,
        wrap_plot_title(title, width=50),
        subtitle=wrap_plot_title(target_context(data, params), width=56),
    )
    for row_index, row in enumerate(selected.itertuples(index=False)):
        limiting_column = int(row.limiting_coordinate_index)
        ax.add_patch(
            Rectangle(
                (limiting_column - 0.5, row_index - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#111111",
                linewidth=2.0,
                zorder=4,
            )
        )
        for column_index, value in enumerate(matrix[row_index]):
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=NOTEBOOK_ANNOTATION_FONTSIZE,
                color="white" if abs(value) > 0.55 * extent else "#111111",
            )
    ax.axvline(coordinate_count - 0.5, color="#555555", linewidth=1.2)
    ax.axvline(coordinate_count + 2.5, color="#777777", linewidth=0.9)
    colorbar = add_flush_colorbar(
        fig,
        ax,
        image,
        label=value_label,
        pad=0.06,
        ticklabelsize=NOTEBOOK_TICK_FONTSIZE,
        extend=colorbar_extend,
        extendrect=True,
    )
    colorbar.ax.yaxis.label.set_size(NOTEBOOK_COLORBAR_LABEL_FONTSIZE)
    context.artifact_metadata["notebook_view"] = {
        "title": title,
        "context": target_context(data, params),
        "score_units": "objective_input_units",
        "softmin_scale": float(data.softmin_scale),
        "value_label": value_label,
        "color_scale": {
            "center": 0.0,
            "extent": extent,
            "extend": colorbar_extend,
            "context": (
                f"{saturated_count:,} matrix values saturate at the color endpoints; exact labels and tidy data "
                "are retained"
                if saturated_count
                else "Symmetric behavior evidence in input units around the zero reference direction"
            ),
        },
    }
    save_figure(context, fig)
    if context.save_data:
        context.save_df(decomposition_tidy(selected, coordinate_labels=data.coordinate_labels))
    plt.close(fig)


def decomposition_tidy(selected: pd.DataFrame, *, coordinate_labels: Sequence[str]) -> pd.DataFrame:
    """Return one row per state-level coordinate or scalar summary."""

    summary = (
        (RESPONSE_FAMILY_REF, "family"),
        (ON_SIGNAL_FAMILY_REF, "family"),
        (OFF_SIGNAL_SUPPRESSION_FAMILY_REF, "family"),
        (HARD_BOTTLENECK_REF, "diagnostic"),
        (BEHAVIOR_SCORE_REF, "score"),
    )
    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        clearances = tuple(float(value) for value in row.coordinate_clearances)
        if len(clearances) != len(coordinate_labels):
            raise ValueError("Behavior decomposition coordinate values and labels are misaligned.")
        for index, (component_id, value) in enumerate(zip(coordinate_labels, clearances, strict=True)):
            rows.append(
                {
                    "id": str(row.id),
                    "rank": int(row.view__rank_competition),
                    "component_kind": "coordinate",
                    "component_id": str(component_id),
                    "value": value,
                    "limiting": index == int(row.limiting_coordinate_index),
                }
            )
        for component_id, component_kind in summary:
            rows.append(
                {
                    "id": str(row.id),
                    "rank": int(row.view__rank_competition),
                    "component_kind": component_kind,
                    "component_id": component_id,
                    "value": float(getattr(row, component_id)),
                    "limiting": False,
                }
            )
    return pd.DataFrame.from_records(rows)


def _candidate_row_labels(context: Any, *, selected: pd.DataFrame, mode: object) -> dict[str, str]:
    label_mode = str(mode).strip().lower()
    if label_mode not in {"short_id", "alias"}:
        raise ValueError("candidate_label_mode must be 'short_id' or 'alias'.")
    ids = selected["id"].astype(str).tolist()
    if label_mode == "short_id":
        return {candidate_id: short_candidate_id(candidate_id) for candidate_id in ids}
    records_path = context.data_paths.get("records")
    if records_path is None:
        raise ValueError("candidate_label_mode='alias' requires the built-in records.parquet input.")
    return resolve_candidate_display_aliases(records_path, ids)


def _coordinate_display_label(label: str, *, states: dict[str, str]) -> str:
    family, raw = str(label).split(":", maxsplit=1)
    if family == "response":
        on_state, off_state = raw.split(">", maxsplit=1)
        return f"r[{states[on_state]}]−r[{states[off_state]}]"
    if family == "on_signal":
        return f"b[{states[raw]}]"
    if family == "off_signal_suppression":
        return f"−b[{states[raw]}]"
    raise ValueError(f"Unknown behavior coordinate family: {family!r}.")


__all__ = ["decomposition_tidy", "render_selected_decomposition"]
