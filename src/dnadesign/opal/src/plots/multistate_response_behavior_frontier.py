"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/multistate_response_behavior_frontier.py

Family-landscape plot for the threshold-free Multistate Response Behavior objective.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.ledger import OBSERVED_EVENTS_ARTIFACT_KEY
from ..registries.plots import PlotMeta, register_plot
from ._mpl_utils import (
    DEFAULT_SQUARE_FIGSIZE,
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
    ensure_mpl_config_dir,
    observed_batch_marker_map,
    scatter_smart,
    wrap_plot_title,
)
from .candidate_annotations import (
    observed_candidate_display_labels,
    resolve_candidate_display_aliases,
    short_candidate_id,
)
from .multistate_response_behavior_data import (
    BEHAVIOR_SCORE_REF,
    HARD_BOTTLENECK_REF,
    OFF_SIGNAL_SUPPRESSION_FAMILY_REF,
    ON_SIGNAL_FAMILY_REF,
    RESPONSE_FAMILY_REF,
    SUMMARY_DETAIL_SCOPE,
    load_multistate_response_behavior_plot_data,
)
from .multistate_response_behavior_support import (
    figsize,
    nonempty,
    nonnegative_int,
    positive_float,
    save_figure,
    selection_view_title,
    target_context,
    unit_float,
)

KIND = "multistate_response_behavior_frontier"
COLOR_CONTEXT = "red = stronger OFF-signal suppression; 0 = reference-relative family score; not feasibility"


@register_plot(
    KIND,
    meta=PlotMeta(
        summary="Predicted behavior families with observed events and allocated candidates identified.",
        premise=(
            "A threshold-free behavior score should keep response ordering, target-ON signal, and "
            "target-OFF signal suppression separately visible."
        ),
        decision_value=(
            "Shows which behavior families support or limit candidates allocated by the active selection view."
        ),
        rationale="A three-channel landscape exposes family tradeoffs that one smooth behavior score would hide.",
        alt_text=(
            "Square scatter plot of response-family score against target-ON signal-family score. Color encodes "
            "target-OFF signal-suppression family score, with red indicating stronger suppression relative to the "
            "same-state reference. Distinct marker shapes identify observed batches and diamonds identify candidates "
            "allocated by the active campaign view. No line denotes feasibility because this objective has no "
            "acceptance threshold."
        ),
        non_claim_boundary=(
            "Natural zero is a reference direction, not feasibility, and predicted behavior does not establish "
            "measured promoter performance or prospective hill-climbing efficacy."
        ),
        tier="decision",
        params={
            "title": "Optional complete-sentence title.",
            "response_family_label": "Display label for the response-family axis.",
            "on_signal_family_label": "Display label for the ON-signal-family axis.",
            "off_signal_suppression_family_label": "Display label for the OFF-signal-suppression color scale.",
            "target_name": "Optional display name for the target behavior.",
            "state_labels": "Optional exact state-id to display-label mapping.",
            "figsize_in": "Two-item figure size in inches.",
            "point_size": "Candidate point size (default 10).",
            "point_alpha": "Candidate point alpha (default 0.35).",
            "rasterize_at": "Rasterize points at or above this count (default 10000).",
            "color_extent": (
                "Optional positive symmetric color extent. Values outside the extent remain plotted and saturate "
                "at rectangular colorbar extensions. The extent must be fixed independently of the active view."
            ),
            "surface_label": "Optional notebook-facing label.",
        },
        requires=[
            "as_of_round",
            "run_id",
            "id",
            "pred__y_hat_model",
            "pred__score_channels",
            "view__rank_competition",
            "view__is_selected",
            OBSERVED_EVENTS_ARTIFACT_KEY,
        ],
        notes=[
            "Reads one multistate_response_behavior_v1 run and replays predictions and run-pinned observed events "
            "through the canonical public mathematics API."
        ],
        data_shape="candidate behavior-family landscape",
        tidy_schema=[
            "id",
            BEHAVIOR_SCORE_REF,
            HARD_BOTTLENECK_REF,
            RESPONSE_FAMILY_REF,
            ON_SIGNAL_FAMILY_REF,
            OFF_SIGNAL_SUPPRESSION_FAMILY_REF,
            "all_reference_directions_met",
            "limiting_coordinate_label",
            "selected",
            "rank",
            "record_kind",
            "observed_round",
            "batch_id",
            "batch_key",
            "display_label",
        ],
        failure_modes=[
            "ambiguous round or run",
            "objective or selection score reference mismatch",
            "missing or malformed response vectors",
            "persisted behavior score disagrees with canonical objective math",
        ],
        objective_family="multistate_response_behavior",
        data_layer="predictions_plus_labels",
        round_scope="single_round",
        label_requirement="required",
        notebook_view={
            "adapter": "layered_scatter_v1",
            "record_kind_column": "record_kind",
            "prediction_value": "prediction",
            "observed_value": "observed_label",
            "selection_column": "selected",
            "batch_column": "batch_key",
            "label_column": "display_label",
            "x_column": RESPONSE_FAMILY_REF,
            "y_column": ON_SIGNAL_FAMILY_REF,
            "color_column": OFF_SIGNAL_SUPPRESSION_FAMILY_REF,
            "interactive": {
                "adapter": "three_axis_scatter_v1",
                "score_column": BEHAVIOR_SCORE_REF,
                "score_label": r"Behavior score, $S_{\mathrm{MSRB}}$",
                "prediction_sample_limit": 8_000,
                "sampling_method": "sha256_id_v1",
            },
        },
    ),
)
def render_family_frontier(context: Any, params: dict) -> None:
    """Render all three behavior families without inventing a feasibility boundary."""

    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    apply_plot_style()
    data = load_multistate_response_behavior_plot_data(context, detail_scope=SUMMARY_DETAIL_SCOPE)
    frame = data.frame
    observed = data.observed_frame
    response_label = nonempty(
        params.get("response_family_label", r"Response-ordering family score, $S_R$"),
        field="response_family_label",
    )
    on_label = nonempty(
        params.get("on_signal_family_label", r"ON-signal family score, $S_{\mathrm{ON}}$"),
        field="on_signal_family_label",
    )
    off_label = nonempty(
        params.get(
            "off_signal_suppression_family_label",
            r"OFF-signal-suppression family score, $S_{\mathrm{OFF}}$",
        ),
        field="off_signal_suppression_family_label",
    )
    figure_size = figsize(params.get("figsize_in", DEFAULT_SQUARE_FIGSIZE))
    point_size = positive_float(params.get("point_size", 10.0), name="point_size")
    point_alpha = unit_float(params.get("point_alpha", 0.35), name="point_alpha")
    rasterize_at = nonnegative_int(params.get("rasterize_at", 10_000), name="rasterize_at")
    predicted_color = frame[OFF_SIGNAL_SUPPRESSION_FAMILY_REF].to_numpy(dtype=float)
    observed_color = observed[OFF_SIGNAL_SUPPRESSION_FAMILY_REF].to_numpy(dtype=float)
    visible_color = np.concatenate((predicted_color, observed_color))
    full_color_extent = max(float(np.max(np.abs(visible_color))), 1.0e-9)
    configured_color_extent = params.get("color_extent")
    color_extent = (
        full_color_extent
        if configured_color_extent is None
        else positive_float(configured_color_extent, name="color_extent")
    )
    saturated_color_count = int(np.count_nonzero(np.abs(visible_color) > color_extent))
    saturated_below = bool(np.any(visible_color < -color_extent))
    saturated_above = bool(np.any(visible_color > color_extent))
    colorbar_extend = (
        "both"
        if saturated_below and saturated_above
        else "min"
        if saturated_below
        else "max"
        if saturated_above
        else "neither"
    )
    norm = TwoSlopeNorm(vmin=-color_extent, vcenter=0.0, vmax=color_extent)
    fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
    apply_notebook_axes_style(ax, square=True)
    points = scatter_smart(
        ax,
        frame[RESPONSE_FAMILY_REF],
        frame[ON_SIGNAL_FAMILY_REF],
        c=predicted_color,
        cmap=SIGNED_MARGIN_CMAP,
        norm=norm,
        s=point_size,
        alpha=point_alpha,
        rasterize_at=rasterize_at,
        label=f"Predicted pool (n={len(frame):,})",
        zorder=2,
    )
    selected = frame["view__is_selected"].to_numpy(dtype=bool)
    if not selected.any():
        raise ValueError("Behavior family landscape has no allocated candidates.")
    observed_markers = observed_batch_marker_map(tuple(sorted(observed["batch_key"].astype(str).unique())))
    for batch_key, batch in observed.groupby("batch_key", sort=True):
        batch_key = str(batch_key)
        ax.scatter(
            batch[RESPONSE_FAMILY_REF],
            batch[ON_SIGNAL_FAMILY_REF],
            c=batch[OFF_SIGNAL_SUPPRESSION_FAMILY_REF],
            cmap=SIGNED_MARGIN_CMAP,
            norm=norm,
            edgecolors="#111111",
            marker=observed_markers[batch_key],
            s=max(30.0, point_size * 2.0),
            linewidths=0.9,
            label=f"Observed · {compact_batch_label(batch_key)} (n={len(batch)})",
            zorder=3,
        )
    ax.scatter(
        frame.loc[selected, RESPONSE_FAMILY_REF],
        frame.loc[selected, ON_SIGNAL_FAMILY_REF],
        c=predicted_color[selected],
        cmap=SIGNED_MARGIN_CMAP,
        norm=norm,
        marker="D",
        s=max(36.0, point_size * 2.5),
        edgecolors="#111111",
        linewidths=1.2,
        label=f"Allocated to view (n={int(selected.sum())})",
        zorder=4,
    )
    ax.set_xlabel(response_label, fontsize=NOTEBOOK_AXIS_LABEL_FONTSIZE, labelpad=8)
    ax.set_ylabel(on_label, fontsize=NOTEBOOK_AXIS_LABEL_FONTSIZE, labelpad=8)
    title = selection_view_title(params.get("title", "Multistate behavior family landscape"), context=context)
    target = wrap_plot_title(target_context(data, params), width=56)
    ax.set_title(
        f"{wrap_plot_title(title, width=50)}\n{target}",
        loc="center",
        fontweight="semibold",
        fontsize=NOTEBOOK_TITLE_FONTSIZE,
        pad=10,
        linespacing=1.25,
    )
    ax.tick_params(axis="both", labelsize=NOTEBOOK_TICK_FONTSIZE)
    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            markerfacecolor="#D0D0D0",
            markeredgecolor="#666666",
            markeredgewidth=1.0,
            label=f"Predicted pool (n={len(frame):,})",
        )
    ]
    for batch_key, batch in observed.groupby("batch_key", sort=True):
        batch_key = str(batch_key)
        legend_handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=observed_markers[batch_key],
                markersize=7,
                markerfacecolor="none",
                markeredgecolor="#111111",
                markeredgewidth=1.0,
                label=f"Observed · {compact_batch_label(batch_key)} (n={len(batch)})",
            )
        )
    legend_handles.append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="D",
            markersize=7,
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=1.2,
            label=f"Allocated to view (n={int(selected.sum())})",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=NOTEBOOK_LEGEND_FONTSIZE,
        ncol=3,
        frameon=False,
        markerscale=1.5,
        handletextpad=0.45,
        columnspacing=0.8,
    )
    colorbar = add_flush_colorbar(
        fig,
        ax,
        points,
        label=off_label,
        pad=0.065,
        ticklabelsize=NOTEBOOK_TICK_FONTSIZE,
        extend=colorbar_extend,
        extendrect=True,
    )
    colorbar.ax.yaxis.label.set_size(NOTEBOOK_COLORBAR_LABEL_FONTSIZE)
    aliases = _display_aliases(context, frame=frame, observed=observed, selected=selected)
    context.artifact_metadata["notebook_view"] = {
        "title": title,
        "context": target,
        "x_label": response_label,
        "y_label": on_label,
        "color_label": off_label,
        "reference_lines": {"x": [], "y": []},
        "color_scale": {
            "center": 0.0,
            "extent": color_extent,
            "context": (
                f"{COLOR_CONTEXT}; {saturated_color_count:,} visible values saturate at the color endpoints; "
                "points and exact ledger values are retained"
                if saturated_color_count
                else COLOR_CONTEXT
            ),
            "extend": colorbar_extend,
        },
        "x_limits": [float(value) for value in ax.get_xlim()],
        "y_limits": [float(value) for value in ax.get_ylim()],
    }
    save_figure(context, fig)
    if context.save_data:
        context.save_df(frontier_tidy(frame, observed, aliases=aliases))
    plt.close(fig)


def frontier_tidy(
    frame: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    aliases: Mapping[str, str],
) -> pd.DataFrame:
    """Return the generic layered-scatter records for predictions and observations."""

    shared = (
        BEHAVIOR_SCORE_REF,
        HARD_BOTTLENECK_REF,
        RESPONSE_FAMILY_REF,
        ON_SIGNAL_FAMILY_REF,
        OFF_SIGNAL_SUPPRESSION_FAMILY_REF,
        "all_reference_directions_met",
        "limiting_coordinate_label",
    )
    predictions = pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "record_kind": "prediction",
            **{column: frame[column] for column in shared},
            "selected": frame["view__is_selected"].astype(bool),
            "rank": frame["view__rank_competition"].astype(int),
            "observed_round": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "batch_id": pd.Series([pd.NA] * len(frame), dtype="string"),
            "batch_key": pd.Series([pd.NA] * len(frame), dtype="string"),
            "display_label": frame["id"].astype(str).map(aliases).astype("string"),
        }
    )
    labels = pd.DataFrame(
        {
            "id": observed["id"].astype(str),
            "record_kind": "observed_label",
            **{column: observed[column] for column in shared},
            "selected": False,
            "rank": pd.Series([pd.NA] * len(observed), dtype="Int64"),
            "observed_round": observed["observed_round"].astype("Int64"),
            "batch_id": observed["batch_id"].astype("string"),
            "batch_key": observed["batch_key"].astype("string"),
            "display_label": observed_candidate_display_labels(observed, fallbacks=aliases),
        }
    )
    return pd.concat([predictions, labels], ignore_index=True)


def _display_aliases(
    context: Any,
    *,
    frame: pd.DataFrame,
    observed: pd.DataFrame,
    selected: np.ndarray,
) -> dict[str, str]:
    ids = list(dict.fromkeys([*frame.loc[selected, "id"].astype(str), *observed["id"].astype(str)]))
    records_path = context.data_paths.get("records")
    if records_path is not None:
        return resolve_candidate_display_aliases(records_path, ids)
    return {candidate_id: short_candidate_id(candidate_id) for candidate_id in ids}


__all__ = ["frontier_tidy", "render_family_frontier"]
