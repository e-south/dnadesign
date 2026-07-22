"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/observed_objective_over_rounds.py

Renders run-pinned observed objective evidence across campaign batches.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..analysis.observed_objective_history import load_observed_objective_history
from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._mpl_utils import (
    DEFAULT_SQUARE_FIGSIZE,
    apply_notebook_axes_style,
    apply_plot_style,
    categorical_color,
    ensure_mpl_config_dir,
    legend_below_single_row,
    math_label,
    plot_metric_label,
    pretty_batch_label,
    pretty_title,
    save_notebook_square_figure,
)
from ._param_utils import get_bool


@register_plot(
    "observed_objective_over_rounds",
    meta=PlotMeta(
        summary="Observed candidate objective values and between-candidate batch spread over rounds.",
        premise=(
            "Run-pinned observed labels can be compared only when their objective and Y-space contracts are identical."
        ),
        decision_value="Shows how measured candidate values and support vary across batches under fixed semantics.",
        rationale=(
            "Candidate points preserve sample visibility; median and IQR summarize between-candidate batch spread."
        ),
        alt_text=(
            "Candidate objective values by observed batch, with batch median, interquartile range, "
            "and optional cumulative best."
        ),
        non_claim_boundary=(
            "The IQR is between-candidate spread, not uncertainty or a confidence interval. "
            "Observed round and batch record measurement timing, not selection provenance. "
            "A cumulative best is monotone by construction and does not establish model learning, "
            "selection-policy improvement, or a causal round effect."
        ),
        tier="evidence",
        params={
            "run_series": "Required opal.observed_objective_run_series.v1 map with one digest-bound run per round.",
            "zero_boundary": "Show a zero objective boundary when zero has declared meaning (default false).",
            "show_cumulative_best": "Overlay the best observed value through each round (default true).",
            "title": "Optional centered figure title.",
            "metric_label": "Optional publication label for the observed objective channel.",
            "figsize_in": "Optional two-item figure size in inches.",
        },
        requires=[
            "runs.parquet objective and selection definitions",
            "digest-bound labels/observed_events.parquet per declared run",
            "objective pointwise_params_v1 replay contract",
        ],
        notes=[
            "The run-series map is authoritative; implicit latest-run selection is not supported.",
            "Cumulative snapshots must preserve prior candidate-round events exactly.",
            "Campaign-history events without batch IDs are grouped by their observed round.",
        ],
        data_shape="candidate objective points and batch summaries over rounds",
        tidy_schema=[
            "row_kind",
            "observed_round",
            "batch_id",
            "id",
            "display_label",
            "objective_value",
            "candidate_count",
            "batch_median",
            "between_candidate_q25",
            "between_candidate_q75",
            "cumulative_best",
            "selection_view_id",
            "objective_name",
            "score_ref",
            "objective_mode",
            "y_space",
        ],
        failure_modes=[
            "missing or stale per-run contract digest",
            "objective, calibration, target-mask, score, view, Y-space, or ingest drift",
            "objective does not opt into pointwise observed replay",
            "changed or duplicate candidate-round events in cumulative snapshots",
        ],
        objective_family="generic",
        data_layer="labels_objective",
        round_scope="round_history",
        label_requirement="required",
        requires_model_artifact=False,
    ),
)
def render(context, params: dict) -> None:
    if context.rounds != "all":
        raise ValueError("observed_objective_over_rounds requires rounds: all and an explicit run_series map.")
    if context.run_id is not None:
        raise ValueError("observed_objective_over_rounds pins runs through run_series; do not also set run_id.")
    run_series = params.get("run_series")
    if not isinstance(run_series, Mapping):
        raise ValueError("observed_objective_over_rounds requires params.run_series as a mapping.")

    history = load_observed_objective_history(
        outputs_dir=resolve_outputs_dir(context),
        selection_view_id=context.selection_view_id,
        run_series=run_series,
    )
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_plot_style()
    show_cumulative_best = get_bool(params, ["show_cumulative_best"], True)
    zero_boundary = get_bool(params, ["zero_boundary"], False)
    summary = history.summary.copy()
    summary["batch_position"] = np.arange(len(summary), dtype=float)
    positions = {
        (int(row.observed_round), str(row.batch_id)): float(row.batch_position)
        for row in summary.itertuples(index=False)
    }
    candidates = history.frame.copy()
    candidates["batch_position"] = [
        positions[(int(round_k), str(batch_id))]
        for round_k, batch_id in zip(candidates["observed_round"], candidates["batch_id"])
    ]
    candidates["plot_x"] = _deterministic_jitter(candidates)

    figsize = tuple(params.get("figsize_in", DEFAULT_SQUARE_FIGSIZE))
    if len(figsize) != 2:
        raise ValueError("observed_objective_over_rounds figsize_in must contain two values.")
    fig, ax = plt.subplots(figsize=figsize)
    apply_notebook_axes_style(ax, square=False)
    candidate_color = "#969696"
    summary_color = categorical_color(0)
    ax.scatter(
        candidates["plot_x"],
        candidates["objective_value"],
        s=52,
        facecolors="white",
        edgecolors=candidate_color,
        linewidths=1.0,
        alpha=0.98,
        zorder=3,
    )
    cap_half_width = 0.08
    median_half_width = 0.20
    for row in summary.itertuples(index=False):
        x = float(row.batch_position)
        low = float(row.between_candidate_q25)
        high = float(row.between_candidate_q75)
        median = float(row.batch_median)
        ax.vlines(x, low, high, color=summary_color, linewidth=2.0, zorder=4)
        ax.hlines([low, high], x - cap_half_width, x + cap_half_width, color=summary_color, linewidth=1.6, zorder=4)
        ax.hlines(median, x - median_half_width, x + median_half_width, color=summary_color, linewidth=3.0, zorder=5)
    if show_cumulative_best:
        ax.plot(
            summary["batch_position"],
            summary["cumulative_best"],
            color="#D55E00",
            marker="D",
            markersize=5.5,
            linewidth=1.8,
            zorder=4,
        )
    if zero_boundary:
        ax.axhline(0.0, color="#4D4D4D", linestyle="--", linewidth=1.1, alpha=0.85, zorder=1)

    ax.set_xticks(summary["batch_position"].tolist())
    ax.set_xticklabels(
        [
            f"{pretty_batch_label(str(row.batch_id))}\nRound {int(row.observed_round)} · n={int(row.candidate_count)}"
            for row in summary.itertuples(index=False)
        ]
    )
    ax.set_xlabel("Observed batch")
    metric_label = _observed_metric_label(params, history.score_channel)
    ax.set_ylabel(metric_label)
    default_title = pretty_title(f"{history.selection_view_id} observed {history.score_channel} by batch")
    ax.set_title(pretty_title(params.get("title", default_title)), loc="center")
    ax.margins(x=0.10)

    candidate_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=candidate_color,
        markeredgewidth=1.0,
        markersize=7,
    )
    median_handle = Line2D([], [], color=summary_color, linewidth=3.0)
    iqr_handle = Line2D([], [], color=summary_color, linewidth=1.8, marker="|", markersize=9)
    handles = [candidate_handle, median_handle, iqr_handle]
    labels = ["Candidate", "Batch median", "Between-candidate IQR"]
    if show_cumulative_best:
        handles.append(Line2D([], [], color="#D55E00", marker="D", linewidth=1.8, markersize=5.5))
        labels.append("Cumulative best")
    if not _place_evidence_legend(fig, ax, handles=handles, labels=labels):
        fig.tight_layout(pad=0.35)

    context.artifact_metadata["observed_objective_history"] = {
        "schema_version": "opal.observed_objective_history_plot.v1",
        "selection_view_id": history.selection_view_id,
        "objective_name": history.objective_name,
        "score_ref": history.score_ref,
        "objective_mode": history.objective_mode,
        "y_space": history.y_space,
        "comparability_sha256": history.comparability_sha256,
    }
    out = context.output_dir / context.filename
    context.output_dir.mkdir(parents=True, exist_ok=True)
    save_notebook_square_figure(fig, out, dpi=context.dpi, tight=False)
    plt.close(fig)

    if context.save_data:
        context.save_df(_tidy_frame(history.frame, summary))


def _deterministic_jitter(candidates: pd.DataFrame) -> np.ndarray:
    jittered = pd.Series(index=candidates.index, dtype=float)
    for (_round_k, _batch_id), batch in candidates.groupby(["observed_round", "batch_id"], sort=True):
        ordered = batch.sort_values("id", kind="stable")
        offsets = np.array([0.0]) if len(ordered) == 1 else np.linspace(-0.14, 0.14, len(ordered))
        jittered.loc[ordered.index] = ordered["batch_position"].to_numpy(dtype=float) + offsets
    return jittered.loc[candidates.index].to_numpy(dtype=float)


def _observed_metric_label(params: Mapping[str, object], score_channel: str) -> str:
    if any(params.get(key) not in (None, "") for key in ("metric_label", "score_label", "y_label", "axis_label")):
        return plot_metric_label(params, score_channel)
    return math_label(score_channel)


def _place_evidence_legend(fig, ax, *, handles: list[object], labels: list[str]) -> bool:
    if len(labels) <= 3:
        return legend_below_single_row(fig, ax, handles=handles, labels=labels, bottom=0.12)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1), pad=0.35)
    return True


def _tidy_frame(candidates: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        "observed_round",
        "batch_id",
        "id",
        "display_label",
        "objective_value",
        "selection_view_id",
        "objective_name",
        "score_ref",
        "objective_mode",
        "y_space",
        "evidence_as_of_round",
        "evidence_run_id",
        "evidence_observed_events_sha256",
    ]
    candidate_rows = candidates.loc[:, candidate_columns].copy()
    candidate_rows.insert(0, "row_kind", "candidate")
    summary_rows = summary.drop(columns=["batch_position"]).copy()
    summary_rows.insert(0, "row_kind", "batch_summary")
    return pd.concat([candidate_rows, summary_rows], ignore_index=True, sort=False)


__all__ = ["render"]
