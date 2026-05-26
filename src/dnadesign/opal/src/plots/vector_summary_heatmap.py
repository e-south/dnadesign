"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/vector_summary_heatmap.py

Generic vector-over-rounds heatmap primitive for OPAL ledger predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..registries.plots import PlotMeta, register_plot
from ._cohort_utils import positive_ranks, selected_mask
from ._events_util import load_events, load_events_with_setpoint, resolve_outputs_dir
from ._mpl_utils import (
    DEFAULT_SQUARE_FIGSIZE,
    add_flush_colorbar,
    apply_notebook_axes_style,
    apply_plot_style,
    apply_y_axis_scale,
    ensure_mpl_config_dir,
    pretty_label,
    pretty_title,
    save_notebook_square_figure,
    sequential_colormap,
    wrap_plot_title,
)


@register_plot(
    "vector_summary_heatmap",
    meta=PlotMeta(
        summary="Mean vector-channel summary by reference/cohort/round.",
        params={
            "vector_field": "List-valued prediction field (default pred__y_hat_model).",
            "cohort": "selected|top_k|all_pool (default selected).",
            "top_k": "Rank cutoff for top_k cohort (default 10).",
            "include_reference_vector": "Include a reference-vector row (default false).",
            "reference_vector": ("Optional explicit vector baseline; if omitted, objective setpoint metadata is used."),
            "reference_label": "Optional y-axis label for the reference row.",
            "channel_labels": "Optional channel labels, same length as the vector.",
            "aggregation": "Currently mean.",
            "reference_mse_panel": "When a reference vector is present, add a round-wise MSE panel (default false).",
            "cmap": "Matplotlib colormap (default opal_seafoam: low values white, high values dark seafoam).",
            "value_label": "Colorbar label (default Mean predicted response for prediction vectors).",
        },
        requires=["as_of_round", "run_id", "pred__y_hat_model"],
        notes=["SFXI can configure semantic channel labels, but the primitive is vector-shaped."],
        data_shape="vector over rounds plus optional reference-distance series",
        tidy_schema=["row_type", "round", "cohort", "channel", "value", "n"],
        objective_family="generic",
        data_layer="predictions_vector",
        round_scope="round_history",
        failure_modes=[
            "missing vector field",
            "vector length mismatch",
            "missing reference vector when include_reference_vector is true",
            "selected/top_k cohort columns missing",
            "non-finite vector values",
        ],
    ),
)
def render(context, params: dict) -> None:
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import pandas as pd

    apply_plot_style()
    vector_field = str(params.get("vector_field", "pred__y_hat_model"))
    cohort = str(params.get("cohort", "selected")).strip().lower()
    if cohort not in {"selected", "top_k", "all_pool"}:
        raise ValueError("cohort must be one of selected, top_k, all_pool.")
    top_k = int(params.get("top_k", 10))
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    aggregation = str(params.get("aggregation", "mean")).strip().lower()
    if aggregation != "mean":
        raise ValueError("vector_summary_heatmap currently supports aggregation: mean.")
    explicit_reference = _explicit_reference_vector_param(params)
    include_reference = bool(
        params.get(
            "include_reference_vector",
            params.get("include_target_vector", params.get("include_setpoint", False)),
        )
    )
    show_reference_mse = bool(params.get("reference_mse_panel", False))
    font_size = float(params.get("font_size", 13))
    title_font_size = float(params.get("title_font_size", font_size))
    tick_font_size = float(params.get("tick_font_size", font_size))

    need = {"as_of_round", "run_id", vector_field}
    row_filters = []
    if cohort == "selected":
        need.add("sel__is_selected")
        row_filters.append({"column": "sel__is_selected", "op": "eq", "value": True})
    elif cohort == "top_k":
        need.add("sel__rank_competition")
        row_filters.append({"column": "sel__rank_competition", "op": "lte", "value": top_k})
    outputs_dir = resolve_outputs_dir(context)
    if include_reference and explicit_reference is None:
        df = load_events_with_setpoint(
            outputs_dir,
            need,
            round_selector=context.rounds,
            run_id=context.run_id,
            row_filters=row_filters,
        )
    else:
        df = load_events(
            outputs_dir, need, round_selector=context.rounds, run_id=context.run_id, row_filters=row_filters
        )
    if df.empty:
        raise ValueError("vector_summary_heatmap had zero rows after round/run filtering.")
    if vector_field not in df.columns:
        raise ValueError(f"vector_summary_heatmap missing vector field: {vector_field}")
    df = _cohort_frame(df, cohort=cohort, top_k=top_k)
    if df.empty:
        raise ValueError(f"vector_summary_heatmap cohort {cohort!r} has no rows.")

    vectors = [_coerce_vector(value, field=vector_field) for value in df[vector_field].tolist()]
    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        raise ValueError("vector_summary_heatmap vector length mismatch.")
    df = df.copy()
    df["__vector__"] = vectors

    labels = params.get("channel_labels")
    if labels is None:
        channel_labels = [f"ch_{index}" for index in range(dim)]
    else:
        channel_labels = [str(label) for label in labels]
        if len(channel_labels) != dim:
            raise ValueError(f"channel_labels length {len(channel_labels)} does not match vector length {dim}.")

    rows = []
    matrix_rows = []
    y_labels = []
    reference = None
    if include_reference:
        if explicit_reference is not None:
            reference = _coerce_vector(explicit_reference, field="params.reference_vector")
        elif "obj__diag__setpoint" not in df.columns:
            raise ValueError("include_reference_vector requires reference_vector or objective metadata setpoint.")
        else:
            setpoints = [
                _coerce_vector(value, field="obj__diag__setpoint") for value in df["obj__diag__setpoint"].dropna()
            ]
            if not setpoints:
                raise ValueError("include_reference_vector requested, but no objective setpoint vector was found.")
            reference = setpoints[0]
        if len(reference) != dim:
            raise ValueError(f"reference_vector length {len(reference)} does not match vector length {dim}.")
        reference_label = str(params.get("reference_label", "reference")).strip() or "reference"
        matrix_rows.append(reference)
        y_labels.append(reference_label)
        for channel, value in zip(channel_labels, reference):
            rows.append(
                {
                    "row_type": "reference_vector",
                    "round": None,
                    "cohort": reference_label,
                    "channel": channel,
                    "value": value,
                    "n": None,
                }
            )

    mse_rows = []
    for round_index, sub in df.groupby("as_of_round"):
        arr = np.asarray(sub["__vector__"].to_list(), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(f"round {round_index} vector matrix has invalid shape {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"round {round_index} contains non-finite vector values.")
        summary = arr.mean(axis=0)
        n_selected = int(arr.shape[0])
        matrix_rows.append(summary.tolist())
        y_labels.append(_round_row_label(int(round_index), cohort=cohort, n=n_selected))
        if reference is not None:
            ref_arr = np.asarray(reference, dtype=float)
            mse_rows.append(
                {
                    "row_type": "reference_mse",
                    "round": int(round_index),
                    "cohort": cohort,
                    "channel": "mse",
                    "value": float(np.mean((summary - ref_arr) ** 2)),
                    "n": n_selected,
                }
            )
        for channel, value in zip(channel_labels, summary.tolist()):
            rows.append(
                {
                    "row_type": "round",
                    "round": int(round_index),
                    "cohort": cohort,
                    "channel": channel,
                    "value": float(value),
                    "n": n_selected,
                }
            )

    matrix = np.asarray(matrix_rows, dtype=float)
    if show_reference_mse and reference is not None:
        figsize = tuple(params.get("figsize_in", (10.8, 5.2)))
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.045, 0.34, 1.12], wspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])
        spacer_ax = fig.add_subplot(gs[0, 2])
        spacer_ax.axis("off")
        ax_mse = fig.add_subplot(gs[0, 3])
    else:
        figsize = tuple(params.get("figsize_in", DEFAULT_SQUARE_FIGSIZE))
        fig, ax = plt.subplots(figsize=figsize)
        cax = None
        ax_mse = None
    cmap = sequential_colormap(params.get("cmap", "opal_seafoam"))
    masked_matrix = np.ma.masked_invalid(matrix)
    x_edges = np.arange(dim + 1)
    y_edges = np.arange(len(y_labels) + 1)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        masked_matrix,
        cmap=cmap,
        edgecolors="white",
        linewidth=0.75,
        shading="flat",
    )
    ax.set_xlim(0, dim)
    ax.set_ylim(len(y_labels), 0)
    ax.set_aspect("equal", adjustable="box")
    apply_notebook_axes_style(ax, grid=False, square=False)
    ax.set_xticks(np.arange(dim) + 0.5)
    ax.set_xticklabels(
        _heatmap_channel_tick_labels(channel_labels),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_font_size,
    )
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_yticklabels(y_labels, fontsize=tick_font_size)
    channel_axis_label = str(params.get("channel_axis_label", "Vector channel")).strip()
    if channel_axis_label:
        ax.set_xlabel(channel_axis_label, fontsize=font_size)
    ax.set_title(pretty_title(params.get("title", "Vector summary heatmap")), fontsize=title_font_size)
    value_label = str(params.get("value_label", f"{pretty_label(aggregation)} predicted response"))
    if cax is None:
        add_flush_colorbar(fig, ax, im, label=value_label)
    else:
        fig.subplots_adjust(left=0.11, right=0.965, bottom=0.24, top=0.84, wspace=0.08)
        fig.canvas.draw()
        heatmap_box = ax.get_position()
        cbar_width = max(0.014, heatmap_box.width * 0.035)
        cbar_pad = 0.012
        cax.set_position([heatmap_box.x1 + cbar_pad, heatmap_box.y0, cbar_width, heatmap_box.height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(_short_colorbar_title(value_label), rotation=90, labelpad=8, va="center")
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.tick_params(labelsize=tick_font_size)
        cbar.ax.yaxis.label.set_size(font_size)
    if ax_mse is not None:
        apply_notebook_axes_style(ax_mse, square=False)
        mse_frame = pd.DataFrame(mse_rows).sort_values("round")
        ax_mse.plot(
            mse_frame["round"].astype(int),
            mse_frame["value"].astype(float),
            color="#005F56",
            marker="o",
            linewidth=2.2,
            markersize=6,
        )
        ax_mse.set_xlabel("Round", fontsize=font_size)
        mse_label = str(params.get("reference_mse_metric_label") or "MSE = mean((mean selected y_hat - reference)^2)")
        ax_mse.set_ylabel(mse_label, fontsize=font_size, labelpad=18)
        ax_mse.set_title(
            wrap_plot_title(pretty_title(params.get("reference_mse_title", "Target-vector MSE")), width=24),
            fontsize=title_font_size,
        )
        ax_mse.set_xticks(mse_frame["round"].astype(int).tolist())
        ax_mse.tick_params(axis="both", labelsize=tick_font_size)
        apply_y_axis_scale(
            ax_mse,
            limits=params.get("reference_mse_y_limits", params.get("reference_mse_limits")),
            reference_lines=params.get("reference_mse_reference_lines"),
            include_zero_tick=bool(params.get("reference_mse_include_zero_tick", True)),
        )
        try:
            ax_mse.set_box_aspect(1.0)
        except Exception:
            pass
    left_margin = 0.13 if ax_mse is not None else 0.18
    if ax_mse is None:
        fig.subplots_adjust(left=left_margin, right=0.96, bottom=0.22, top=0.86)
    out = context.output_dir / context.filename
    save_notebook_square_figure(fig, out, dpi=context.dpi, tight=False)
    plt.close(fig)

    if context.save_data:
        if mse_rows:
            rows.extend(mse_rows)
        context.save_df(pd.DataFrame(rows))


def _cohort_frame(df, *, cohort: str, top_k: int):
    if cohort == "all_pool":
        return df.copy()
    if cohort == "selected":
        if "sel__is_selected" not in df.columns:
            raise ValueError("selected cohort requires sel__is_selected.")
        return df[selected_mask(df["sel__is_selected"])].copy()
    if cohort == "top_k":
        if "sel__rank_competition" not in df.columns:
            raise ValueError("top_k cohort requires sel__rank_competition.")
        return df[positive_ranks(df["sel__rank_competition"]) <= int(top_k)].copy()
    raise ValueError(f"Unknown cohort: {cohort}")


def _round_row_label(round_index: int, *, cohort: str, n: int) -> str:
    cohort_label = pretty_label(cohort)
    if str(cohort).strip().lower() == "selected":
        return f"R{int(round_index)} (n={int(n)})"
    return f"R{int(round_index)}: {cohort_label} (n={int(n)})"


def _short_colorbar_title(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    return text.replace("predicted ", "").replace("Predicted ", "")


def _heatmap_channel_tick_labels(labels: Sequence[str]) -> list[str]:
    return [str(label) for label in labels]


def _coerce_vector(value: object, *, field: str) -> list[float]:
    raw = value
    if hasattr(raw, "as_py"):
        raw = raw.as_py()
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError(f"{field} must contain vectors; got {type(value).__name__}.")
    out = [float(item) for item in raw]
    if not out:
        raise ValueError(f"{field} contains an empty vector.")
    if not all(math.isfinite(item) for item in out):
        raise ValueError(f"{field} contains non-finite values.")
    return out


def _explicit_reference_vector_param(params: dict) -> object | None:
    for key in ("reference_vector", "target_vector", "setpoint", "setpoint_vector"):
        if key in params:
            return params[key]
    return None
