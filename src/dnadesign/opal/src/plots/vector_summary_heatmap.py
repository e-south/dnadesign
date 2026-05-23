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
from ._mpl_utils import apply_notebook_axes_style, apply_plot_style, ensure_mpl_config_dir, save_notebook_square_figure


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
        },
        requires=["as_of_round", "run_id", "pred__y_hat_model"],
        notes=["SFXI can configure semantic channel labels, but the primitive is vector-shaped."],
        data_shape="vector over rounds",
        tidy_schema=["row_type", "round", "cohort", "channel", "value"],
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

    need = {"as_of_round", "run_id", vector_field}
    if cohort == "selected":
        need.add("sel__is_selected")
    elif cohort == "top_k":
        need.add("sel__rank_competition")
    outputs_dir = resolve_outputs_dir(context)
    if include_reference and explicit_reference is None:
        df = load_events_with_setpoint(outputs_dir, need, round_selector=context.rounds, run_id=context.run_id)
    else:
        df = load_events(outputs_dir, need, round_selector=context.rounds, run_id=context.run_id)
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
                }
            )

    for round_index, sub in df.groupby("as_of_round"):
        arr = np.asarray(sub["__vector__"].to_list(), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(f"round {round_index} vector matrix has invalid shape {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError(f"round {round_index} contains non-finite vector values.")
        summary = arr.mean(axis=0)
        matrix_rows.append(summary.tolist())
        y_labels.append(f"r{int(round_index)}:{cohort}")
        for channel, value in zip(channel_labels, summary.tolist()):
            rows.append(
                {
                    "row_type": "round",
                    "round": int(round_index),
                    "cohort": cohort,
                    "channel": channel,
                    "value": float(value),
                }
            )

    matrix = np.asarray(matrix_rows, dtype=float)
    figsize = tuple(params.get("figsize_in", (7.2, 7.2)))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=str(params.get("cmap", "viridis")))
    apply_notebook_axes_style(ax)
    ax.set_xticks(range(dim))
    ax.set_xticklabels(channel_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Vector channel")
    ax.set_title(str(params.get("title", "Vector summary heatmap")))
    fig.colorbar(im, ax=ax, label=aggregation)
    fig.tight_layout()
    out = context.output_dir / context.filename
    save_notebook_square_figure(fig, out, dpi=context.dpi)
    plt.close(fig)

    if context.save_data:
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
