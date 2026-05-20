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
from ._mpl_utils import ensure_mpl_config_dir


@register_plot(
    "vector_summary_heatmap",
    meta=PlotMeta(
        summary="Mean vector-channel summary by setpoint/cohort/round.",
        params={
            "vector_field": "List-valued prediction field (default pred__y_hat_model).",
            "cohort": "selected|top_k|all_pool (default selected).",
            "top_k": "Rank cutoff for top_k cohort (default 10).",
            "include_setpoint": "Include objective setpoint row when objective metadata has one (default false).",
            "setpoint": (
                "Optional explicit setpoint vector; overrides objective metadata when include_setpoint is true."
            ),
            "channel_labels": "Optional channel labels, same length as the vector.",
            "aggregation": "Currently mean.",
        },
        requires=["as_of_round", "run_id", "pred__y_hat_model"],
        notes=["SFXI can configure semantic channel labels, but the primitive is vector-shaped."],
        data_shape="vector over rounds",
        tidy_schema=["row_type", "round", "cohort", "channel", "value"],
        failure_modes=[
            "missing vector field",
            "vector length mismatch",
            "missing setpoint when include_setpoint is true",
            "selected/top_k cohort columns missing",
            "non-finite vector values",
        ],
    ),
)
def render(context, params: dict) -> None:
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import pandas as pd

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
    include_setpoint = bool(params.get("include_setpoint", False))

    need = {"as_of_round", "run_id", vector_field}
    if cohort == "selected":
        need.add("sel__is_selected")
    elif cohort == "top_k":
        need.add("sel__rank_competition")
    outputs_dir = resolve_outputs_dir(context)
    if include_setpoint:
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
    if include_setpoint:
        explicit_setpoint = params.get("setpoint", params.get("setpoint_vector"))
        if explicit_setpoint is not None:
            setpoint = _coerce_vector(explicit_setpoint, field="params.setpoint")
        elif "obj__diag__setpoint" not in df.columns:
            raise ValueError("include_setpoint requires objective metadata setpoint.")
        else:
            setpoints = [
                _coerce_vector(value, field="obj__diag__setpoint") for value in df["obj__diag__setpoint"].dropna()
            ]
            if not setpoints:
                raise ValueError("include_setpoint requested, but no setpoint vector was found.")
            setpoint = setpoints[0]
        if len(setpoint) != dim:
            raise ValueError(f"setpoint length {len(setpoint)} does not match vector length {dim}.")
        matrix_rows.append(setpoint)
        y_labels.append("setpoint")
        for channel, value in zip(channel_labels, setpoint):
            rows.append(
                {
                    "row_type": "setpoint",
                    "round": None,
                    "cohort": "setpoint",
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
    figsize = tuple(params.get("figsize_in", (8.0, 4.8)))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=str(params.get("cmap", "viridis")))
    ax.set_xticks(range(dim))
    ax.set_xticklabels(channel_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Vector channel")
    ax.set_title(str(params.get("title", "Vector summary heatmap")))
    fig.colorbar(im, ax=ax, label=aggregation)
    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
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
