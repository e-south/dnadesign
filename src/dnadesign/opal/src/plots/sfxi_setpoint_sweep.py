"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_setpoint_sweep.py

Setpoint sweep diagnostics for SFXI objectives (labels-only).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import polars as pl

from ..analysis.dashboard.charts import sfxi_setpoint_sweep
from ..analysis.dashboard.util import list_series_to_numpy
from ..analysis.facade import read_labels, read_runs
from ..analysis.sfxi.setpoint_sweep import sweep_setpoints
from ..analysis.sfxi.state_order import STATE_ORDER
from ..core.utils import ExitCodes, OpalError
from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._param_utils import get_float, get_int, get_str
from .sfxi_diag_data import labels_current_round, parse_setpoint_from_runs, resolve_single_round


@register_plot(
    "sfxi_setpoint_sweep",
    meta=PlotMeta(
        summary="Setpoint sweep diagnostics for labels.",
        params={
            "y_col": "Label vector column (default y_obs).",
            "top_k": "Top-K used for sweep metrics (default 5).",
            "tau": "Threshold for logic_fidelity fraction (default 0.8).",
            "percentile": "Denom percentile (default 95).",
            "min_n": "Min labels required (default 5).",
            "eps": "Epsilon for denom (default 1e-8).",
            "delta": "Log2 intensity offset delta (default 0.0).",
        },
        requires=["labels.parquet", "runs.parquet"],
        notes=["Uses current-round labels for denom/logic metrics (objective-consistent)."],
    ),
)
def render(context, params: dict) -> None:
    outputs_dir = resolve_outputs_dir(context)
    runs_df = read_runs(outputs_dir / "ledger" / "runs.parquet")
    round_k = resolve_single_round(runs_df, round_selector=context.rounds)

    y_col = get_str(params, ["y_col", "y_column"], "y_obs")
    top_k = get_int(params, ["top_k"], 5)
    tau = get_float(params, ["tau"], 0.8)
    percentile = get_int(params, ["percentile", "p"], 95)
    min_n = get_int(params, ["min_n"], 5)
    eps = get_float(params, ["eps"], 1.0e-8)
    delta = get_float(params, ["delta", "intensity_log2_offset_delta"], 0.0)

    labels_df = read_labels(outputs_dir / "ledger" / "labels.parquet")
    labels_df = labels_current_round(labels_df, round_k=round_k)
    if labels_df.is_empty():
        raise OpalError("No labels available for setpoint sweep.", ExitCodes.BAD_ARGS)
    if y_col not in labels_df.columns:
        raise OpalError(f"labels.parquet missing column: {y_col}", ExitCodes.CONTRACT_VIOLATION)

    labels_vec = list_series_to_numpy(labels_df.get_column(y_col), expected_len=8)
    if labels_vec is None:
        raise OpalError("Invalid label vectors (expected length-8 values).", ExitCodes.CONTRACT_VIOLATION)

    setpoint = parse_setpoint_from_runs(runs_df.filter(pl.col("as_of_round") == int(round_k)))

    sweep_df = sweep_setpoints(
        labels_vec8=labels_vec,
        current_setpoint=setpoint,
        percentile=percentile,
        min_n=min_n,
        eps=eps,
        delta=delta,
        top_k=top_k,
        tau=tau,
        state_order=STATE_ORDER,
    )

    denom_note = f"denom={percentile}th pct E_raw (min_n={min_n})"
    fig = sfxi_setpoint_sweep.make_setpoint_sweep_figure(
        sweep_df,
        metrics=[
            "median_logic_fidelity",
            "top_k_logic_fidelity",
            "frac_logic_fidelity_gt_tau",
            "clip_hi_fraction",
        ],
        subtitle=f"R={round_k} · labels={labels_vec.shape[0]} · {denom_note}",
    )
    out_dir = context.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / context.filename, dpi=context.dpi, format=context.format)
