"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_intensity_scaling.py

Intensity scaling diagnostics for SFXI setpoints.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import polars as pl

from ..analysis.dashboard.charts import sfxi_intensity_scaling
from ..analysis.dashboard.util import list_series_to_numpy
from ..analysis.facade import load_predictions_with_setpoint, read_labels, read_runs
from ..analysis.sfxi.setpoint_sweep import sweep_setpoints
from ..analysis.sfxi.state_order import STATE_ORDER
from ..core.utils import ExitCodes, OpalError
from ..objectives import sfxi_math
from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._param_utils import get_bool, get_float, get_int, get_str
from .sfxi_diag_data import (
    labels_current_round,
    parse_exponents_from_runs,
    parse_setpoint_from_runs,
    resolve_run_id,
    resolve_single_round,
)


@register_plot(
    "sfxi_intensity_scaling",
    meta=PlotMeta(
        summary="Intensity scaling diagnostics (denom + clipping + E_raw distribution).",
        params={
            "y_col": "Label vector column (default y_obs).",
            "percentile": "Denom percentile (default 95).",
            "min_n": "Min labels required (default 5).",
            "eps": "Epsilon for denom (default 1e-8).",
            "delta": "Log2 intensity offset delta (default 0.0).",
            "include_pool": "Compute pool clip fractions (default false).",
        },
        requires=["labels.parquet", "runs.parquet"],
        notes=["Uses current-round labels for scaling; pool optional."],
    ),
)
def render(context, params: dict) -> None:
    outputs_dir = resolve_outputs_dir(context)
    runs_df = read_runs(outputs_dir / "ledger" / "runs.parquet")
    round_k = resolve_single_round(runs_df, round_selector=context.rounds)
    run_id = resolve_run_id(runs_df, round_k=round_k, run_id=context.run_id)

    y_col = get_str(params, ["y_col", "y_column"], "y_obs")
    percentile = get_int(params, ["percentile", "p"], 95)
    min_n = get_int(params, ["min_n"], 5)
    eps = get_float(params, ["eps"], 1.0e-8)
    delta = get_float(params, ["delta", "intensity_log2_offset_delta"], 0.0)
    include_pool = get_bool(params, ["include_pool", "pool"], False)

    labels_df = read_labels(outputs_dir / "ledger" / "labels.parquet")
    labels_df = labels_current_round(labels_df, round_k=round_k)
    if labels_df.is_empty():
        raise OpalError("No labels available for intensity scaling diagnostics.", ExitCodes.BAD_ARGS)
    if y_col not in labels_df.columns:
        raise OpalError(f"labels.parquet missing column: {y_col}", ExitCodes.CONTRACT_VIOLATION)

    labels_vec = list_series_to_numpy(labels_df.get_column(y_col), expected_len=8)
    if labels_vec is None:
        raise OpalError("Invalid label vectors (expected length-8 values).", ExitCodes.CONTRACT_VIOLATION)

    setpoint = parse_setpoint_from_runs(runs_df.filter(pl.col("as_of_round") == int(round_k)))
    beta, gamma = parse_exponents_from_runs(runs_df.filter(pl.col("as_of_round") == int(round_k)))

    pool_vec = None
    if include_pool:
        pred_df = load_predictions_with_setpoint(
            outputs_dir,
            {"pred__y_hat_model"},
            round_selector=round_k,
            run_id=run_id,
            require_run_id=False,
        )
        pool_vec = list_series_to_numpy(pred_df.get_column("pred__y_hat_model"), expected_len=8)
        if pool_vec is None:
            raise OpalError("Invalid pool vectors (expected length-8 values).", ExitCodes.CONTRACT_VIOLATION)

    sweep_df = sweep_setpoints(
        labels_vec8=labels_vec,
        current_setpoint=setpoint,
        percentile=percentile,
        min_n=min_n,
        eps=eps,
        delta=delta,
        beta=beta,
        gamma=gamma,
        pool_vec8=pool_vec,
        state_order=STATE_ORDER,
    )

    label_effect_raw, _ = sfxi_math.effect_raw_from_y_star(
        labels_vec[:, 4:8],
        setpoint,
        delta=delta,
        eps=eps,
        state_order=STATE_ORDER,
    )

    pool_effect_raw = None
    if pool_vec is not None:
        pool_effect_raw, _ = sfxi_math.effect_raw_from_y_star(
            pool_vec[:, 4:8],
            setpoint,
            delta=delta,
            eps=eps,
            state_order=STATE_ORDER,
        )

    denom_note = f"denom={percentile}th pct E_raw (min_n={min_n})"
    fig = sfxi_intensity_scaling.make_intensity_scaling_figure(
        sweep_df,
        label_effect_raw=label_effect_raw,
        pool_effect_raw=pool_effect_raw,
        subtitle=f"R={round_k} · labels={labels_vec.shape[0]} · {denom_note}",
    )
    out_dir = context.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / context.filename, dpi=context.dpi, format=context.format)
