"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_support_diagnostics.py

Logic-space support diagnostics for SFXI predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..analysis.dashboard.charts import sfxi_support_diagnostics
from ..analysis.dashboard.util import list_series_to_numpy
from ..analysis.facade import load_predictions_with_setpoint, read_labels, read_runs
from ..analysis.sfxi.state_order import STATE_ORDER
from ..analysis.sfxi.support import dist_to_labeled_logic
from ..core.utils import ExitCodes, OpalError
from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._param_utils import get_int, get_str, normalize_metric_field
from .sfxi_diag_data import labels_asof_round, resolve_run_id, resolve_single_round


@register_plot(
    "sfxi_support_diagnostics",
    meta=PlotMeta(
        summary="Distance to labeled logic vs score diagnostics.",
        params={
            "y_axis": "Metric for Y-axis (default score).",
            "hue": "Metric for color (default effect_scaled).",
            "sample_n": "Optional sample size for plotting.",
            "seed": "Random seed for sampling (default 0).",
            "batch_size": "Batch size for distance computation (default 2048).",
        },
        requires=["pred__y_hat_model", "pred__y_obj_scalar"],
        notes=["Uses labels-as-of round for support distances."],
    ),
)
def render(context, params: dict) -> None:
    outputs_dir = resolve_outputs_dir(context)
    runs_df = read_runs(outputs_dir / "ledger" / "runs.parquet")
    round_k = resolve_single_round(runs_df, round_selector=context.rounds)
    run_id = resolve_run_id(runs_df, round_k=round_k, run_id=context.run_id)

    y_axis = normalize_metric_field(get_str(params, ["y_axis", "y_field", "y"], "score"))
    hue = normalize_metric_field(get_str(params, ["hue", "color", "color_by"], "effect_scaled"))
    sample_n = get_int(params, ["sample_n", "n", "sample"], 0)
    seed = get_int(params, ["seed"], 0)
    batch_size = get_int(params, ["batch_size"], 2048)

    need = {"id", "pred__y_hat_model", "pred__y_obj_scalar", "sel__is_selected"}
    if y_axis:
        need.add(y_axis)
    if hue:
        need.add(hue)

    pred_df = load_predictions_with_setpoint(
        outputs_dir,
        need,
        round_selector=round_k,
        run_id=run_id,
        require_run_id=False,
    )
    if pred_df.is_empty():
        raise OpalError("No predictions available for support diagnostics.", ExitCodes.BAD_ARGS)
    if y_axis not in pred_df.columns:
        raise OpalError(f"Missing y-axis column: {y_axis}", ExitCodes.CONTRACT_VIOLATION)
    if hue and hue not in pred_df.columns:
        raise OpalError(f"Missing hue column: {hue}", ExitCodes.CONTRACT_VIOLATION)

    labels_df = read_labels(outputs_dir / "ledger" / "labels.parquet")
    labels_df = labels_asof_round(labels_df, round_k=round_k)
    if labels_df.is_empty():
        raise OpalError("No labels available for support diagnostics.", ExitCodes.BAD_ARGS)
    if "y_obs" not in labels_df.columns:
        raise OpalError("labels.parquet missing y_obs.", ExitCodes.CONTRACT_VIOLATION)

    labels_vec = list_series_to_numpy(labels_df.get_column("y_obs"), expected_len=8)
    if labels_vec is None:
        raise OpalError("Invalid label vectors (expected length-8 values).", ExitCodes.CONTRACT_VIOLATION)
    label_logic = np.asarray(labels_vec[:, 0:4], dtype=float)

    pred_vec = list_series_to_numpy(pred_df.get_column("pred__y_hat_model"), expected_len=8)
    if pred_vec is None:
        raise OpalError("Invalid prediction vectors (expected length-8 values).", ExitCodes.CONTRACT_VIOLATION)
    pred_logic = np.asarray(pred_vec[:, 0:4], dtype=float)

    distances = dist_to_labeled_logic(
        pred_logic,
        label_logic,
        state_order=STATE_ORDER,
        batch_size=batch_size,
    )
    df_plot = pred_df.with_columns(pl.Series("dist_to_labeled_logic", distances))

    total = df_plot.height
    subtitle = None
    if sample_n > 0 and total > sample_n:
        df_plot = df_plot.sample(n=sample_n, seed=seed, shuffle=True)
        subtitle = f"sampled {df_plot.height}/{total}"

    fig = sfxi_support_diagnostics.make_support_diagnostics_figure(
        df_plot,
        x_col="dist_to_labeled_logic",
        y_col=y_axis,
        hue_col=hue,
        selected_col="sel__is_selected",
        subtitle=subtitle,
    )
    out_dir = context.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / context.filename, dpi=context.dpi, format=context.format)
