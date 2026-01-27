"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_setpoint_decomposition.py

Setpoint decomposition plot for a single record's SFXI prediction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from ..analysis.dashboard.charts import sfxi_setpoint_decomposition
from ..analysis.dashboard.util import list_series_to_numpy
from ..analysis.facade import load_predictions_with_setpoint, read_runs
from ..core.utils import ExitCodes, OpalError
from ..objectives import sfxi_math
from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._param_utils import get_float, get_str
from .sfxi_diag_data import resolve_run_id, resolve_single_round


@register_plot(
    "sfxi_setpoint_decomposition",
    meta=PlotMeta(
        summary="Per-state setpoint residual and intensity contribution (single record).",
        params={
            "record_id": "Record id to visualize (required).",
            "delta": "Log2 intensity offset delta (default 0.0).",
        },
        requires=["pred__y_hat_model", "obj__diag__setpoint"],
        notes=["Reads outputs/ledger/predictions and run metadata (setpoint)."],
    ),
)
def render(context, params: dict) -> None:
    outputs_dir = resolve_outputs_dir(context)
    runs_df = read_runs(outputs_dir / "ledger" / "runs.parquet")
    round_k = resolve_single_round(runs_df, round_selector=context.rounds)
    run_id = resolve_run_id(runs_df, round_k=round_k, run_id=context.run_id)

    record_id = get_str(params, ["record_id", "id"], None)
    if not record_id:
        raise OpalError("record_id is required for setpoint decomposition.", ExitCodes.BAD_ARGS)
    delta = get_float(params, ["delta", "intensity_log2_offset_delta"], 0.0)

    df = load_predictions_with_setpoint(
        outputs_dir,
        {"id", "pred__y_hat_model", "obj__diag__setpoint"},
        round_selector=round_k,
        run_id=run_id,
        require_run_id=False,
    )
    df = df.filter(df["id"] == str(record_id))
    if df.is_empty():
        raise OpalError(f"No prediction found for id={record_id!r}.", ExitCodes.BAD_ARGS)

    vec = list_series_to_numpy(df.get_column("pred__y_hat_model"), expected_len=8)
    if vec is None or vec.shape[0] == 0:
        raise OpalError("Invalid pred__y_hat_model vectors.", ExitCodes.CONTRACT_VIOLATION)
    y_hat = vec[0]
    if y_hat.shape[0] < 8:
        raise OpalError("pred__y_hat_model must have length >= 8.", ExitCodes.CONTRACT_VIOLATION)
    v_hat = y_hat[0:4]
    y_star = y_hat[4:8]

    setpoint_raw = df.get_column("obj__diag__setpoint").to_list()
    if not setpoint_raw:
        raise OpalError("Missing obj__diag__setpoint for selected record.", ExitCodes.CONTRACT_VIOLATION)
    setpoint = sfxi_math.parse_setpoint_vector(setpoint_raw[0])

    fig = sfxi_setpoint_decomposition.make_setpoint_decomposition_figure(
        v_hat=v_hat,
        y_star=y_star,
        setpoint=np.asarray(setpoint, dtype=float),
        delta=float(delta),
        subtitle=f"id={record_id} · R={round_k}",
    )
    out_dir = context.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / context.filename, dpi=context.dpi, format=context.format)
