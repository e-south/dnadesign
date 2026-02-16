"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_uncertainty.py

Uncertainty diagnostics using artifact model ensemble spread (RF).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..analysis.campaign import load_campaign_data
from ..analysis.dashboard.artifacts import resolve_round_artifacts
from ..analysis.dashboard.charts import sfxi_uncertainty
from ..analysis.dashboard.models import load_model_artifact, load_round_ctx_from_dir, unwrap_artifact_model
from ..analysis.dashboard.util import list_series_to_numpy
from ..analysis.facade import load_predictions_with_setpoint, read_runs
from ..analysis.sfxi.uncertainty import UncertaintyContext, compute_uncertainty, supports_uncertainty
from ..core.utils import ExitCodes, OpalError
from ..registries.plots import PlotMeta, register_plot
from ..storage.parquet_io import read_parquet_df
from ._events_util import resolve_outputs_dir
from ._param_utils import get_str, normalize_metric_field, reject_params
from .sfxi_diag_data import (
    parse_delta_from_runs,
    parse_exponents_from_runs,
    parse_setpoint_from_runs,
    resolve_run_id,
    resolve_single_round,
)


def _coerce_y_ops(value: object) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, pl.Series):
        items = value.to_list()
        if not items:
            return []
        value = items[0]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise OpalError("training__y_ops must be a list of entries.", ExitCodes.CONTRACT_VIOLATION)


@register_plot(
    "sfxi_uncertainty",
    meta=PlotMeta(
        summary="Uncertainty vs score diagnostics (artifact model).",
        params={
            "kind": "Uncertainty kind (score only).",
            "y_axis": "Metric for Y-axis (default score).",
            "hue": "Metric for color (default logic_fidelity).",
        },
        requires=["model artifact", "predictions", "records"],
        notes=["Loads artifact model from outputs/rounds/round_<r>/model/model.joblib."],
    ),
)
def render(context, params: dict) -> None:
    outputs_dir = resolve_outputs_dir(context)
    runs_df = read_runs(outputs_dir / "ledger" / "runs.parquet")
    round_k = resolve_single_round(runs_df, round_selector=context.rounds)
    run_id = resolve_run_id(runs_df, round_k=round_k, run_id=context.run_id)

    kind = get_str(params, ["kind"], "score")
    y_axis = normalize_metric_field(get_str(params, ["y_axis", "y_field", "y"], "score"))
    hue = normalize_metric_field(get_str(params, ["hue", "color", "color_by"], "logic_fidelity"))
    reject_params(
        params,
        ["sample_n", "sample", "n", "seed", "components", "reduction"],
        ctx="sfxi_uncertainty",
    )
    if kind != "score":
        raise ValueError("sfxi_uncertainty only supports kind=score.")

    run_sel = runs_df.filter(pl.col("as_of_round") == int(round_k))
    if run_id is not None and "run_id" in run_sel.columns:
        run_sel = run_sel.filter(pl.col("run_id") == str(run_id))
    if run_sel.is_empty():
        raise OpalError("No run metadata found for requested round/run.", ExitCodes.BAD_ARGS)

    run_row = run_sel.head(1)
    setpoint = parse_setpoint_from_runs(run_sel)
    beta, gamma = parse_exponents_from_runs(run_sel)
    delta = parse_delta_from_runs(run_sel)
    denom = run_row["objective__denom_value"][0] if "objective__denom_value" in run_row.columns else None
    y_ops_raw = run_row["training__y_ops"][0] if "training__y_ops" in run_row.columns else []
    y_ops = _coerce_y_ops(y_ops_raw)

    artifacts, err = resolve_round_artifacts(context.workspace.workdir, as_of_round=round_k)
    if artifacts is None or "model/model.joblib" not in artifacts:
        raise OpalError(err or "Model artifact not found.", ExitCodes.BAD_ARGS)
    model_path = Path(artifacts["model/model.joblib"])
    obj, err = load_model_artifact(model_path)
    if err:
        raise OpalError(f"Model artifact load failed: {err}", ExitCodes.BAD_ARGS)
    model = unwrap_artifact_model(obj)
    if model is None:
        raise OpalError("Unsupported model artifact format.", ExitCodes.BAD_ARGS)
    if not supports_uncertainty(model=model):
        raise OpalError("Model does not support uncertainty.", ExitCodes.BAD_ARGS)

    round_dir = Path(artifacts["round_dir"]) if "round_dir" in artifacts else None
    round_ctx, ctx_err = (None, None)
    if round_dir is not None:
        round_ctx, ctx_err = load_round_ctx_from_dir(round_dir)
    if y_ops and round_ctx is None:
        raise OpalError(ctx_err or "round_ctx.json is required to invert y-ops.", ExitCodes.BAD_ARGS)

    campaign = load_campaign_data(context.workspace.config_path, allow_dir=True)
    x_col = campaign.config.data.x_column_name
    if not x_col:
        raise OpalError("x_column_name missing from campaign config.", ExitCodes.BAD_ARGS)
    records_path = context.data_paths.get("records")
    if records_path is None:
        raise OpalError("records path not available in PlotContext.", ExitCodes.BAD_ARGS)

    need = {"id", "pred__score_selected", "obj__logic_fidelity"}
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
        raise OpalError("No predictions available for uncertainty plot.", ExitCodes.BAD_ARGS)

    records = read_parquet_df(records_path, columns=["id", x_col])
    rec_df = pl.from_pandas(records)
    df_joined = pred_df.join(rec_df, on="id", how="left")
    if x_col not in df_joined.columns:
        raise OpalError(f"records.parquet missing x column: {x_col}", ExitCodes.CONTRACT_VIOLATION)

    X = list_series_to_numpy(df_joined.get_column(x_col), expected_len=None)
    if X is None:
        raise OpalError("Invalid X vectors for uncertainty computation.", ExitCodes.CONTRACT_VIOLATION)

    ctx = UncertaintyContext(
        setpoint=np.asarray(setpoint, dtype=float),
        beta=beta,
        gamma=gamma,
        delta=delta,
        denom=None if denom is None else float(denom),
        y_ops=y_ops or [],
        round_ctx=round_ctx,
    )
    result = compute_uncertainty(
        model,
        X,
        ctx=ctx,
        batch_size=2048,
    )

    df_plot = df_joined.with_columns(pl.Series("uncertainty", result.values))
    if y_axis not in df_plot.columns:
        raise OpalError(f"Missing y-axis column: {y_axis}", ExitCodes.CONTRACT_VIOLATION)
    if hue and hue not in df_plot.columns:
        raise OpalError(f"Missing hue column: {hue}", ExitCodes.CONTRACT_VIOLATION)

    fig = sfxi_uncertainty.make_uncertainty_figure(
        df_plot,
        x_col="uncertainty",
        y_col=y_axis,
        hue_col=hue,
        subtitle=None,
    )
    out_dir = context.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / context.filename, dpi=context.dpi, format=context.format)
