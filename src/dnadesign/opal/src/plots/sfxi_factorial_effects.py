"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_factorial_effects.py

Plots factorial effects for SFXI logic vectors from ledger predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import polars as pl

from ..analysis.dashboard.charts import sfxi_factorial_effects
from ..analysis.facade import load_predictions_with_setpoint, read_labels, read_runs
from ..core.utils import ExitCodes, OpalError
from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._param_utils import get_bool, get_int, get_str, reject_params
from .sfxi_diag_data import labels_asof_round, resolve_run_id, resolve_single_round


@register_plot(
    "sfxi_factorial_effects",
    meta=PlotMeta(
        summary="Factorial-effects map from predicted SFXI logic vectors.",
        params={
            "size_by": "Column for point size (default obj__effect_scaled).",
            "top_k": "Optional Top-K threshold for overlay (default 10).",
            "include_labels": "Overlay labeled records (default true).",
            "rasterize_at": "Rasterize scatter above this count (default None).",
        },
        requires=["pred__y_hat_model", "sel__is_selected", "sel__rank_competition"],
        notes=["Reads outputs/ledger/predictions and labels.parquet (optional) for overlays."],
    ),
)
def render(context, params: dict) -> None:
    outputs_dir = resolve_outputs_dir(context)
    runs_df = read_runs(outputs_dir / "ledger" / "runs.parquet")
    round_k = resolve_single_round(runs_df, round_selector=context.rounds)
    run_id = resolve_run_id(runs_df, round_k=round_k, run_id=context.run_id)

    size_by = get_str(params, ["size_by", "size", "size_field"], "obj__effect_scaled")
    top_k = get_int(params, ["top_k"], 10)
    include_labels = get_bool(params, ["include_labels", "labels"], True)
    rasterize_at = params.get("rasterize_at", None)
    if rasterize_at is not None:
        rasterize_at = int(rasterize_at)
    reject_params(params, ["sample_n", "sample", "n", "seed"], ctx="sfxi_factorial_effects")

    need = {"id", "pred__y_hat_model", "sel__is_selected", "sel__rank_competition"}
    if size_by:
        need.add(size_by)
    df = load_predictions_with_setpoint(
        outputs_dir,
        need,
        round_selector=round_k,
        run_id=run_id,
        require_run_id=False,
    )
    if df.is_empty():
        raise OpalError("No prediction rows found for selected round.", ExitCodes.BAD_ARGS)

    if size_by and size_by not in df.columns:
        raise OpalError(f"Missing size_by column: {size_by}", ExitCodes.CONTRACT_VIOLATION)

    if include_labels:
        labels_df = read_labels(outputs_dir / "ledger" / "labels.parquet")
        labels_df = labels_asof_round(labels_df, round_k=round_k)
        label_ids = labels_df.select("id").drop_nulls().to_series().to_list()
        if label_ids:
            df = df.with_columns(pl.col("id").cast(pl.Utf8).is_in(label_ids).alias("__is_labeled"))
        else:
            df = df.with_columns(pl.lit(False).alias("__is_labeled"))
    else:
        df = df.with_columns(pl.lit(False).alias("__is_labeled"))

    df = df.with_columns((pl.col("sel__rank_competition") <= int(top_k)).alias("__is_top_k"))

    fig = sfxi_factorial_effects.make_factorial_effects_figure(
        df,
        logic_col="pred__y_hat_model",
        size_col=size_by,
        label_col="__is_labeled",
        selected_col="sel__is_selected",
        top_k_col="__is_top_k",
        subtitle=None,
        rasterize_at=rasterize_at,
    )

    out_dir = context.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / context.filename, dpi=context.dpi, format=context.format)
