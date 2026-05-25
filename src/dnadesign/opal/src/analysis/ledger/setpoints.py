"""Prediction reads joined with objective setpoint metadata."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

from ...core.utils import ExitCodes, OpalError
from .io import read_runs
from .predictions import read_predictions
from .rounds import RoundSelector


def load_predictions_with_setpoint(
    outputs_dir: Path,
    base_columns: Iterable[str],
    *,
    round_selector: RoundSelector | None = None,
    run_id: str | None = None,
    row_filters: Sequence[Mapping[str, Any]] | None = None,
    require_run_id: bool = True,
) -> pl.DataFrame:
    """
    Read ledger predictions and join setpoint vectors from ledger run metadata.

    Returns a polars DataFrame with an added ``obj__diag__setpoint`` column.
    """

    pred_dir = outputs_dir / "ledger" / "predictions"
    runs_path = outputs_dir / "ledger" / "runs.parquet"
    runs_df = read_runs(runs_path)

    want = set(map(str, base_columns)) | {"run_id"}
    df = read_predictions(
        pred_dir,
        columns=sorted(want),
        round_selector=round_selector,
        run_id=run_id,
        runs_df=runs_df,
        row_filters=row_filters,
        require_run_id=require_run_id,
    )
    if df.is_empty():
        raise OpalError(
            "outputs/ledger/predictions had zero rows after projection.",
            ExitCodes.BAD_ARGS,
        )
    if "objective__params" not in runs_df.columns:
        raise OpalError(
            "outputs/ledger/runs.parquet is missing objective__params (cannot resolve setpoints).",
            ExitCodes.BAD_ARGS,
        )

    meta = runs_df.select(["run_id", "objective__params"]).with_columns(
        pl.col("objective__params")
        .map_elements(_extract_setpoint, return_dtype=pl.List(pl.Float64))
        .alias("obj__diag__setpoint")
    )
    out = df.join(meta.select(["run_id", "obj__diag__setpoint"]), on="run_id", how="left")
    _validate_setpoint_join(out)
    return out


def _extract_setpoint(obj: Any) -> list[float] | None:
    vec = (obj or {}).get("setpoint_vector")
    if vec is None:
        return None
    try:
        vals = [float(x) for x in vec]
    except Exception:
        return None
    if not vals:
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return vals


def _validate_setpoint_join(out: pl.DataFrame) -> None:
    if out["obj__diag__setpoint"].drop_nulls().is_empty():
        raise OpalError(
            "Could not resolve setpoint for any rows: run_meta lacks objective__params.setpoint_vector.",
            ExitCodes.BAD_ARGS,
        )
    missing = (
        out.filter(pl.col("obj__diag__setpoint").is_null()).select(pl.col("run_id").unique()).to_series().to_list()
    )
    if missing:
        raise OpalError(
            f"Missing objective__params.setpoint_vector for run_id(s): {sorted(missing)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    setpoints = [tuple(sp) for sp in out["obj__diag__setpoint"].drop_nulls().to_list()]
    unique = {sp for sp in setpoints}
    if len(unique) > 1:
        raise OpalError(
            f"Multiple setpoint vectors found for selected rows: {sorted(unique)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
