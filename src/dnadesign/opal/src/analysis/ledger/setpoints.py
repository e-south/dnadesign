"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/ledger/setpoints.py

Prediction reads joined with objective setpoint metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

from ...core.utils import ExitCodes, OpalError
from .io import read_runs
from .predictions import read_selection_view_predictions
from .rounds import RoundSelector


def load_predictions_with_setpoint(
    outputs_dir: Path,
    base_columns: Iterable[str],
    *,
    selection_view_id: str,
    round_selector: RoundSelector | None = None,
    run_id: str | None = None,
    row_filters: Sequence[Mapping[str, Any]] | None = None,
    require_run_id: bool = True,
) -> pl.DataFrame:
    """
    Read ledger predictions and join one selection view's setpoint metadata.

    Returns a polars DataFrame with an added ``obj__diag__setpoint`` column.
    """

    pred_dir = outputs_dir / "ledger" / "predictions"
    runs_path = outputs_dir / "ledger" / "runs.parquet"
    runs_df = read_runs(runs_path)

    want = set(map(str, base_columns)) | {"run_id"}
    df = read_selection_view_predictions(
        pred_dir,
        selection_view_id=selection_view_id,
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
    defs_column = "selection_views__defs_json"
    if defs_column not in runs_df.columns:
        raise OpalError(
            f"outputs/ledger/runs.parquet is missing {defs_column} (cannot resolve selection views).",
            ExitCodes.BAD_ARGS,
        )

    meta = runs_df.select(["run_id", defs_column]).with_columns(
        pl.col(defs_column)
        .map_elements(
            lambda raw: _extract_view_setpoint(raw, selection_view_id=selection_view_id),
            return_dtype=pl.List(pl.Float64),
        )
        .alias("obj__diag__setpoint")
    )
    out = df.join(meta.select(["run_id", "obj__diag__setpoint"]), on="run_id", how="left")
    _validate_setpoint_join(out, selection_view_id=selection_view_id)
    return out


def _extract_view_setpoint(raw: Any, *, selection_view_id: str) -> list[float] | None:
    try:
        definitions = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(definitions, list):
        return None
    matching = [
        definition
        for definition in definitions
        if isinstance(definition, Mapping) and definition.get("selection_view_id") == selection_view_id
    ]
    if len(matching) != 1:
        return None
    params = matching[0].get("objective_params")
    if not isinstance(params, Mapping):
        return None
    vec = params.get("setpoint_vector")
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


def _validate_setpoint_join(out: pl.DataFrame, *, selection_view_id: str) -> None:
    if out["obj__diag__setpoint"].drop_nulls().is_empty():
        raise OpalError(
            f"Could not resolve selection view {selection_view_id!r} with objective_params.setpoint_vector.",
            ExitCodes.BAD_ARGS,
        )
    missing = (
        out.filter(pl.col("obj__diag__setpoint").is_null()).select(pl.col("run_id").unique()).to_series().to_list()
    )
    if missing:
        raise OpalError(
            f"Selection view {selection_view_id!r} is missing objective_params.setpoint_vector "
            f"for run_id(s): {sorted(missing)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    setpoints = [tuple(sp) for sp in out["obj__diag__setpoint"].drop_nulls().to_list()]
    unique = {sp for sp in setpoints}
    if len(unique) > 1:
        raise OpalError(
            f"Multiple setpoint vectors found for selected rows: {sorted(unique)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
