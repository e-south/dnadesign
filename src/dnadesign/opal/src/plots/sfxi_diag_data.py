"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/sfxi_diag_data.py

Shared helpers for SFXI diagnostic plot plugins.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

from ..analysis.facade import latest_round
from ..core.utils import ExitCodes, OpalError
from ..objectives import sfxi_math


def resolve_single_round(
    runs_df: pl.DataFrame,
    *,
    round_selector: str | int | list[int] | None,
) -> int:
    if runs_df.is_empty():
        raise OpalError("No runs available. Run `opal run ...` first.", ExitCodes.BAD_ARGS)
    if round_selector in (None, "unspecified", "latest"):
        return latest_round(runs_df)
    if round_selector == "all":
        raise OpalError("Select a single round for this plot (e.g., --round latest or --round 3).", ExitCodes.BAD_ARGS)
    if isinstance(round_selector, list):
        if len(round_selector) != 1:
            raise OpalError("Select a single round for this plot.", ExitCodes.BAD_ARGS)
        return int(round_selector[0])
    return int(round_selector)


def resolve_run_id(
    runs_df: pl.DataFrame,
    *,
    round_k: int,
    run_id: str | None,
) -> str | None:
    if run_id is not None:
        return str(run_id)
    if "run_id" not in runs_df.columns:
        return None
    run_ids = (
        runs_df.filter(pl.col("as_of_round") == int(round_k))
        .select(pl.col("run_id").drop_nulls().unique())
        .to_series()
        .to_list()
    )
    run_ids = sorted({str(v) for v in run_ids if v is not None})
    if len(run_ids) > 1:
        raise OpalError(
            f"Multiple run_ids exist for round {round_k}; pass --run-id to disambiguate.",
            ExitCodes.BAD_ARGS,
        )
    return run_ids[0] if run_ids else None


def labels_asof_round(labels_df: pl.DataFrame, *, round_k: int) -> pl.DataFrame:
    if labels_df.is_empty():
        return labels_df
    if "observed_round" not in labels_df.columns:
        raise OpalError("labels.parquet missing observed_round.", ExitCodes.CONTRACT_VIOLATION)
    return labels_df.filter(pl.col("observed_round") <= int(round_k))


def parse_setpoint_from_runs(runs_df: pl.DataFrame) -> Sequence[float]:
    if "objective__params" not in runs_df.columns:
        raise OpalError("runs.parquet missing objective__params (setpoint unavailable).", ExitCodes.BAD_ARGS)

    def _extract(obj):
        vec = (obj or {}).get("setpoint_vector")
        if vec is None:
            return None
        try:
            return [float(x) for x in vec]
        except Exception:
            return None

    vals = runs_df.select(
        pl.col("objective__params").map_elements(_extract, return_dtype=pl.List(pl.Float64)).alias("setpoint")
    )["setpoint"].drop_nulls()
    if vals.is_empty():
        raise OpalError("No setpoint_vector found in runs.parquet.", ExitCodes.BAD_ARGS)
    unique = {tuple(v) for v in vals.to_list()}
    if len(unique) > 1:
        raise OpalError(f"Multiple setpoints found: {sorted(unique)}.", ExitCodes.CONTRACT_VIOLATION)
    return sfxi_math.parse_setpoint_vector(list(unique)[0])
