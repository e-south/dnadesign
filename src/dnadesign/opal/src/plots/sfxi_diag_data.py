"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/sfxi_diag_data.py

Shared helpers for SFXI diagnostic plot plugins.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import polars as pl

from ..analysis.ledger import latest_round
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


def labels_current_round(labels_df: pl.DataFrame, *, round_k: int) -> pl.DataFrame:
    if labels_df.is_empty():
        return labels_df
    if "observed_round" not in labels_df.columns:
        raise OpalError("labels.parquet missing observed_round.", ExitCodes.CONTRACT_VIOLATION)
    return labels_df.filter(pl.col("observed_round") == int(round_k))


def _objective_params(runs_df: pl.DataFrame, *, selection_view_id: str) -> list[dict]:
    field = "objective__defs_json"
    if field not in runs_df.columns:
        raise OpalError(f"runs.parquet missing {field}.", ExitCodes.BAD_ARGS)
    params: list[dict] = []
    for raw in runs_df[field].to_list():
        try:
            definitions = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OpalError(f"runs.parquet {field} is invalid JSON: {exc}", ExitCodes.CONTRACT_VIOLATION) from exc
        matches = [
            definition
            for definition in definitions
            if isinstance(definition, dict) and definition.get("selection_view_id") == selection_view_id
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("params"), dict):
            raise OpalError(
                f"Selection view {selection_view_id!r} is not defined exactly once with objective params.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        params.append(dict(matches[0]["params"]))
    return params


def parse_setpoint_from_runs(runs_df: pl.DataFrame, *, selection_view_id: str) -> Sequence[float]:
    values = []
    for params in _objective_params(runs_df, selection_view_id=selection_view_id):
        try:
            values.append(tuple(float(value) for value in params["setpoint_vector"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        raise OpalError("No setpoint_vector found in runs.parquet.", ExitCodes.BAD_ARGS)
    unique = set(values)
    if len(unique) > 1:
        raise OpalError(f"Multiple setpoints found: {sorted(unique)}.", ExitCodes.CONTRACT_VIOLATION)
    setpoint = list(unique)[0]
    return sfxi_math.parse_setpoint_vector({"setpoint_vector": list(setpoint)})


def parse_exponents_from_runs(runs_df: pl.DataFrame, *, selection_view_id: str) -> tuple[float, float]:
    values = []
    for params in _objective_params(runs_df, selection_view_id=selection_view_id):
        try:
            values.append((float(params["logic_exponent_beta"]), float(params["intensity_exponent_gamma"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        raise OpalError(
            "No logic_exponent_beta/intensity_exponent_gamma found in runs.parquet.",
            ExitCodes.BAD_ARGS,
        )
    unique = set(values)
    if len(unique) > 1:
        raise OpalError(
            f"Multiple logic_exponent_beta/intensity_exponent_gamma pairs found: {sorted(unique)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    beta, gamma = list(unique)[0]
    if not (np.isfinite(beta) and np.isfinite(gamma)):
        raise OpalError(
            "logic_exponent_beta/intensity_exponent_gamma must be finite.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return float(beta), float(gamma)


def parse_delta_from_runs(runs_df: pl.DataFrame, *, selection_view_id: str) -> float:
    values = []
    for params in _objective_params(runs_df, selection_view_id=selection_view_id):
        try:
            values.append(float(params["intensity_log2_offset_delta"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        raise OpalError(
            "No intensity_log2_offset_delta found in runs.parquet.",
            ExitCodes.BAD_ARGS,
        )
    unique = set(values)
    if len(unique) > 1:
        raise OpalError(
            f"Multiple intensity_log2_offset_delta values found: {sorted(unique)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    delta = list(unique)[0]
    if not np.isfinite(delta) or delta < 0.0:
        raise OpalError(
            "intensity_log2_offset_delta must be finite and >= 0.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return float(delta)
