"""Column, row, round, and run scoping for prediction-ledger reads."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import polars as pl

from ...core.utils import ExitCodes, OpalError
from .rounds import RoundSelector, latest_round


def apply_row_filters(
    lf: pl.LazyFrame,
    row_filters: Sequence[Mapping[str, Any]] | None,
) -> pl.LazyFrame:
    if not row_filters:
        return lf
    schema_cols = set(lf.collect_schema().keys())
    out = lf
    for raw in row_filters:
        column = str(raw.get("column") or "").strip()
        op = str(raw.get("op") or "eq").strip().lower()
        if not column:
            raise OpalError("prediction row filter requires a non-empty column.", ExitCodes.BAD_ARGS)
        if column not in schema_cols:
            raise OpalError(
                f"outputs/ledger/predictions is missing filter column: {column}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        value = raw.get("value")
        expr = pl.col(column)
        if op == "eq":
            out = out.filter(expr == value)
        elif op == "lte":
            out = out.filter(expr <= value)
        elif op == "lt":
            out = out.filter(expr < value)
        elif op == "gte":
            out = out.filter(expr >= value)
        elif op == "gt":
            out = out.filter(expr > value)
        elif op == "is_in":
            if not isinstance(value, list):
                raise OpalError("prediction row filter op='is_in' requires list value.", ExitCodes.BAD_ARGS)
            out = out.filter(expr.is_in(value))
        else:
            raise OpalError(f"unsupported prediction row filter op: {op!r}", ExitCodes.BAD_ARGS)
    return out


def select_existing_columns(
    lf: pl.LazyFrame,
    columns: Sequence[str] | None,
    *,
    allow_missing: bool,
) -> list[str] | None:
    if not columns:
        return None
    want = [column for column in columns if column]
    schema_cols = set(lf.collect_schema().keys())
    missing = [column for column in want if column not in schema_cols]
    if missing and not allow_missing:
        raise OpalError(
            f"outputs/ledger/predictions is missing columns: {sorted(missing)}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return [column for column in want if column in schema_cols] if allow_missing else want


def apply_round_filter(
    lf: pl.LazyFrame,
    *,
    round_selector: RoundSelector | None,
    runs_df: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    if round_selector in (None, "unspecified", "latest"):
        if runs_df is not None and not runs_df.is_empty():
            return lf.filter(pl.col("as_of_round") == latest_round(runs_df))
        latest = lf.select(pl.col("as_of_round").max()).collect()
        return lf if latest.is_empty() else lf.filter(pl.col("as_of_round") == int(latest["as_of_round"][0]))
    if round_selector == "all":
        return lf
    if isinstance(round_selector, list):
        return lf.filter(pl.col("as_of_round").is_in([int(value) for value in round_selector]))
    return lf.filter(pl.col("as_of_round") == int(round_selector))


def selected_rounds(round_selector: RoundSelector | None, runs_df: pl.DataFrame | None) -> list[int]:
    if runs_df is None or runs_df.is_empty():
        return []
    if round_selector in (None, "unspecified", "latest"):
        return [latest_round(runs_df)]
    if round_selector == "all":
        return sorted({int(value) for value in runs_df["as_of_round"].to_list()})
    if isinstance(round_selector, list):
        return [int(value) for value in round_selector]
    return [int(round_selector)]


def resolve_round_for_run_id(run_id: str, runs_df: pl.DataFrame) -> int:
    if runs_df.is_empty():
        raise OpalError("outputs/ledger/runs.parquet is empty; cannot resolve run_id.", ExitCodes.BAD_ARGS)
    if "run_id" not in runs_df.columns or "as_of_round" not in runs_df.columns:
        raise OpalError(
            "outputs/ledger/runs.parquet missing required columns (run_id, as_of_round).",
            ExitCodes.BAD_ARGS,
        )
    frame = runs_df.filter(pl.col("run_id") == str(run_id)).select(pl.col("as_of_round").drop_nulls().unique())
    if frame.is_empty():
        raise OpalError(f"run_id {run_id!r} not found in outputs/ledger/runs.parquet.", ExitCodes.BAD_ARGS)
    rounds = sorted({int(value) for value in frame.to_series().to_list()})
    if len(rounds) > 1:
        raise OpalError(
            f"run_id {run_id!r} appears in multiple rounds {rounds}; outputs/ledger/runs.parquet is inconsistent.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return rounds[0]


def require_run_id_if_ambiguous(
    *,
    runs_df: pl.DataFrame | None,
    round_selector: RoundSelector | None,
    run_id: str | None,
    require_run_id: bool,
) -> None:
    if not require_run_id or run_id is not None:
        return
    if runs_df is None or runs_df.is_empty():
        raise OpalError(
            "Run ID is required, but outputs/ledger/runs.parquet is missing or empty. "
            "Provide run_id explicitly (e.g., --run-id) or generate ledger runs first.",
            ExitCodes.BAD_ARGS,
        )
    rounds = selected_rounds(round_selector, runs_df)
    if not rounds:
        raise OpalError(
            "Run ID is required to disambiguate ledger predictions, but no runs were found for selection.",
            ExitCodes.BAD_ARGS,
        )
    frame = runs_df.filter(pl.col("as_of_round").is_in(rounds))
    if frame.is_empty():
        raise OpalError(
            f"No runs found in outputs/ledger/runs.parquet for selected rounds {rounds}.", ExitCodes.BAD_ARGS
        )
    counts = frame.group_by("as_of_round").agg(pl.col("run_id").n_unique().alias("n_runs"))
    multi = counts.filter(pl.col("n_runs") > 1).select(pl.col("as_of_round")).to_series().to_list()
    if multi:
        raise OpalError(
            "Multiple run_id values found for round(s) "
            f"{sorted(int(value) for value in multi)}. Specify run_id to avoid mixing reruns "
            "(ledger is append-only; rerunning a round creates a new run_id). "
            "Use `opal runs list` or `opal status --with-ledger` to find valid run_id values.",
            ExitCodes.BAD_ARGS,
        )


__all__ = [
    "apply_round_filter",
    "apply_row_filters",
    "require_run_id_if_ambiguous",
    "resolve_round_for_run_id",
    "select_existing_columns",
]
