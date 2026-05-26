"""Manifest-ledger prediction reads with explicit round and run contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

from ...core.utils import ExitCodes, OpalError
from .io import scan_predictions
from .rounds import RoundSelector, latest_round


def read_predictions(
    pred_dir: Path,
    *,
    columns: Sequence[str] | None = None,
    round_selector: RoundSelector | None = None,
    run_id: str | None = None,
    runs_df: pl.DataFrame | None = None,
    row_filters: Sequence[Mapping[str, Any]] | None = None,
    allow_missing: bool = False,
    require_run_id: bool = True,
) -> pl.DataFrame:
    lf = scan_predictions(pred_dir)
    if run_id is not None:
        if runs_df is None or runs_df.is_empty():
            raise OpalError(
                "run_id was provided but outputs/ledger/runs.parquet is missing or empty. "
                "Pass runs_df or call CampaignAnalysis.read_predictions so OPAL can resolve run_id -> as_of_round. "
                "Use `opal runs list` or `opal status --with-ledger` to find valid run_id values.",
                ExitCodes.BAD_ARGS,
            )
        run_round = _resolve_round_for_run_id(str(run_id), runs_df)
        if round_selector in (None, "unspecified", "latest"):
            round_selector = [run_round]
        elif round_selector != "all":
            selected = _selected_rounds(round_selector, runs_df)
            if run_round not in selected:
                raise OpalError(
                    f"run_id {run_id!r} belongs to as_of_round={run_round}, "
                    f"but round_selector={round_selector!r} excludes it.",
                    ExitCodes.BAD_ARGS,
                )
    _require_run_id_if_ambiguous(
        runs_df=runs_df,
        round_selector=round_selector,
        run_id=run_id,
        require_run_id=require_run_id,
    )
    want = _select_existing_columns(lf, columns, allow_missing=allow_missing)
    lf = _apply_round_filter(lf, round_selector=round_selector, runs_df=runs_df)
    if run_id is not None:
        lf = lf.filter(pl.col("run_id") == str(run_id))
    lf = _apply_row_filters(lf, row_filters)
    if want is not None:
        lf = lf.select(want)
    return lf.collect()


def _apply_row_filters(lf: pl.LazyFrame, row_filters: Sequence[Mapping[str, Any]] | None) -> pl.LazyFrame:
    if not row_filters:
        return lf
    try:
        schema_cols = set(lf.collect_schema().keys())
    except Exception:
        schema_cols = set(lf.schema.keys())
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


def _select_existing_columns(
    lf: pl.LazyFrame,
    columns: Sequence[str] | None,
    *,
    allow_missing: bool,
) -> list[str] | None:
    if not columns:
        return None
    want = [c for c in columns if c]
    try:
        schema_cols = set(lf.collect_schema().keys())
    except Exception:
        schema_cols = set(lf.schema.keys())
    missing = [c for c in want if c not in schema_cols]
    if missing and not allow_missing:
        raise OpalError(
            f"outputs/ledger/predictions is missing columns: {sorted(missing)}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if allow_missing:
        want = [c for c in want if c in schema_cols]
    return want


def _apply_round_filter(
    lf: pl.LazyFrame,
    *,
    round_selector: RoundSelector | None,
    runs_df: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    if round_selector in (None, "unspecified", "latest"):
        if runs_df is not None and not runs_df.is_empty():
            return lf.filter(pl.col("as_of_round") == latest_round(runs_df))
        latest = lf.select(pl.col("as_of_round").max()).collect()
        if latest.is_empty():
            return lf
        latest_val = int(latest["as_of_round"][0])
        return lf.filter(pl.col("as_of_round") == latest_val)
    if round_selector == "all":
        return lf
    if isinstance(round_selector, list):
        return lf.filter(pl.col("as_of_round").is_in([int(x) for x in round_selector]))
    return lf.filter(pl.col("as_of_round") == int(round_selector))


def _selected_rounds(round_selector: RoundSelector | None, runs_df: pl.DataFrame | None) -> list[int]:
    if runs_df is None or runs_df.is_empty():
        return []
    if round_selector in (None, "unspecified", "latest"):
        return [latest_round(runs_df)]
    if round_selector == "all":
        return sorted({int(x) for x in runs_df["as_of_round"].to_list()})
    if isinstance(round_selector, list):
        return [int(x) for x in round_selector]
    return [int(round_selector)]


def _resolve_round_for_run_id(run_id: str, runs_df: pl.DataFrame) -> int:
    if runs_df.is_empty():
        raise OpalError(
            "outputs/ledger/runs.parquet is empty; cannot resolve run_id.",
            ExitCodes.BAD_ARGS,
        )
    if "run_id" not in runs_df.columns or "as_of_round" not in runs_df.columns:
        raise OpalError(
            "outputs/ledger/runs.parquet missing required columns (run_id, as_of_round).",
            ExitCodes.BAD_ARGS,
        )
    df = runs_df.filter(pl.col("run_id") == str(run_id)).select(pl.col("as_of_round").drop_nulls().unique())
    if df.is_empty():
        raise OpalError(
            f"run_id {run_id!r} not found in outputs/ledger/runs.parquet.",
            ExitCodes.BAD_ARGS,
        )
    rounds = sorted({int(x) for x in df.to_series().to_list()})
    if len(rounds) > 1:
        raise OpalError(
            f"run_id {run_id!r} appears in multiple rounds {rounds}; outputs/ledger/runs.parquet is inconsistent.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return rounds[0]


def _require_run_id_if_ambiguous(
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
    rounds = _selected_rounds(round_selector, runs_df)
    if not rounds:
        raise OpalError(
            "Run ID is required to disambiguate ledger predictions, but no runs were found for selection.",
            ExitCodes.BAD_ARGS,
        )
    df = runs_df.filter(pl.col("as_of_round").is_in(rounds))
    if df.is_empty():
        raise OpalError(
            f"No runs found in outputs/ledger/runs.parquet for selected rounds {rounds}.",
            ExitCodes.BAD_ARGS,
        )
    counts = df.group_by("as_of_round").agg(pl.col("run_id").n_unique().alias("n_runs"))
    multi = counts.filter(pl.col("n_runs") > 1).select(pl.col("as_of_round")).to_series().to_list()
    if multi:
        raise OpalError(
            "Multiple run_id values found for round(s) "
            f"{sorted(int(x) for x in multi)}. Specify run_id to avoid mixing reruns "
            "(ledger is append-only; rerunning a round creates a new run_id). "
            "Use `opal runs list` or `opal status --with-ledger` to find valid run_id values.",
            ExitCodes.BAD_ARGS,
        )
