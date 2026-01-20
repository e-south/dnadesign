"""Mismatch debugger helpers for selection artifacts vs ledger predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .context import DashboardContext


@dataclass(frozen=True)
class MismatchResult:
    message: str
    table: pl.DataFrame


def compare_selection_to_ledger(
    *,
    selection_path: str,
    context: DashboardContext | None = None,
    ledger_preds_df: pl.DataFrame | None = None,
    selected_run_id: str | None,
    eps: float = 1.0e-6,
) -> MismatchResult:
    if ledger_preds_df is None and context is not None:
        ledger_preds_df = context.ledger_preds_df
    if not selection_path:
        return MismatchResult("No selection CSV path provided.", pl.DataFrame())
    if ledger_preds_df is None or ledger_preds_df.is_empty():
        return MismatchResult("Ledger predictions unavailable; cannot compare.", pl.DataFrame())

    path = Path(selection_path)
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            df_csv = pl.read_parquet(path)
        else:
            df_csv = pl.read_csv(path)
    except Exception as exc:
        return MismatchResult(f"Failed to read selection file: {exc}", pl.DataFrame())

    if "id" not in df_csv.columns:
        return MismatchResult("CSV missing `id` column.", pl.DataFrame())

    csv_score_col = None
    if "pred__y_obj_scalar" in df_csv.columns:
        csv_score_col = "pred__y_obj_scalar"
    elif "selection_score" in df_csv.columns:
        csv_score_col = "selection_score"
    if csv_score_col is None:
        return MismatchResult(
            "CSV missing score column (pred__y_obj_scalar or selection_score).",
            pl.DataFrame(),
        )

    df_csv = df_csv.with_columns(
        pl.col("id").cast(pl.Utf8).alias("id"),
        pl.col(csv_score_col).cast(pl.Float64),
    )
    csv_run_ids: list[str] = []
    if "run_id" in df_csv.columns:
        csv_run_ids = df_csv.select(pl.col("run_id").drop_nulls().unique()).to_series().to_list()
        csv_run_ids = [str(x) for x in csv_run_ids]

    if "pred__y_obj_scalar" not in ledger_preds_df.columns:
        return MismatchResult("Ledger predictions missing pred__y_obj_scalar.", pl.DataFrame())

    df_ledger = ledger_preds_df.select(
        [
            pl.col("id").cast(pl.Utf8).alias("id"),
            pl.col("pred__y_obj_scalar").cast(pl.Float64).alias("ledger_score"),
        ]
    )
    df_join = df_csv.join(df_ledger, on="id", how="inner")
    if df_join.is_empty():
        return MismatchResult("No overlapping IDs between CSV and ledger predictions.", pl.DataFrame())

    df_join = df_join.with_columns(
        (pl.col(csv_score_col) - pl.col("ledger_score")).abs().alias("abs_diff"),
        (pl.col(csv_score_col) - pl.col("ledger_score")).alias("diff"),
    )
    mismatch_count = df_join.filter(pl.col("abs_diff") > eps).height
    lines = [
        f"Compared `{df_join.height}` rows (CSV vs ledger).",
        f"Mismatches (abs diff > {eps:g}): `{mismatch_count}`",
        f"Max abs diff: `{df_join.select(pl.col('abs_diff').max()).item():.6g}`",
    ]
    if csv_run_ids:
        lines.append(f"CSV run_id(s): `{csv_run_ids}`")
        if selected_run_id and str(selected_run_id) not in csv_run_ids:
            lines.append(f"Warning: CSV run_id does not match selected ledger run_id `{selected_run_id}`.")
    else:
        lines.append("Warning: CSV missing run_id; comparison may be run-agnostic.")
    df_top = df_join.sort("abs_diff", descending=True).head(10)
    return MismatchResult("\n".join(lines), df_top)
