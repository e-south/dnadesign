"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/ledger/io.py

Ledger sink validation and read helpers for OPAL analysis consumers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from ...core.utils import ExitCodes, OpalError


def require_columns(df: pl.DataFrame, columns: Iterable[str], *, ctx: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise OpalError(
            f"{ctx}: missing required columns {sorted(missing)}",
            ExitCodes.CONTRACT_VIOLATION,
        )


def scan_predictions(pred_dir: Path) -> pl.LazyFrame:
    ensure_predictions_dir(pred_dir)
    files = _pred_parquet_paths(pred_dir)
    return pl.scan_parquet([str(path) for path in files])


def scan_runs(runs_path: Path) -> pl.LazyFrame:
    ensure_runs_path(runs_path)
    return pl.scan_parquet(str(runs_path))


def scan_labels(labels_path: Path) -> pl.LazyFrame:
    ensure_labels_path(labels_path)
    return pl.scan_parquet(str(labels_path))


def read_runs(runs_path: Path) -> pl.DataFrame:
    return scan_runs(runs_path).collect()


def read_labels(labels_path: Path) -> pl.DataFrame:
    return scan_labels(labels_path).collect()


def ensure_predictions_dir(pred_dir: Path) -> None:
    if not pred_dir.exists():
        raise OpalError(
            f"Missing predictions sink: {pred_dir}. Run `opal run -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )
    if not _pred_parquet_paths(pred_dir):
        raise OpalError(
            f"Predictions sink is empty: {pred_dir}. Run `opal run -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )


def ensure_runs_path(runs_path: Path) -> None:
    if not runs_path.exists():
        raise OpalError(
            f"Missing runs sink: {runs_path}. Run `opal run -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )


def ensure_labels_path(labels_path: Path) -> None:
    if not labels_path.exists():
        raise OpalError(
            f"Missing labels sink: {labels_path}. Run `opal ingest-y -c <campaign.yaml> --round <k>` first.",
            ExitCodes.BAD_ARGS,
        )


def _pred_parquet_paths(pred_dir: Path) -> list[Path]:
    return sorted({p for p in pred_dir.rglob("*.parquet") if p.is_file()})
