"""Ledger and optional-table helpers for campaign progress notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import polars as pl

from .models import OptionalTableRead


def build_ledger_status_table(workdir: Path | None) -> pl.DataFrame:
    if workdir is None:
        return pl.DataFrame(
            {
                "artifact": ["state", "labels", "runs", "predictions"],
                "status": ["missing workdir"] * 4,
                "rows": [None] * 4,
                "path": [""] * 4,
            }
        )

    state_path = workdir / "state.json"
    labels_path = workdir / "outputs" / "ledger" / "labels.parquet"
    runs_path = workdir / "outputs" / "ledger" / "runs.parquet"
    predictions_dir = workdir / "outputs" / "ledger" / "predictions"
    prediction_parts = sorted(predictions_dir.glob("*.parquet")) if predictions_dir.exists() else []
    rows = [
        {
            "artifact": "state",
            "status": "present" if state_path.exists() else "missing",
            "rows": None,
            "path": str(state_path),
        },
        {
            "artifact": "labels",
            "status": "present" if labels_path.exists() else "missing",
            "rows": _parquet_row_count(labels_path),
            "path": str(labels_path),
        },
        {
            "artifact": "runs",
            "status": "present" if runs_path.exists() else "missing",
            "rows": _parquet_row_count(runs_path),
            "path": str(runs_path),
        },
        {
            "artifact": "predictions",
            "status": "present" if prediction_parts else "missing",
            "rows": len(prediction_parts) if prediction_parts else None,
            "path": str(predictions_dir),
        },
    ]
    return pl.DataFrame(rows)


def read_optional_table(
    name: str,
    path: Path | str | None,
    loader: Callable[[], pl.DataFrame],
) -> OptionalTableRead:
    table_path = Path(path) if path is not None else None
    try:
        df = loader()
    except Exception as exc:
        return OptionalTableRead(
            name=str(name),
            path=table_path,
            df=pl.DataFrame(),
            status="unavailable",
            message=str(exc),
        )
    status = "available"
    message = "available"
    if df.is_empty():
        status = "empty"
        message = "table exists but has zero rows"
    return OptionalTableRead(
        name=str(name),
        path=table_path,
        df=df,
        status=status,
        message=message,
    )


def unavailable_table(name: str, path: Path | str | None, message: str) -> OptionalTableRead:
    table_path = Path(path) if path is not None else None
    return OptionalTableRead(
        name=str(name),
        path=table_path,
        df=pl.DataFrame(),
        status="unavailable",
        message=str(message),
    )


def table_status_lines(table: OptionalTableRead) -> list[str]:
    path_text = str(table.path) if table.path is not None else ""
    return [
        f"- {table.name}: **{table.status}**",
        f"- Rows: `{table.df.height}`",
        f"- Path: `{path_text}`",
        f"- Detail: {table.message}",
    ]


def _parquet_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(pl.scan_parquet(str(path)).select(pl.len()).collect().item())
    except Exception:
        return None
