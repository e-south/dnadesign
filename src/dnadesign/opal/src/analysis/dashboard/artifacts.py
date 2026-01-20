"""Artifact resolution helpers for dashboard notebooks."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from .util import deep_as_py


def _coerce_artifacts_map(raw: Any) -> dict[str, str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    raw = deep_as_py(raw)
    if not isinstance(raw, dict):
        return None
    out: dict[str, str] = {}
    for key, val in raw.items():
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            out[str(key)] = str(val[1])
        elif isinstance(val, str):
            out[str(key)] = val
    return out or None


def resolve_run_artifacts(
    ledger_runs_df: pl.DataFrame | None,
    *,
    run_id: str | None,
) -> tuple[dict[str, str] | None, str | None]:
    if ledger_runs_df is None or ledger_runs_df.is_empty():
        return None, "Ledger runs unavailable."
    if not run_id:
        return None, "run_id is required to resolve artifacts."
    if "run_id" not in ledger_runs_df.columns or "artifacts" not in ledger_runs_df.columns:
        return None, "Ledger runs missing required columns (run_id, artifacts)."
    rows = ledger_runs_df.filter(pl.col("run_id") == str(run_id)).select(pl.col("artifacts")).to_series().to_list()
    if not rows:
        return None, f"run_id {run_id} not found in ledger runs."
    artifacts = _coerce_artifacts_map(rows[0])
    if not artifacts:
        return None, "Artifacts field missing or unparseable."
    return artifacts, None
