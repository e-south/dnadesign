"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/runs/index.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .store import runs_root


def _index_path(root: Path) -> Path:
    return runs_root(root) / "index.parquet"


def add_or_update_index(row: dict, root: Path | None = None) -> None:
    idx_path = _index_path(root or Path.cwd())
    try:
        df = pd.read_parquet(idx_path)
    except Exception:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_parquet(idx_path, index=False)


def list_runs(filters: dict | None = None, root: Path | None = None) -> pd.DataFrame:
    idx_path = _index_path(root or Path.cwd())
    if not idx_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(idx_path)
    if filters:
        for k, v in filters.items():
            if v is None or k not in df.columns:
                continue
            df = df[df[k] == v]
    return df.sort_values("created_utc", ascending=False, ignore_index=True)
