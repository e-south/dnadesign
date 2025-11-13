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


def _index_path(root: Path | None) -> Path:
    # Always resolve via runs_root() to keep a single, consistent home.
    return runs_root(root) / "index.parquet"


def add_or_update_index(row: dict, root: Path | None = None) -> None:
    """
    Append a single run row into results/index.parquet, evolving schema as needed.
    """
    idx_path = _index_path(root)
    df = pd.read_parquet(idx_path) if idx_path.exists() else pd.DataFrame()

    # Column superset = existing ∪ new row’s keys
    cols = sorted(set(df.columns).union(row.keys()))
    new_row_df = pd.DataFrame([{c: row.get(c, pd.NA) for c in cols}], columns=cols)

    if df.empty:
        # First write: avoid appending into an empty frame (triggers pandas’ concat deprecation)
        out = new_row_df
    else:
        # Reindex to superset (introduces any new columns); no fill_value to avoid all‑NA blocks
        base = df.reindex(columns=cols, copy=True)
        out = pd.concat([base, new_row_df], ignore_index=True, sort=False)

    # Persist deterministically (tmp → replace)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = idx_path.with_suffix(idx_path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(idx_path)


def list_runs(filters: dict | None = None, root: Path | None = None) -> pd.DataFrame:
    idx_path = _index_path(root)
    if not idx_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(idx_path)
    if filters:
        for k, v in filters.items():
            if v is None or k not in df.columns:
                continue
            df = df[df[k] == v]
    if "created_utc" in df.columns:
        return df.sort_values("created_utc", ascending=False, ignore_index=True)
    return df.reset_index(drop=True)
