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
    idx_path = _index_path(root)
    # Read deterministically. Exclude empty frames from concat to avoid
    # pandas' deprecation around empty/all‑NA entries changing dtype behavior.
    try:
        df = pd.read_parquet(idx_path)
    except Exception:
        df = pd.DataFrame()
    parts = [df] if not df.empty else []
    parts.append(pd.DataFrame([row]))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(idx_path, index=False)


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
    return df.sort_values("created_utc", ascending=False, ignore_index=True)
