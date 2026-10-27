"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/runs/index_store.py

Append-friendly storage helpers for the cluster run index.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
import pandas as pd

from ..layout import explicit_results_root
from .contracts import RunIndexEntry
from .store import runs_root

INDEX_DELTA_DIRNAME = "index.delta"
INDEX_MAX_DELTA_FILES = 32


def index_snapshot_path(root: Path | None, *, materialize: bool) -> Path:
    resolved_root = runs_root(root) if materialize else explicit_results_root(root)
    return resolved_root / "index.parquet"


def index_delta_dir(root: Path | None, *, materialize: bool) -> Path:
    resolved_root = runs_root(root) if materialize else explicit_results_root(root)
    return resolved_root / INDEX_DELTA_DIRNAME


def delta_paths(delta_dir: Path) -> list[Path]:
    if not delta_dir.exists():
        return []
    return sorted(path for path in delta_dir.glob("*.parquet") if path.is_file())


def ordered_columns(columns: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([*RunIndexEntry.columns(), *(str(column) for column in columns if column)]))


def write_snapshot(df: pd.DataFrame, idx_path: Path) -> None:
    ordered = ordered_columns(df.columns)
    out = df.reindex(columns=ordered)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = idx_path.with_suffix(idx_path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(idx_path)


def collapse_index_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "_cluster_order" not in combined.columns:
        combined["_cluster_order"] = np.arange(len(combined), dtype=np.int64)
    if "run_slug" in combined.columns:
        combined = (
            combined.sort_values("_cluster_order", ascending=True, ignore_index=True)
            .drop_duplicates(subset=["run_slug"], keep="last")
            .reset_index(drop=True)
        )
    return combined


def append_index_delta(row: RunIndexEntry | dict[str, object], *, root: Path | None) -> None:
    idx_path = index_snapshot_path(root, materialize=True)
    delta_root = index_delta_dir(root, materialize=True)
    delta_root.mkdir(parents=True, exist_ok=True)
    payload = row.payload() if isinstance(row, RunIndexEntry) else dict(row)
    existing_columns: list[str] = []
    if idx_path.exists():
        try:
            import pyarrow.dataset as ds

            existing_columns.extend(ds.dataset(str(idx_path), format="parquet").schema.names)
        except Exception:
            pass
    cols = ordered_columns([*existing_columns, *payload.keys()])
    new_row_df = pd.DataFrame([{c: payload.get(c, pd.NA) for c in cols}], columns=cols)
    delta_path = delta_root / f"{payload.get('run_slug', 'run')}__{uuid4().hex}.parquet"
    tmp = delta_path.with_suffix(delta_path.suffix + ".tmp")
    new_row_df.to_parquet(tmp, index=False)
    tmp.replace(delta_path)


def clear_index_deltas(delta_dir_path: Path) -> None:
    shutil.rmtree(delta_dir_path, ignore_errors=True)
