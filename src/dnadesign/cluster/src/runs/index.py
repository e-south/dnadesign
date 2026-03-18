"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/runs/index.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contracts import RunIndexEntry
from .index_store import (
    INDEX_MAX_DELTA_FILES,
    append_index_delta,
    clear_index_deltas,
    collapse_index_frames,
    delta_paths,
    index_delta_dir,
    index_snapshot_path,
    write_snapshot,
)


def compact_index(root: Path | None = None) -> None:
    idx_path = index_snapshot_path(root, materialize=True)
    delta_dir = index_delta_dir(root, materialize=True)
    if not delta_paths(delta_dir):
        return
    snapshot = list_runs(root=root)
    write_snapshot(snapshot, idx_path)
    clear_index_deltas(delta_dir)


def add_or_update_index(row: RunIndexEntry | dict[str, Any], root: Path | None = None) -> None:
    """
    Append a single run row into the run-index delta log and compact opportunistically.
    """
    delta_dir = index_delta_dir(root, materialize=True)
    append_index_delta(row, root=root)
    if len(delta_paths(delta_dir)) >= INDEX_MAX_DELTA_FILES:
        compact_index(root=root)


def _normalize_columns(columns: Iterable[str] | None) -> list[str]:
    if columns is None:
        return []
    return list(dict.fromkeys(str(column) for column in columns if column))


def _available_columns(idx_path: Path) -> set[str] | None:
    try:
        import pyarrow.dataset as ds
    except Exception:
        return None
    try:
        return set(ds.dataset(str(idx_path), format="parquet").schema.names)
    except Exception:
        return None


def _parquet_filters(
    filters: dict[str, Any], *, available_columns: set[str] | None
) -> list[list[tuple[str, str, Any]]] | None:
    predicates: list[tuple[str, str, Any]] = []
    for column, value in filters.items():
        if value is None:
            continue
        if available_columns is not None and column not in available_columns:
            continue
        if isinstance(value, (list, tuple, set, dict)):
            continue
        predicates.append((column, "=", value))
    return [predicates] if predicates else None


def _read_index_frame(idx_path: Path, *, filters: dict[str, Any], columns: Iterable[str] | None) -> pd.DataFrame:
    available_columns = _available_columns(idx_path)
    requested_columns = _normalize_columns(columns)
    read_columns: list[str] | None = None
    if requested_columns:
        read_columns = _normalize_columns((*requested_columns, *filters.keys(), "created_utc", "run_slug"))
        if available_columns is not None:
            read_columns = [column for column in read_columns if column in available_columns]
    kwargs: dict[str, Any] = {}
    if read_columns:
        kwargs["columns"] = read_columns
    parquet_filters = _parquet_filters(filters, available_columns=available_columns)
    if parquet_filters is not None:
        kwargs["filters"] = parquet_filters
    try:
        return pd.read_parquet(idx_path, **kwargs)
    except TypeError:
        kwargs.pop("filters", None)
        return pd.read_parquet(idx_path, **kwargs)


def list_runs(
    filters: dict[str, Any] | None = None,
    root: Path | None = None,
    *,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    idx_path = index_snapshot_path(root, materialize=False)
    delta_dir = index_delta_dir(root, materialize=False)
    delta_files = delta_paths(delta_dir)
    if not idx_path.exists() and not delta_files:
        return pd.DataFrame()
    active_filters = dict(filters or {})
    requested_columns = _normalize_columns(columns)
    frames: list[pd.DataFrame] = []
    order_offset = 0
    if idx_path.exists():
        snapshot = _read_index_frame(idx_path, filters=active_filters, columns=requested_columns or None)
        if not snapshot.empty:
            snapshot = snapshot.copy()
            snapshot["_cluster_order"] = np.arange(order_offset, order_offset + len(snapshot), dtype=np.int64)
            order_offset += len(snapshot)
            frames.append(snapshot)
    for delta_path in delta_files:
        delta = _read_index_frame(delta_path, filters=active_filters, columns=requested_columns or None)
        if delta.empty:
            continue
        delta = delta.copy()
        delta["_cluster_order"] = np.arange(order_offset, order_offset + len(delta), dtype=np.int64)
        order_offset += len(delta)
        frames.append(delta)
    if not frames:
        return pd.DataFrame(columns=requested_columns)
    df = collapse_index_frames(frames)
    for column, value in active_filters.items():
        if value is None or column not in df.columns:
            continue
        df = df[df[column] == value]
    if "_cluster_order" in df.columns:
        df = df.drop(columns="_cluster_order")
    if "created_utc" in df.columns:
        df = df.sort_values("created_utc", ascending=False, ignore_index=True)
    else:
        df = df.reset_index(drop=True)
    if requested_columns:
        visible_columns = [column for column in requested_columns if column in df.columns]
        return df.loc[:, visible_columns]
    return df
