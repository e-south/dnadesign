"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/io/read.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_table(ctx: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """Load Parquet/CSV with optional column projection."""
    p: Path = ctx["file"]
    if ctx["kind"] in ("usr", "parquet"):
        if columns:
            # Be strict; if the engine can't project, fail fast.
            # This prevents large accidental loads and keeps behavior predictable.
            return pd.read_parquet(p, columns=columns)
        return pd.read_parquet(p)
    elif ctx["kind"] == "csv":
        return pd.read_csv(p, usecols=columns) if columns else pd.read_csv(p)
    raise ValueError(f"Unknown context kind: {ctx['kind']}")


def _parse_json_array_cell(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.array(v, dtype=np.float32)
    if isinstance(v, str):
        try:
            arr = json.loads(v)
            return np.array(arr, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Cell is not a valid JSON array: {v[:50]}...") from e
    raise ValueError(f"Unsupported value type for JSON array column: {type(v)}")


def extract_X(
    df: pd.DataFrame, x_col: str | None = None, x_cols: list[str] | None = None
) -> np.ndarray:
    if (x_col is None) == (x_cols is None):
        raise ValueError("Provide exactly one of --x-col or --x-cols.")
    if x_col:
        if x_col not in df.columns:
            raise KeyError(f"X column '{x_col}' not found in the table.")
        s = df[x_col]
        first = s.iloc[0]
        # Preallocate to avoid list-of-arrays blowup
        if isinstance(first, (list, tuple, np.ndarray)):
            dim = int(len(first))
            n = int(len(s))
            X = np.empty((n, dim), dtype=np.float32)
            for i, v in enumerate(s):
                try:
                    arr = np.asarray(v, dtype=np.float32)
                except Exception as e:
                    raise TypeError(
                        f"Row {i} in '{x_col}' cannot be coerced to float32. "
                        f"First bad value (repr): {repr(v)[:80]}"
                    ) from e
                if arr.ndim != 1:
                    raise ValueError(f"Row {i} in '{x_col}' is not 1-D.")
                if arr.shape[0] != dim:
                    raise ValueError(
                        f"X must be fixed-length; row {i} has length {arr.shape[0]} but expected {dim}."
                    )
                X[i, :] = arr
        elif isinstance(first, str):
            arr0 = _parse_json_array_cell(first)
            dim = int(arr0.shape[0])
            n = int(len(s))
            X = np.empty((n, dim), dtype=np.float32)
            X[0, :] = arr0
            for i, v in enumerate(s.iloc[1:], start=1):
                arr = _parse_json_array_cell(v)
                if arr.ndim != 1:
                    raise ValueError(f"Row {i} in '{x_col}' is not 1-D.")
                if arr.shape[0] != dim:
                    raise ValueError(
                        f"X must be fixed-length; row {i} has length {arr.shape[0]} but expected {dim}."
                    )
                X[i, :] = arr
        else:
            raise TypeError(
                f"Unsupported cell type for X column '{x_col}': {type(first)}. "
                f"Use --x-cols to pass multiple numeric columns instead."
            )
        if not np.isfinite(X).all():
            bad = np.argwhere(~np.isfinite(X))
            i, j = int(bad[0, 0]), int(bad[0, 1])
            raise ValueError(
                f"X contains NaN/inf at row {i}, column {j}. Clean the source column before clustering."
            )
        return X
    else:
        for c in x_cols:
            if c not in df.columns:
                raise KeyError(f"X sub-column '{c}' not found.")
        sub = df[x_cols].to_numpy(dtype=np.float32, copy=False)
        if sub.ndim != 2:
            raise ValueError("X matrix must be 2-D after selecting x_cols.")
        if not np.isfinite(sub).all():
            i = np.argwhere(~np.isfinite(sub))
            raise ValueError(f"X contains NaN/inf at indices {i[:5].tolist()}...")
        return sub
