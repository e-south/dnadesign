"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plots/trajectory_common.py

Shared helpers for trajectory scatter and sweep plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_CHAIN_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
_CHAIN_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


def _require_df(df: pd.DataFrame | None, *, name: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{name} data is required for trajectory plot.")
    return df


def _require_numeric(df: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"{context} missing required column '{column}'.")
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{context} column '{column}' must be numeric.")
    return values.astype(float)


def _prepare_chain_df(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = _require_df(trajectory_df, name="Trajectory").copy()
    if "chain" not in plot_df.columns:
        raise ValueError("Trajectory points must include chain for chain trajectory plotting.")
    chain_values = _require_numeric(plot_df, "chain", context="Trajectory")
    if "sweep_idx" in plot_df.columns:
        sweep_values = _require_numeric(plot_df, "sweep_idx", context="Trajectory")
    elif "sweep" in plot_df.columns:
        sweep_values = _require_numeric(plot_df, "sweep", context="Trajectory")
    else:
        raise ValueError("Trajectory points must include sweep_idx for trajectory plotting.")
    plot_df["chain"] = chain_values.astype(int)
    plot_df["sweep_idx"] = sweep_values.astype(int)
    return plot_df


def _best_update_indices(values: np.ndarray) -> list[int]:
    if values.size == 0:
        return []
    indices: list[int] = []
    running_best = float("-inf")
    for idx, value in enumerate(values):
        value_f = float(value)
        if value_f > running_best:
            indices.append(int(idx))
            running_best = value_f
    return indices


def _stride_indices(
    n: int,
    *,
    stride: int,
    priority_indices: list[int] | None = None,
) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=int)
    if stride <= 1 or n <= 1:
        keep_idx = np.arange(n, dtype=int)
    else:
        keep_idx = np.arange(0, n, int(stride), dtype=int)
        if n - 1 not in keep_idx:
            keep_idx = np.append(keep_idx, n - 1)
    if priority_indices:
        for idx in priority_indices:
            if 0 <= int(idx) < n:
                keep_idx = np.append(keep_idx, int(idx))
    return np.unique(keep_idx.astype(int))
