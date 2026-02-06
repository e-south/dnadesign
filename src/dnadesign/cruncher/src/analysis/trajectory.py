"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/trajectory.py

Build deterministic optimization trajectory points for plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _score_columns(tf_names: Iterable[str]) -> list[str]:
    return [f"score_{tf}" for tf in tf_names]


def _subsample_indices(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def project_scores(
    df: pd.DataFrame,
    tf_names: Iterable[str],
) -> Tuple[pd.Series, pd.Series, str, str, pd.Series, pd.Series]:
    tf_list = list(tf_names)
    score_cols = _score_columns(tf_list)
    missing = [col for col in score_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns for trajectory projection: {missing}")
    scores = df[score_cols].to_numpy(dtype=float)
    sorted_scores = np.sort(scores, axis=1)
    worst = pd.Series(sorted_scores[:, 0], index=df.index)
    second = pd.Series(sorted_scores[:, 1] if sorted_scores.shape[1] > 1 else sorted_scores[:, 0], index=df.index)
    if len(tf_list) == 2:
        x = df[score_cols[0]].astype(float)
        y = df[score_cols[1]].astype(float)
        return x, y, score_cols[0], score_cols[1], worst, second
    return worst, second, "worst_tf_score", "second_worst_tf_score", worst, second


def build_trajectory_points(
    sequences_df: pd.DataFrame,
    tf_names: Iterable[str],
    *,
    max_points: int,
) -> pd.DataFrame:
    if sequences_df is None or sequences_df.empty:
        return pd.DataFrame()

    df = sequences_df.copy()
    if "chain" in df.columns:
        df = df[df["chain"] == 0]
    if "draw" in df.columns:
        df = df.sort_values("draw")

    x, y, x_metric, y_metric, worst, second = project_scores(df, tf_names)
    score_cols = _score_columns(tf_names)
    cols = [c for c in ("draw", "phase", "chain") if c in df.columns] + score_cols
    out = df[cols].copy()
    out["x"] = x
    out["y"] = y
    out["x_metric"] = x_metric
    out["y_metric"] = y_metric
    out["worst_tf_score"] = worst
    out["second_worst_tf_score"] = second

    idx = _subsample_indices(len(out), max_points)
    return out.iloc[idx].reset_index(drop=True)
