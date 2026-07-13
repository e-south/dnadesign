"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/selection.py

Selection helpers for policy simulations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_top_rows(frame: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    if int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1; got {top_k}.")
    eligible = frame[frame["eligible"].astype(bool)].copy()
    scores = eligible["score"].to_numpy(dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("eligible policy scores must be finite.")
    eligible["rank"] = _competition_ranks(scores)
    return eligible[eligible["rank"] <= int(top_k)].reset_index(drop=True)


def _competition_ranks(sorted_scores: np.ndarray) -> np.ndarray:
    if len(sorted_scores) == 0:
        return np.array([], dtype=int)
    starts = np.ones(len(sorted_scores), dtype=bool)
    starts[1:] = ~np.isclose(
        sorted_scores[1:],
        sorted_scores[:-1],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    rank_starts = np.where(starts, np.arange(1, len(sorted_scores) + 1), 0)
    return np.maximum.accumulate(rank_starts)
