"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/selection/top_n.py

Assumes candidates are pre-sorted by (-y_pred, id). Computes competition
ranks and marks selected_top_k_bool for all rows whose score meets the
k-th score threshold (ties included). Returns the full scored table with
ranks/flags plus the effective top-k count after ties.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from ..ranking import competition_rank, selection_threshold_for_top_k


def select_top_n(df: pd.DataFrame, score_col: str, k: int) -> Tuple[pd.DataFrame, int]:
    """
    Assumes df is already sorted by (-score, id).
    Returns df with columns ['id','sequence','y_pred','rank_competition','selected_top_k_bool']
    for the entire scored universe; selection flag is True for items >= tie-threshold.
    """
    scores = df[score_col].to_numpy(dtype=float)
    ranks = competition_rank(scores)  # requires DESC
    df = df.copy()
    df["rank_competition"] = ranks
    cutoff = selection_threshold_for_top_k(scores, k)
    df["selected_top_k_bool"] = df[score_col] >= cutoff
    top_k_effective = int(df["selected_top_k_bool"].sum())
    return df, top_k_effective
