"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/selection/top_n.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..registries import register_selection
from ..utils import competition_rank


@register_selection("top_n")
def select_top_n(
    ids: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    tie_handling: str = "competition_rank",
) -> dict:
    ids = np.asarray(ids)
    scores = np.asarray(scores, dtype=float)
    assert ids.shape[0] == scores.shape[0]

    # Stable sort key (-score, id) for determinism
    order = np.lexsort((ids, -scores))
    scores_sorted = scores[order]
    ranks = competition_rank(scores_sorted)

    if tie_handling != "competition_rank":
        raise ValueError(f"Unsupported tie_handling: {tie_handling}")

    if top_k <= 0 or len(scores_sorted) == 0:
        selected = np.zeros_like(scores_sorted, dtype=bool)
    else:
        max_rank = ranks[min(top_k - 1, len(ranks) - 1)]
        selected = ranks <= max_rank

    return dict(order_idx=order, rank_competition=ranks, selected_bool=selected)


def select_top_n_df(
    df: pd.DataFrame,
    *,
    score_col: str = "selection_score",
    top_k: int,
    tie_handling: str = "competition_rank",
) -> tuple[pd.DataFrame, int]:
    """
    Returns:
      df_out: sorted by (-score, id) with added columns:
        - rank_competition (int)
        - selected_top_k_bool (bool)
      top_k_effective: number of selected after tie inclusion
    """
    if score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not found in DataFrame")

    ids = df["id"].astype(str).to_numpy()
    scores = df[score_col].to_numpy(dtype=float)
    res = select_top_n(ids, scores, top_k=top_k, tie_handling=tie_handling)

    out = df.iloc[res["order_idx"]].copy().reset_index(drop=True)
    out["rank_competition"] = res["rank_competition"].astype(int)
    out["selected_top_k_bool"] = res["selected_bool"].astype(bool)
    top_k_effective = int(out["selected_top_k_bool"].sum())
    return out, top_k_effective
