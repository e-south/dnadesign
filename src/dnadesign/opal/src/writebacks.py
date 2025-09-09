"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/writebacks.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from .utils import ExitCodes, OpalError


def write_round_columns(
    df: pd.DataFrame,
    campaign_slug: str,
    k: int,
    scored: pd.DataFrame,  # columns: id,y_pred,rank_competition,selected_top_k_bool
    allow_overwrite: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    pred_col = f"opal__{campaign_slug}__r{k}__pred"
    rank_col = f"opal__{campaign_slug}__r{k}__rank_competition"
    sel_col = f"opal__{campaign_slug}__r{k}__selected_top_k_bool"

    for col in (pred_col, rank_col, sel_col):
        if col in df.columns and not allow_overwrite:
            raise OpalError(
                f"Column already exists: {col} (use --resume/--force to overwrite)",
                ExitCodes.EXISTS_NEEDS_RESUME,
            )

    if pred_col not in df.columns:
        df[pred_col] = pd.NA
    if rank_col not in df.columns:
        df[rank_col] = pd.NA
    if sel_col not in df.columns:
        df[sel_col] = False

    # align by id
    map_idx = {i: idx for idx, i in enumerate(df["id"].tolist())}
    for row in scored[
        ["id", "y_pred", "rank_competition", "selected_top_k_bool"]
    ].itertuples(index=False):
        rid, yhat, rnk, sel = row
        if rid in map_idx:
            idx = map_idx[rid]
            df.at[idx, pred_col] = float(yhat)
            df.at[idx, rank_col] = int(rnk)
            df.at[idx, sel_col] = bool(sel)

    return df
