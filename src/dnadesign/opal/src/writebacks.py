"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/writebacks.py

Round write-backs into records.parquet.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd


def write_round_columns(
    df: pd.DataFrame,
    slug: str,
    round_k: int,
    scored_df: pd.DataFrame,
    allow_overwrite: bool = False,
    *,
    objective_name: Optional[str] = "sfxi_v1",
    fingerprint_short: Optional[str] = None,
) -> pd.DataFrame:
    """
    Persist per-round outputs back to the source records table.

    scored_df columns expected:
      - id
      - y_pred_vec (list or JSON-string)
      - y_pred_std_vec (list or JSON-string)   # not stored; scalar uncertainty is
      - selection_score (float)
      - rank_competition (int)
      - selected_top_k_bool (bool)
      - uncertainty_mean_all_std (float)

    Writes columns:
      opal__{slug}__r{round}__pred_y
      opal__{slug}__r{round}__selection_score__{objective_name}
      opal__{slug}__r{round}__rank_competition
      opal__{slug}__r{round}__selected_top_k_bool
      opal__{slug}__r{round}__uncertainty__mean_all_std
      opal__{slug}__r{round}__flags
      opal__{slug}__r{round}__fingerprint
    """
    pred_col = f"opal__{slug}__r{round_k}__pred_y"
    score_suffix = objective_name or "objective"
    score_col = f"opal__{slug}__r{round_k}__selection_score__{score_suffix}"
    rank_col = f"opal__{slug}__r{round_k}__rank_competition"
    sel_col = f"opal__{slug}__r{round_k}__selected_top_k_bool"
    uncs_col = f"opal__{slug}__r{round_k}__uncertainty__mean_all_std"
    flags_col = f"opal__{slug}__r{round_k}__flags"
    fp_col = f"opal__{slug}__r{round_k}__fingerprint"

    df_in = scored_df.copy()

    def _norm(v):
        if isinstance(v, str):
            try:
                json.loads(v)  # validate
                return v
            except Exception:
                pass
        return json.dumps(v)

    if "y_pred_vec" in df_in.columns:
        df_in["y_pred_vec"] = df_in["y_pred_vec"].map(_norm)

    left = df.set_index("id")
    # Optional diag/flags passthroughs
    cols = [
        "y_pred_vec",
        "selection_score",
        "rank_competition",
        "selected_top_k_bool",
        "uncertainty_mean_all_std",
    ]
    if "flags" in df_in.columns:
        cols.append("flags")
    right = df_in.set_index("id")[cols]
    joined = left.join(right, how="left")

    # overwrite policy
    for col, src in [
        (pred_col, "y_pred_vec"),
        (score_col, "selection_score"),
        (rank_col, "rank_competition"),
        (sel_col, "selected_top_k_bool"),
        (uncs_col, "uncertainty_mean_all_std"),
        (flags_col, "flags" if "flags" in right.columns else None),
    ]:
        if src is None:
            continue
        if col in joined.columns and not allow_overwrite:
            mask_old = joined[col].notna()
            mask_new = joined[src].notna()
            if (mask_old & mask_new).any():
                raise ValueError(
                    f"Column already exists with values: {col}. Use --resume/--force to overwrite."
                )
        joined[col] = joined[src]

    # Broadcast fingerprint (constant across rows for a given run)
    if fingerprint_short:
        if fp_col in joined.columns and not allow_overwrite and joined[fp_col].notna().any():
            raise ValueError(
                f"Column already exists with values: {fp_col}. Use --resume/--force to overwrite."
            )
        joined[fp_col] = fingerprint_short

    # drop staging columns that were only used for the join
    joined = joined.drop(
        columns=[
            c
            for c in [
                "y_pred_vec",
                "selection_score",
                "rank_competition",
                "selected_top_k_bool",
                "uncertainty_mean_all_std",
                "flags",
            ]
            if c in joined.columns
        ]
    )

    return joined.reset_index()
