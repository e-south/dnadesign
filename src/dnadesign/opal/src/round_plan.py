"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/round_plan.py

Shared round planning logic for explain/run (keeps counts aligned).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .data_access import RecordsStore


@dataclass(frozen=True)
class RoundPlan:
    as_of_round: int
    training_df: pd.DataFrame
    candidate_df: pd.DataFrame
    candidate_total_before_filter: int
    candidate_filtered_out: int
    training_policy: Dict[str, object]
    training_dedup_policy: str
    allow_resuggest: bool
    selection_excludes_labeled: bool
    warnings: List[str]


def plan_round(
    store: RecordsStore,
    df: pd.DataFrame,
    cfg,
    as_of_round: int,
    *,
    warnings: Optional[List[str]] = None,
) -> RoundPlan:
    policy = cfg.training.policy or {}
    cumulative_training = bool(policy.get("cumulative_training", True))
    dedup_policy = str(policy.get("label_cross_round_deduplication_policy", "latest_only"))
    allow_resuggest = bool(policy.get("allow_resuggesting_candidates_until_labeled", True))

    train_df = store.training_labels_with_round(
        df,
        int(as_of_round),
        cumulative_training=cumulative_training,
        dedup_policy=dedup_policy,
    )

    cand_df = store.candidate_universe(df, int(as_of_round))
    total_before = int(len(cand_df))

    sel_params = dict(cfg.selection.selection.params)
    exclude_already_labeled = bool(sel_params.get("exclude_already_labeled", True))
    if exclude_already_labeled:
        labeled_ids = (
            store.labeled_id_set_leq_round(df, int(as_of_round))
            if allow_resuggest
            else store.labeled_id_set_any_round(df)
        )
        if labeled_ids:
            cand_df = cand_df.loc[~cand_df["id"].astype(str).isin(labeled_ids)].copy()

    filtered_out = total_before - int(len(cand_df))

    return RoundPlan(
        as_of_round=int(as_of_round),
        training_df=train_df,
        candidate_df=cand_df,
        candidate_total_before_filter=total_before,
        candidate_filtered_out=filtered_out,
        training_policy=dict(policy),
        training_dedup_policy=dedup_policy,
        allow_resuggest=allow_resuggest,
        selection_excludes_labeled=exclude_already_labeled,
        warnings=list(warnings or []),
    )
