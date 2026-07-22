"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/candidate_scopes/builders.py

Candidate-scope builders for DenseGen TFBS probe campaign surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from .contracts import TfbsCandidateScope, count_fixed_slot_position_scope_policy


def build_count_fixed_slot_position_scope(labels: pd.DataFrame, *, label_name: str) -> TfbsCandidateScope:
    """Return candidate IDs where the target-family count is fixed to exactly one."""

    policy = count_fixed_slot_position_scope_policy(label_name)
    required = {"id", label_name, policy.target_family_count_column}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"count-fixed candidate scope missing column(s): {missing}")
    frame = labels.loc[:, ["id", label_name, policy.target_family_count_column]].copy()
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"count-fixed candidate scope requires unique ids; duplicates={duplicates}")
    counts = pd.to_numeric(frame[policy.target_family_count_column], errors="raise")
    scoped = frame.loc[counts == int(policy.required_count_value)].copy()
    if scoped.empty:
        raise ValueError(
            "count-fixed candidate scope has zero rows "
            f"for {label_name} where {policy.target_family_count_column} == {policy.required_count_value}"
        )
    label_values = pd.to_numeric(scoped[label_name], errors="raise")
    marginal = {str(key): int(value) for key, value in label_values.value_counts(dropna=False).sort_index().items()}
    if len(marginal) < 2:
        raise ValueError(
            "count-fixed candidate scope requires at least two target-label classes "
            f"for {label_name}; observed={marginal}"
        )
    ids = tuple(sorted(scoped["id"].astype(str).tolist()))
    return TfbsCandidateScope(policy=policy, ids=ids, row_count=len(ids), positive_label_marginal=marginal)


def filter_labels_to_scope(labels: pd.DataFrame, *, scope: TfbsCandidateScope) -> pd.DataFrame:
    """Return labels restricted to a previously materialized candidate scope."""

    if "id" not in labels.columns:
        raise ValueError("candidate-scope label filtering requires id column")
    ids = set(scope.ids)
    out = labels.loc[labels["id"].astype(str).isin(ids)].copy()
    found = set(out["id"].astype(str).tolist())
    missing = sorted(ids - found)
    if missing:
        raise ValueError(f"label table missing count-fixed candidate-scope id(s): {missing[:10]}")
    out["__scope_order__"] = (
        out["id"].astype(str).map({candidate_id: idx for idx, candidate_id in enumerate(scope.ids)})
    )
    return out.sort_values("__scope_order__").drop(columns=["__scope_order__"]).reset_index(drop=True)
