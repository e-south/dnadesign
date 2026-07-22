"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/round/stages/selection_allocation_contracts.py

Validation and candidate-key projection for selection-batch allocation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ....config.types import SelectionBatchAllocationBlock
from ....core.utils import OpalError
from .selection_types import SelectionEvaluation


def candidate_key_by_id(
    *,
    candidate_df: pd.DataFrame,
    id_order_pool: List[str],
    deduplicate_by: str,
) -> dict[str, str]:
    """Validate and project candidate IDs to explicit batch keys.

    Candidate IDs are unique identities. Batch keys may be shared by more than
    one candidate because coordinated allocation uses those shared keys to
    deduplicate equivalent candidates.
    """

    key_column = str(deduplicate_by or "id").strip()
    required_columns = list(dict.fromkeys(["id", key_column]))
    missing = sorted(set(required_columns) - set(candidate_df.columns))
    if missing:
        raise OpalError(f"selection_batch candidate data is missing column(s): {missing}")

    candidates = candidate_df.loc[:, required_columns].copy()
    if candidates["id"].isna().any():
        raise OpalError("selection_batch candidate ids cannot be null.")
    candidates["id"] = candidates["id"].astype(str)
    if candidates["id"].str.strip().eq("").any():
        raise OpalError("selection_batch candidate ids cannot be blank.")
    if candidates["id"].duplicated().any():
        raise OpalError("selection_batch candidate ids must be unique.")
    if candidates[key_column].isna().any():
        raise OpalError(f"selection_batch deduplicate column {key_column!r} contains null values.")
    candidates[key_column] = candidates[key_column].astype(str)
    if candidates[key_column].str.strip().eq("").any():
        raise OpalError(f"selection_batch deduplicate column {key_column!r} contains blank values.")

    key_by_id = dict(zip(candidates["id"], candidates[key_column], strict=True))
    pool_ids = [str(candidate_id) for candidate_id in id_order_pool]
    if len(pool_ids) != len(set(pool_ids)):
        raise OpalError("selection_batch candidate pool ids must be unique.")
    unknown_ids = sorted(set(pool_ids) - set(key_by_id))
    if unknown_ids:
        raise OpalError(f"selection_batch candidate data is missing candidate pool ids: {unknown_ids[:10]}")
    pool_key_by_id = {candidate_id: key_by_id[candidate_id] for candidate_id in pool_ids}

    return pool_key_by_id


def validate_allocation_contract(
    *,
    selections: Dict[str, SelectionEvaluation],
    allocation: SelectionBatchAllocationBlock,
    expected_unique_count: int | None,
    pool_size: int,
) -> tuple[list[str], int]:
    """Require coherent exact quotas and complete within-view orderings."""

    if allocation.strategy != "round_robin_next_best_unallocated":
        raise OpalError(f"Unsupported selection_batch allocation strategy: {allocation.strategy!r}.")
    priority = [str(view_id) for view_id in allocation.view_priority]
    missing = sorted(set(selections) - set(priority))
    unknown = sorted(set(priority) - set(selections))
    if missing or unknown or len(priority) != len(selections):
        raise OpalError(
            "selection_batch allocation view_priority must be an exact permutation of selection views; "
            f"missing={missing}, unknown={unknown}."
        )

    quota_total = 0
    for view_id in priority:
        selection = selections[view_id]
        if selection.tie_handling != "ordinal":
            raise OpalError(f"Selection view {view_id!r} must use tie_handling='ordinal' for unique batch allocation.")
        if not bool(selection.sel_params.get("require_exact_top_k", False)):
            raise OpalError(
                f"Selection view {view_id!r} must set require_exact_top_k=true for unique batch allocation."
            )
        if selection.selected_effective != selection.top_k:
            raise OpalError(
                f"Selection view {view_id!r} produced {selection.selected_effective} preferred candidates; "
                f"expected exactly {selection.top_k}."
            )
        order_idx = np.asarray(selection.order_idx, dtype=int).reshape(-1)
        if (
            order_idx.size != pool_size
            or np.unique(order_idx).size != pool_size
            or (pool_size > 0 and (int(order_idx.min()) < 0 or int(order_idx.max()) >= pool_size))
        ):
            raise OpalError(f"Selection view {view_id!r} does not provide a complete candidate ordering.")
        aligned_arrays = {
            "ranks_ordinal": selection.ranks_ordinal,
            "ranks_competition": selection.ranks_competition,
            "preferred_bool": selection.preferred_bool,
            "selected_bool": selection.selected_bool,
            "scores": selection.scores,
            "objective scores": selection.y_obj_scalar,
        }
        for label, values in aligned_arrays.items():
            if np.asarray(values).reshape(-1).size != pool_size:
                raise OpalError(
                    f"Selection view {view_id!r} {label} does not align with the {pool_size}-candidate pool."
                )
        quota_total += int(selection.top_k)

    if expected_unique_count is None:
        raise OpalError("selection_batch allocation requires expected_unique_count.")
    if int(expected_unique_count) != quota_total:
        raise OpalError(
            "selection_batch expected_unique_count must equal the effective selection-view quota sum; "
            f"expected_unique_count={int(expected_unique_count)}, quota_sum={quota_total}."
        )
    if pool_size < quota_total:
        raise OpalError(
            f"selection_batch allocation requires {quota_total} unique candidates but the pool has only {pool_size}."
        )
    return priority, quota_total


__all__ = ["candidate_key_by_id", "validate_allocation_contract"]
