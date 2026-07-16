"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/round/stages/selection_allocation.py

Deterministic cross-view allocation of unique selection-batch slots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np
import pandas as pd

from ....config.types import SelectionBatchAllocationBlock
from ....core.utils import OpalError
from .selection_allocation_contracts import candidate_key_by_id, validate_allocation_contract
from .selection_types import SelectionAllocationEvaluation, SelectionEvaluation

ALLOCATION_TRACE_COLUMNS = [
    "decision_order",
    "selection_view_id",
    "allocation_slot",
    "decision",
    "id",
    "selection_batch_key",
    "deduplicate_by",
    "rank_ordinal",
    "rank_competition",
    "score",
    "selection_score",
    "score_ref",
    "selection_origin",
    "conflicting_selection_view_id",
    "conflicting_allocation_slot",
]


def allocate_unique_selection_slots(
    *,
    candidate_df: pd.DataFrame,
    id_order_pool: List[str],
    selections: Dict[str, SelectionEvaluation],
    deduplicate_by: str | None,
    expected_unique_count: int | None,
    allocation: SelectionBatchAllocationBlock,
) -> SelectionAllocationEvaluation:
    """Allocate exact view quotas without comparing scores across views."""

    key_column = str(deduplicate_by or "id").strip()
    pool_ids = [str(candidate_id) for candidate_id in id_order_pool]
    key_by_id = candidate_key_by_id(
        candidate_df=candidate_df,
        id_order_pool=pool_ids,
        deduplicate_by=key_column,
        require_unique_keys=True,
    )
    priority, quota_total = validate_allocation_contract(
        selections=selections,
        allocation=allocation,
        expected_unique_count=expected_unique_count,
        pool_size=len(pool_ids),
    )
    unique_pool_keys = len(set(key_by_id.values()))
    if unique_pool_keys < quota_total:
        raise OpalError(
            f"selection_batch allocation requires {quota_total} unique {key_column} values, "
            f"but the candidate pool contains {unique_pool_keys}."
        )

    preferred_keys = {
        key_by_id[pool_ids[int(idx)]]
        for selection in selections.values()
        for idx in np.flatnonzero(selection.preferred_bool)
    }
    initial_membership_count = sum(int(selection.preferred_bool.sum()) for selection in selections.values())

    allocated_masks = {view_id: np.zeros(len(pool_ids), dtype=bool) for view_id in priority}
    allocation_slots = {view_id: np.zeros(len(pool_ids), dtype=int) for view_id in priority}
    cursors = {view_id: 0 for view_id in priority}
    owners: dict[str, tuple[str, int, str]] = {}
    trace_rows: list[dict[str, object]] = []
    decision_order = 0
    replacement_count_by_view = {view_id: 0 for view_id in priority}
    skipped_overlap_count_by_view = {view_id: 0 for view_id in priority}

    maximum_quota = max(int(selections[view_id].top_k) for view_id in priority)
    for allocation_slot in range(1, maximum_quota + 1):
        for view_id in priority:
            selection = selections[view_id]
            if allocation_slot > int(selection.top_k):
                continue
            allocated = False
            while cursors[view_id] < len(selection.order_idx):
                idx = int(selection.order_idx[cursors[view_id]])
                cursors[view_id] += 1
                candidate_id = pool_ids[idx]
                key = key_by_id[candidate_id]
                origin = "preferred_top_k" if bool(selection.preferred_bool[idx]) else "next_best_unallocated"
                decision_order += 1
                owner = owners.get(key)
                trace_row: dict[str, object] = {
                    "decision_order": decision_order,
                    "selection_view_id": view_id,
                    "allocation_slot": allocation_slot,
                    "decision": "allocated" if owner is None else "skipped_already_allocated",
                    "id": candidate_id,
                    "selection_batch_key": key,
                    "deduplicate_by": key_column,
                    "rank_ordinal": int(selection.ranks_ordinal[idx]),
                    "rank_competition": int(selection.ranks_competition[idx]),
                    "score": float(selection.y_obj_scalar[idx]),
                    "selection_score": float(selection.scores[idx]),
                    "score_ref": selection.score_ref,
                    "selection_origin": origin,
                    "conflicting_selection_view_id": None if owner is None else owner[0],
                    "conflicting_allocation_slot": None if owner is None else int(owner[1]),
                }
                trace_rows.append(trace_row)
                if owner is not None:
                    skipped_overlap_count_by_view[view_id] += 1
                    continue
                owners[key] = (view_id, allocation_slot, candidate_id)
                allocated_masks[view_id][idx] = True
                allocation_slots[view_id][idx] = allocation_slot
                if origin == "next_best_unallocated":
                    replacement_count_by_view[view_id] += 1
                allocated = True
                break
            if not allocated:
                raise OpalError(
                    f"selection_batch allocation could not fill view {view_id!r} slot {allocation_slot}; "
                    f"allocated_unique={len(owners)}, required_unique={quota_total}."
                )

    if len(owners) != quota_total:
        raise OpalError(f"selection_batch allocation produced {len(owners)} unique candidates; expected {quota_total}.")

    allocated_selections = {
        view_id: replace(
            selection,
            selected_bool=allocated_masks[view_id],
            allocation_slots=allocation_slots[view_id],
            selected_effective=int(allocated_masks[view_id].sum()),
        )
        for view_id, selection in selections.items()
    }
    trace = pd.DataFrame(trace_rows, columns=ALLOCATION_TRACE_COLUMNS)
    summary = {
        "strategy": allocation.strategy,
        "deduplicate_by": key_column,
        "view_priority": priority,
        "quota_by_view": {view_id: int(selections[view_id].top_k) for view_id in priority},
        "initial_membership_count": initial_membership_count,
        "initial_unique_count": len(preferred_keys),
        "overlap_membership_count": initial_membership_count - len(preferred_keys),
        "skipped_overlap_count": sum(skipped_overlap_count_by_view.values()),
        "replacement_count": sum(replacement_count_by_view.values()),
        "final_unique_count": len(owners),
        "expected_unique_count": int(expected_unique_count),
        "per_view": {
            view_id: {
                "quota": int(selections[view_id].top_k),
                "allocated": int(allocated_masks[view_id].sum()),
                "skipped_overlap_count": skipped_overlap_count_by_view[view_id],
                "replacement_count": replacement_count_by_view[view_id],
            }
            for view_id in priority
        },
    }
    return SelectionAllocationEvaluation(
        selections=allocated_selections,
        trace=trace,
        summary=summary,
    )


__all__ = ["ALLOCATION_TRACE_COLUMNS", "allocate_unique_selection_slots", "candidate_key_by_id"]
