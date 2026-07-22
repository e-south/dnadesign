"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/runtime/round/stages/selection_batch.py

Assemble final per-view selections into one deduplicated batch artifact.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ....core.utils import OpalError
from .selection_allocation import ALLOCATION_TRACE_COLUMNS
from .selection_types import SelectionBatchEvaluation, SelectionEvaluation


def build_selection_batch(
    *,
    candidate_df: pd.DataFrame,
    id_order_pool: List[str],
    selections: Dict[str, SelectionEvaluation],
    deduplicate_by: Optional[str],
    expected_unique_count: Optional[int],
    allocation_trace: pd.DataFrame | None = None,
    allocation_summary: dict[str, Any] | None = None,
) -> SelectionBatchEvaluation:
    key_column = str(deduplicate_by or "id").strip()
    required = {"id", key_column}
    missing = sorted(required - set(candidate_df.columns))
    if missing:
        raise OpalError(f"selection_batch candidate data is missing column(s): {missing}")
    candidates = candidate_df.loc[:, sorted(required)].copy()
    candidates["id"] = candidates["id"].astype(str)
    if candidates["id"].duplicated().any():
        raise OpalError("selection_batch candidate ids must be unique.")
    if candidates[key_column].isna().any():
        raise OpalError(f"selection_batch deduplicate column {key_column!r} contains null values.")
    by_id = candidates.set_index("id", drop=False)
    preferred_by_key: dict[str, list[str]] = {}
    for view_id, selection in selections.items():
        if len(selection.preferred_bool) != len(id_order_pool):
            raise OpalError(f"Selection view {view_id!r} preference mask does not align with the candidate pool.")
        for idx in np.flatnonzero(selection.preferred_bool):
            candidate_id = str(id_order_pool[int(idx)])
            if candidate_id not in by_id.index:
                raise OpalError(f"Selection view {view_id!r} references unknown candidate id {candidate_id!r}.")
            key = str(by_id.at[candidate_id, key_column])
            preferred_views = preferred_by_key.setdefault(key, [])
            if view_id not in preferred_views:
                preferred_views.append(view_id)

    batch: dict[str, dict[str, Any]] = {}
    for view_id, selection in selections.items():
        if len(selection.selected_bool) != len(id_order_pool):
            raise OpalError(f"Selection view {view_id!r} does not align with the candidate pool.")
        for idx in np.flatnonzero(selection.selected_bool):
            candidate_id = str(id_order_pool[int(idx)])
            if candidate_id not in by_id.index:
                raise OpalError(f"Selection view {view_id!r} references unknown candidate id {candidate_id!r}.")
            key_value = by_id.at[candidate_id, key_column]
            key = str(key_value)
            entry = batch.setdefault(
                key,
                {
                    "id": candidate_id,
                    "selection_batch_key": key,
                    "deduplicate_by": key_column,
                    "selection_view_ids": [],
                    "selection_memberships": [],
                    "preferred_view_ids": list(preferred_by_key.get(key, [])),
                    "allocation_view_id": None,
                    "allocation_slot": None,
                },
            )
            if entry["id"] != candidate_id:
                raise OpalError(
                    f"selection_batch {key_column} value {key!r} maps to multiple candidate ids: "
                    f"{entry['id']!r}, {candidate_id!r}."
                )
            entry["selection_view_ids"].append(view_id)
            allocation_slot = int(selection.allocation_slots[int(idx)])
            if allocation_slot > 0:
                if entry["allocation_view_id"] not in (None, view_id):
                    raise OpalError(
                        f"selection_batch key {key!r} was allocated to multiple views: "
                        f"{entry['allocation_view_id']!r}, {view_id!r}."
                    )
                entry["allocation_view_id"] = view_id
                entry["allocation_slot"] = allocation_slot
            entry["selection_memberships"].append(
                {
                    "selection_view_id": view_id,
                    "rank": int(selection.ranks_competition[int(idx)]),
                    "rank_ordinal": int(selection.ranks_ordinal[int(idx)]),
                    "score": float(selection.y_obj_scalar[int(idx)]),
                    "selection_score": float(selection.scores[int(idx)]),
                    "score_ref": selection.score_ref,
                    "allocation_slot": allocation_slot if allocation_slot > 0 else None,
                    "selection_origin": (
                        "next_best_unallocated"
                        if allocation_slot > 0 and not bool(selection.preferred_bool[int(idx)])
                        else "preferred_top_k"
                    ),
                }
            )

    rows = pd.DataFrame(list(batch.values()))
    if not rows.empty:
        rows = rows.sort_values(["selection_batch_key", "id"], kind="stable").reset_index(drop=True)
    unique_count = int(len(rows))
    if expected_unique_count is not None and unique_count != int(expected_unique_count):
        raise OpalError(
            f"selection_batch expected {int(expected_unique_count)} unique candidates, observed {unique_count}. "
            "OPAL does not fill or discard selection slots implicitly."
        )
    trace = allocation_trace.copy() if allocation_trace is not None else pd.DataFrame(columns=ALLOCATION_TRACE_COLUMNS)
    if allocation_summary is None:
        initial_membership_count = sum(int(selection.preferred_bool.sum()) for selection in selections.values())
        summary = {
            "strategy": "logical_union",
            "deduplicate_by": key_column,
            "initial_membership_count": initial_membership_count,
            "initial_unique_count": unique_count,
            "overlap_membership_count": initial_membership_count - unique_count,
            "skipped_overlap_count": 0,
            "replacement_count": 0,
            "final_unique_count": unique_count,
            "expected_unique_count": None if expected_unique_count is None else int(expected_unique_count),
        }
    else:
        summary = dict(allocation_summary)
    return SelectionBatchEvaluation(
        rows=rows,
        deduplicate_by=key_column,
        unique_count=unique_count,
        expected_unique_count=(None if expected_unique_count is None else int(expected_unique_count)),
        allocation_trace=trace,
        allocation_summary=summary,
    )


__all__ = ["build_selection_batch"]
