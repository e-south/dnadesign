"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/runtime/test_selection_batch_allocation.py

Deterministic coordinated allocation tests for multi-view selection batches.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.config.types import SelectionBatchAllocationBlock
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.runtime.round.stages.selection_allocation import allocate_unique_selection_slots
from dnadesign.opal.src.runtime.round.stages.selection_batch import build_selection_batch
from dnadesign.opal.src.runtime.round.stages.selection_types import SelectionEvaluation
from dnadesign.opal.src.runtime.run_round import _validate_allocated_batch_k_override


def _selection(
    view_id: str,
    *,
    id_order_pool: list[str],
    ranked_ids: list[str],
    top_k: int,
) -> SelectionEvaluation:
    index_by_id = {candidate_id: idx for idx, candidate_id in enumerate(id_order_pool)}
    order_idx = np.asarray([index_by_id[candidate_id] for candidate_id in ranked_ids], dtype=int)
    ranks_ordinal = np.empty(len(id_order_pool), dtype=int)
    ranks_ordinal[order_idx] = np.arange(1, len(id_order_pool) + 1, dtype=int)
    scores = np.asarray([float(len(id_order_pool) - rank) for rank in ranks_ordinal], dtype=float)
    preferred_bool = ranks_ordinal <= top_k
    return SelectionEvaluation(
        selection_view_id=view_id,
        y_obj_scalar=scores.copy(),
        diag={},
        obj_summary_stats=None,
        obj_name="scalar_identity_v1",
        obj_params={},
        obj_mode="maximize",
        score_ref=f"{view_id}/scalar",
        uncertainty_ref=None,
        sel_name="top_n",
        sel_params={
            "top_k": top_k,
            "tie_handling": "ordinal",
            "objective_mode": "maximize",
            "require_exact_top_k": True,
        },
        tie_handling="ordinal",
        mode="maximize",
        order_idx=order_idx,
        ranks_ordinal=ranks_ordinal,
        ranks_competition=ranks_ordinal.copy(),
        preferred_bool=preferred_bool.copy(),
        selected_bool=preferred_bool.copy(),
        allocation_slots=np.zeros(len(id_order_pool), dtype=int),
        selected_effective=int(preferred_bool.sum()),
        top_k=top_k,
        obj_sha="fixture",
        scores=scores.copy(),
        uq_scalar=None,
    )


def _allocation(*view_priority: str) -> SelectionBatchAllocationBlock:
    return SelectionBatchAllocationBlock(
        strategy="round_robin_next_best_unallocated",
        view_priority=list(view_priority),
    )


def _selected_ids(selection: SelectionEvaluation, id_order_pool: list[str]) -> list[str]:
    return [
        candidate_id
        for candidate_id, selected in zip(id_order_pool, selection.selected_bool, strict=True)
        if bool(selected)
    ]


def test_single_overlap_advances_lower_priority_view_to_next_unique_candidate() -> None:
    ids = ["a", "b", "c", "d"]
    candidates = pd.DataFrame({"id": ids, "sequence": ["AAA", "BBB", "CCC", "DDD"]})
    selections = {
        view_id: _selection(view_id, id_order_pool=ids, ranked_ids=["d", "c", "b", "a"], top_k=1)
        for view_id in ["target_a", "target_b"]
    }

    allocated = allocate_unique_selection_slots(
        candidate_df=candidates,
        id_order_pool=ids,
        selections=selections,
        deduplicate_by="sequence",
        expected_unique_count=2,
        allocation=_allocation("target_a", "target_b"),
    )

    assert _selected_ids(allocated.selections["target_a"], ids) == ["d"]
    assert _selected_ids(allocated.selections["target_b"], ids) == ["c"]
    assert allocated.summary["initial_unique_count"] == 1
    assert allocated.summary["skipped_overlap_count"] == 1
    assert allocated.summary["replacement_count"] == 1
    assert allocated.summary["final_unique_count"] == 2
    assert allocated.trace["decision"].tolist() == [
        "allocated",
        "skipped_already_allocated",
        "allocated",
    ]
    skipped = allocated.trace.iloc[1]
    assert skipped["conflicting_selection_view_id"] == "target_a"
    assert int(skipped["conflicting_allocation_slot"]) == 1

    batch = build_selection_batch(
        candidate_df=candidates,
        id_order_pool=ids,
        selections=allocated.selections,
        deduplicate_by="sequence",
        expected_unique_count=2,
        allocation_trace=allocated.trace,
        allocation_summary=allocated.summary,
    )
    assert batch.unique_count == 2
    assert set(batch.rows["allocation_view_id"]) == {"target_a", "target_b"}
    shared_preference = batch.rows.loc[batch.rows["id"] == "d", "preferred_view_ids"].iloc[0]
    assert shared_preference == ["target_a", "target_b"]
    replacement_membership = batch.rows.loc[batch.rows["id"] == "c", "selection_memberships"].iloc[0][0]
    assert replacement_membership["selection_origin"] == "next_best_unallocated"


def test_round_robin_allocation_fills_each_view_quota_without_cross_view_score_comparison() -> None:
    ids = list("abcdefg")
    ranked = list(reversed(ids))
    candidates = pd.DataFrame({"id": ids, "sequence": [candidate_id * 3 for candidate_id in ids]})
    selections = {
        view_id: _selection(view_id, id_order_pool=ids, ranked_ids=ranked, top_k=2)
        for view_id in ["target_a", "target_b", "target_c"]
    }

    allocated = allocate_unique_selection_slots(
        candidate_df=candidates,
        id_order_pool=ids,
        selections=selections,
        deduplicate_by="sequence",
        expected_unique_count=6,
        allocation=_allocation("target_a", "target_b", "target_c"),
    )

    assert _selected_ids(allocated.selections["target_a"], ids) == ["d", "g"]
    assert _selected_ids(allocated.selections["target_b"], ids) == ["c", "f"]
    assert _selected_ids(allocated.selections["target_c"], ids) == ["b", "e"]
    assert all(selection.selected_effective == 2 for selection in allocated.selections.values())
    assert allocated.summary["final_unique_count"] == 6
    assert allocated.summary["quota_by_view"] == {"target_a": 2, "target_b": 2, "target_c": 2}


def test_declared_priority_controls_overlap_owner() -> None:
    ids = ["a", "b", "c"]
    candidates = pd.DataFrame({"id": ids, "sequence": ["AAA", "BBB", "CCC"]})
    selections = {
        view_id: _selection(view_id, id_order_pool=ids, ranked_ids=["c", "b", "a"], top_k=1)
        for view_id in ["target_a", "target_b"]
    }

    allocated = allocate_unique_selection_slots(
        candidate_df=candidates,
        id_order_pool=ids,
        selections=selections,
        deduplicate_by="sequence",
        expected_unique_count=2,
        allocation=_allocation("target_b", "target_a"),
    )

    assert _selected_ids(allocated.selections["target_b"], ids) == ["c"]
    assert _selected_ids(allocated.selections["target_a"], ids) == ["b"]


def test_allocation_is_invariant_to_candidate_metadata_row_order() -> None:
    ids = ["a", "b", "c", "d"]
    candidates = pd.DataFrame({"id": ids, "sequence": ["AAA", "BBB", "CCC", "DDD"]})
    selections = {
        view_id: _selection(view_id, id_order_pool=ids, ranked_ids=["d", "c", "b", "a"], top_k=1)
        for view_id in ["target_a", "target_b"]
    }
    kwargs = {
        "id_order_pool": ids,
        "selections": selections,
        "deduplicate_by": "sequence",
        "expected_unique_count": 2,
        "allocation": _allocation("target_a", "target_b"),
    }

    first = allocate_unique_selection_slots(candidate_df=candidates, **kwargs)
    second = allocate_unique_selection_slots(
        candidate_df=candidates.sample(frac=1.0, random_state=11).reset_index(drop=True),
        **kwargs,
    )

    pd.testing.assert_frame_equal(first.trace, second.trace)
    assert first.summary == second.summary


def test_allocation_fails_when_unique_quota_exceeds_pool() -> None:
    ids = ["a", "b", "c"]
    candidates = pd.DataFrame({"id": ids, "sequence": ["AAA", "BBB", "CCC"]})
    selections = {
        view_id: _selection(view_id, id_order_pool=ids, ranked_ids=["c", "b", "a"], top_k=2)
        for view_id in ["target_a", "target_b"]
    }

    with pytest.raises(OpalError, match="requires 4 unique candidates.*pool has only 3"):
        allocate_unique_selection_slots(
            candidate_df=candidates,
            id_order_pool=ids,
            selections=selections,
            deduplicate_by="sequence",
            expected_unique_count=4,
            allocation=_allocation("target_a", "target_b"),
        )


def test_allocation_deduplicates_shared_keys_across_distinct_candidate_ids() -> None:
    ids = ["a", "b", "c", "d"]
    candidates = pd.DataFrame({"id": ids, "sequence": ["AAA", "AAA", "CCC", "DDD"]})
    selections = {
        "target_a": _selection(
            "target_a",
            id_order_pool=ids,
            ranked_ids=["a", "c", "d", "b"],
            top_k=1,
        ),
        "target_b": _selection(
            "target_b",
            id_order_pool=ids,
            ranked_ids=["b", "c", "d", "a"],
            top_k=1,
        ),
    }

    allocated = allocate_unique_selection_slots(
        candidate_df=candidates,
        id_order_pool=ids,
        selections=selections,
        deduplicate_by="sequence",
        expected_unique_count=2,
        allocation=_allocation("target_a", "target_b"),
    )

    assert _selected_ids(allocated.selections["target_a"], ids) == ["a"]
    assert _selected_ids(allocated.selections["target_b"], ids) == ["c"]
    assert allocated.trace["id"].tolist() == ["a", "b", "c"]
    assert allocated.trace["decision"].tolist() == [
        "allocated",
        "skipped_already_allocated",
        "allocated",
    ]
    assert allocated.summary["initial_unique_count"] == 1
    assert allocated.summary["skipped_overlap_count"] == 1
    assert allocated.summary["replacement_count"] == 1
    assert allocated.summary["final_unique_count"] == 2


def test_cli_top_k_override_cannot_drift_an_allocated_batch_contract() -> None:
    request = SimpleNamespace(
        cfg=SimpleNamespace(
            selection_batch=SimpleNamespace(
                allocation=_allocation("target_a", "target_b"),
                expected_unique_count=4,
            ),
            selection_views=[SimpleNamespace(id="target_a"), SimpleNamespace(id="target_b")],
        ),
        k_override=1,
    )

    with pytest.raises(OpalError, match="top-k override is incompatible.*override_quota_sum=2"):
        _validate_allocated_batch_k_override(request)
