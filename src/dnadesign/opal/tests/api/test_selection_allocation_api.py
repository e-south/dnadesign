"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/api/test_selection_allocation_api.py

Parity tests for the read-only public selection-allocation preview API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import dnadesign.opal as opal
from dnadesign.opal.api import preview_round_robin_next_best_unallocated
from dnadesign.opal.src.config.types import SelectionBatchAllocationBlock
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.runtime.round.stages.selection_allocation import allocate_unique_selection_slots
from dnadesign.opal.src.runtime.round.stages.selection_types import SelectionEvaluation


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "dedup_key": ["AAA", "AAA", "CCC", "DDD"],
        }
    )


def _view_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for view_id, ordered_ids in {
        "target-a": ["a", "c", "d", "b"],
        "target-b": ["b", "c", "d", "a"],
    }.items():
        for rank, candidate_id in enumerate(ordered_ids, start=1):
            rows.append(
                {
                    "selection_view_id": view_id,
                    "id": candidate_id,
                    "score": float(10 - rank),
                    "rank": rank,
                    "top_k": 1,
                }
            )
    return pd.DataFrame.from_records(rows)


def _runtime_selection(view_id: str, view_rows: pd.DataFrame, *, pool_ids: list[str]) -> SelectionEvaluation:
    rows = view_rows.loc[view_rows["selection_view_id"].eq(view_id)].sort_values("rank", kind="mergesort")
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(pool_ids)}
    order_idx = np.asarray([index_by_id[str(value)] for value in rows["id"]], dtype=int)
    ranks = np.empty(len(pool_ids), dtype=int)
    scores = np.empty(len(pool_ids), dtype=float)
    for row in rows.itertuples(index=False):
        index = index_by_id[str(row.id)]
        ranks[index] = int(row.rank)
        scores[index] = float(row.score)
    top_k = int(rows["top_k"].iloc[0])
    preferred = ranks <= top_k
    return SelectionEvaluation(
        selection_view_id=view_id,
        y_obj_scalar=scores.copy(),
        diag={},
        obj_summary_stats=None,
        obj_name="preview_fixture",
        obj_params={},
        obj_mode="maximize",
        score_ref=f"{view_id}/preview_score",
        uncertainty_ref=None,
        sel_name="top_n",
        sel_params={"top_k": top_k, "tie_handling": "ordinal", "require_exact_top_k": True},
        tie_handling="ordinal",
        mode="maximize",
        order_idx=order_idx,
        ranks_ordinal=ranks,
        ranks_competition=ranks.copy(),
        preferred_bool=preferred.copy(),
        selected_bool=preferred.copy(),
        allocation_slots=np.zeros(len(pool_ids), dtype=int),
        selected_effective=int(preferred.sum()),
        top_k=top_k,
        obj_sha="preview_fixture",
        scores=scores.copy(),
        uq_scalar=None,
    )


def test_public_allocation_preview_is_exactly_runtime_allocator_output() -> None:
    candidates = _candidate_rows()
    views = _view_rows()
    priority = ["target-a", "target-b"]

    preview = preview_round_robin_next_best_unallocated(
        candidate_rows=candidates,
        view_rows=views,
        view_priority=priority,
    )
    pool_ids = candidates["id"].astype(str).tolist()
    runtime = allocate_unique_selection_slots(
        candidate_df=candidates.rename(columns={"dedup_key": "selection_batch_key"}),
        id_order_pool=pool_ids,
        selections={view_id: _runtime_selection(view_id, views, pool_ids=pool_ids) for view_id in priority},
        deduplicate_by="selection_batch_key",
        expected_unique_count=2,
        allocation=SelectionBatchAllocationBlock(
            strategy="round_robin_next_best_unallocated",
            view_priority=priority,
        ),
    )

    pd.testing.assert_frame_equal(preview.trace, runtime.trace)
    assert preview.summary == runtime.summary
    assert preview.allocated[["selection_view_id", "allocation_slot", "id", "dedup_key"]].to_dict(orient="records") == [
        {"selection_view_id": "target-a", "allocation_slot": 1, "id": "a", "dedup_key": "AAA"},
        {"selection_view_id": "target-b", "allocation_slot": 1, "id": "c", "dedup_key": "CCC"},
    ]


def test_allocation_preview_is_available_from_the_package_public_surface() -> None:
    assert opal.preview_round_robin_next_best_unallocated is preview_round_robin_next_best_unallocated
    assert opal.SELECTION_ALLOCATION_PREVIEW_API_VERSION == "1"


def test_public_allocation_preview_preserves_production_tie_ranks_and_order() -> None:
    rows = _view_rows().loc[lambda frame: frame["selection_view_id"].eq("target-a")].copy()
    rows["top_k"] = 2
    rows.loc[rows["id"].isin(["a", "c"]), "score"] = 9.0

    preview = preview_round_robin_next_best_unallocated(
        candidate_rows=_candidate_rows(),
        view_rows=rows,
        view_priority=["target-a"],
    )

    allocated = preview.allocated.sort_values("allocation_slot")
    assert allocated["id"].tolist() == ["a", "c"]
    assert allocated["rank"].tolist() == [1, 2]
    assert allocated["rank_competition"].tolist() == [1, 1]


def test_public_allocation_preview_rejects_noncanonical_order_with_tied_scores() -> None:
    rows = _view_rows().loc[lambda frame: frame["selection_view_id"].eq("target-a")].copy()
    rows.loc[rows["id"].isin(["a", "c"]), "score"] = 9.0
    rows.loc[rows["id"].eq("a"), "rank"] = 2
    rows.loc[rows["id"].eq("c"), "rank"] = 1

    with pytest.raises(OpalError, match="production score-descending"):
        preview_round_robin_next_best_unallocated(
            candidate_rows=_candidate_rows(),
            view_rows=rows,
            view_priority=["target-a"],
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.assign(rank=1), "complete ordinal ranks"),
        (
            lambda rows: rows.loc[~((rows["selection_view_id"] == "target-b") & (rows["id"] == "d"))],
            "same candidate IDs",
        ),
        (lambda rows: rows.assign(top_k=[1, 1, 1, 1, 2, 1, 1, 1]), "one top_k"),
    ],
)
def test_public_allocation_preview_rejects_ambiguous_view_contracts(mutate, message: str) -> None:
    with pytest.raises(OpalError, match=message):
        preview_round_robin_next_best_unallocated(
            candidate_rows=_candidate_rows(),
            view_rows=mutate(_view_rows()),
            view_priority=["target-a", "target-b"],
        )


def test_public_allocation_preview_rejects_nonunique_or_missing_dedup_keys() -> None:
    candidates = _candidate_rows()
    candidates.loc[0, "id"] = "b"
    with pytest.raises(OpalError, match="candidate IDs must be unique"):
        preview_round_robin_next_best_unallocated(
            candidate_rows=candidates,
            view_rows=_view_rows(),
            view_priority=["target-a", "target-b"],
        )

    candidates = _candidate_rows()
    candidates.loc[0, "dedup_key"] = None
    with pytest.raises(OpalError, match="dedup_key contains null"):
        preview_round_robin_next_best_unallocated(
            candidate_rows=candidates,
            view_rows=_view_rows(),
            view_priority=["target-a", "target-b"],
        )


@pytest.mark.parametrize(
    "column",
    ["id", "dedup_key"],
)
def test_public_allocation_preview_rejects_candidate_identity_whitespace(column: str) -> None:
    candidates = _candidate_rows()
    candidates.loc[0, column] = f" {candidates.loc[0, column]}"
    with pytest.raises(OpalError, match="leading or trailing whitespace"):
        preview_round_robin_next_best_unallocated(
            candidate_rows=candidates,
            view_rows=_view_rows(),
            view_priority=["target-a", "target-b"],
        )


@pytest.mark.parametrize("column", ["selection_view_id", "id"])
def test_public_allocation_preview_rejects_view_identity_whitespace(column: str) -> None:
    rows = _view_rows()
    rows.loc[0, column] = f"{rows.loc[0, column]} "
    with pytest.raises(OpalError, match="leading or trailing whitespace"):
        preview_round_robin_next_best_unallocated(
            candidate_rows=_candidate_rows(),
            view_rows=rows,
            view_priority=["target-a", "target-b"],
        )


def test_public_allocation_preview_reports_missing_and_unknown_views_in_their_actual_directions() -> None:
    rows = _view_rows().replace({"target-b": "target-c"})
    with pytest.raises(OpalError, match=r"missing=\['target-b'\], unknown=\['target-c'\]"):
        preview_round_robin_next_best_unallocated(
            candidate_rows=_candidate_rows(),
            view_rows=rows,
            view_priority=["target-a", "target-b"],
        )


@pytest.mark.parametrize("column", ["score", "rank", "top_k"])
def test_public_allocation_preview_rejects_boolean_numeric_fields(column: str) -> None:
    rows = _view_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = True
    with pytest.raises(OpalError, match="numeric, not boolean"):
        preview_round_robin_next_best_unallocated(
            candidate_rows=_candidate_rows(),
            view_rows=rows,
            view_priority=["target-a", "target-b"],
        )
