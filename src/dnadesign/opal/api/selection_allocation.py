"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/selection_allocation.py

Public read-only preview of OPAL's coordinated unique-slot allocator.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..src.config.types import SelectionBatchAllocationBlock
from ..src.core.utils import ExitCodes, OpalError
from ..src.registries.selection import normalize_selection_result
from ..src.runtime.round.stages.selection_allocation import allocate_unique_selection_slots
from ..src.runtime.round.stages.selection_types import SelectionEvaluation

SELECTION_ALLOCATION_PREVIEW_API_VERSION = "1"


@dataclass(frozen=True)
class SelectionAllocationPreview:
    """Allocated rows, complete decision trace, and runtime summary for one preview."""

    allocated: pd.DataFrame
    trace: pd.DataFrame
    summary: dict[str, Any]


def preview_round_robin_next_best_unallocated(
    *,
    candidate_rows: pd.DataFrame,
    view_rows: pd.DataFrame,
    view_priority: Sequence[str],
) -> SelectionAllocationPreview:
    """Preview the production allocator from explicit metric-neutral rankings.

    ``candidate_rows`` requires ``id`` and ``dedup_key``. ``view_rows`` requires
    one complete ordinal ranking per view with ``selection_view_id``, ``id``,
    ``score``, ``rank``, and one constant positive ``top_k``. The function has no
    campaign side effects and delegates the allocation decision to OPAL's runtime
    allocator.
    """

    candidates = _validated_candidates(candidate_rows)
    priority = _validated_priority(view_priority)
    views = _validated_views(view_rows, candidate_ids=candidates["id"].tolist(), priority=priority)
    pool_ids = candidates["id"].tolist()
    selections = {view_id: _selection_evaluation(view_id, views[view_id], pool_ids=pool_ids) for view_id in priority}
    expected_unique_count = sum(int(selection.top_k) for selection in selections.values())
    result = allocate_unique_selection_slots(
        candidate_df=candidates.rename(columns={"dedup_key": "selection_batch_key"}),
        id_order_pool=pool_ids,
        selections=selections,
        deduplicate_by="selection_batch_key",
        expected_unique_count=expected_unique_count,
        allocation=SelectionBatchAllocationBlock(
            strategy="round_robin_next_best_unallocated",
            view_priority=list(priority),
        ),
    )
    allocated = (
        result.trace.loc[result.trace["decision"].eq("allocated")]
        .loc[
            :,
            [
                "selection_view_id",
                "allocation_slot",
                "id",
                "selection_batch_key",
                "rank_ordinal",
                "rank_competition",
                "score",
                "score_ref",
                "selection_origin",
            ],
        ]
        .rename(columns={"selection_batch_key": "dedup_key", "rank_ordinal": "rank"})
        .reset_index(drop=True)
    )
    return SelectionAllocationPreview(
        allocated=allocated,
        trace=result.trace.copy(),
        summary=dict(result.summary),
    )


def _validated_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise OpalError("Allocation preview candidate_rows must be a pandas DataFrame.", ExitCodes.BAD_ARGS)
    required = {"id", "dedup_key"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise OpalError(f"Allocation preview candidate_rows are missing columns: {missing}.", ExitCodes.BAD_ARGS)
    if raw.empty:
        raise OpalError("Allocation preview candidate_rows must not be empty.", ExitCodes.BAD_ARGS)
    candidates = raw.loc[:, ["id", "dedup_key"]].copy()
    if candidates["id"].isna().any():
        raise OpalError("Allocation preview candidate IDs cannot be null.", ExitCodes.BAD_ARGS)
    candidates["id"] = _exact_string_series(candidates["id"], field="candidate IDs")
    if candidates["id"].duplicated().any():
        raise OpalError("Allocation preview candidate IDs must be unique.", ExitCodes.BAD_ARGS)
    if candidates["dedup_key"].isna().any():
        raise OpalError("Allocation preview dedup_key contains null values.", ExitCodes.BAD_ARGS)
    candidates["dedup_key"] = _exact_string_series(candidates["dedup_key"], field="dedup_key")
    return candidates.reset_index(drop=True)


def _validated_priority(raw: Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise OpalError("Allocation preview view_priority must be a non-empty sequence.", ExitCodes.BAD_ARGS)
    if any(not isinstance(value, str) for value in raw):
        raise OpalError("Allocation preview view_priority entries must be strings.", ExitCodes.BAD_ARGS)
    priority = tuple(raw)
    if any(value != value.strip() for value in priority):
        raise OpalError(
            "Allocation preview view_priority entries must not contain leading or trailing whitespace.",
            ExitCodes.BAD_ARGS,
        )
    if any(not value for value in priority) or len(set(priority)) != len(priority):
        raise OpalError("Allocation preview view_priority entries must be non-empty and unique.", ExitCodes.BAD_ARGS)
    return priority


def _validated_views(
    raw: pd.DataFrame,
    *,
    candidate_ids: list[str],
    priority: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    if not isinstance(raw, pd.DataFrame):
        raise OpalError("Allocation preview view_rows must be a pandas DataFrame.", ExitCodes.BAD_ARGS)
    required = {"selection_view_id", "id", "score", "rank", "top_k"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise OpalError(f"Allocation preview view_rows are missing columns: {missing}.", ExitCodes.BAD_ARGS)
    if raw.empty:
        raise OpalError("Allocation preview view_rows must not be empty.", ExitCodes.BAD_ARGS)
    rows = raw.loc[:, ["selection_view_id", "id", "score", "rank", "top_k"]].copy()
    if rows[["selection_view_id", "id"]].isna().any().any():
        raise OpalError("Allocation preview view and candidate IDs cannot be null.", ExitCodes.BAD_ARGS)
    rows["selection_view_id"] = _exact_string_series(rows["selection_view_id"], field="selection view IDs")
    rows["id"] = _exact_string_series(rows["id"], field="view candidate IDs")
    present_views = set(rows["selection_view_id"])
    if present_views != set(priority):
        raise OpalError(
            "Allocation preview view_priority must be an exact permutation of view_rows; "
            f"missing={sorted(set(priority) - present_views)}, unknown={sorted(present_views - set(priority))}.",
            ExitCodes.BAD_ARGS,
        )
    for column in ("score", "rank", "top_k"):
        if rows[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise OpalError(
                "Allocation preview score, rank, and top_k values must be numeric, not boolean.",
                ExitCodes.BAD_ARGS,
            )
    try:
        scores = pd.to_numeric(rows["score"], errors="raise").to_numpy(dtype=float)
        ranks_numeric = pd.to_numeric(rows["rank"], errors="raise").to_numpy(dtype=float)
        top_k_numeric = pd.to_numeric(rows["top_k"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise OpalError(
            "Allocation preview score, rank, and top_k values must be numeric.", ExitCodes.BAD_ARGS
        ) from exc
    if not np.isfinite(scores).all():
        raise OpalError("Allocation preview scores must be finite.", ExitCodes.BAD_ARGS)
    if not np.isfinite(ranks_numeric).all() or not np.equal(ranks_numeric, np.floor(ranks_numeric)).all():
        raise OpalError("Allocation preview ranks must be finite integers.", ExitCodes.BAD_ARGS)
    if not np.isfinite(top_k_numeric).all() or not np.equal(top_k_numeric, np.floor(top_k_numeric)).all():
        raise OpalError("Allocation preview top_k values must be finite integers.", ExitCodes.BAD_ARGS)
    rows["score"] = scores
    rows["rank"] = ranks_numeric.astype(int)
    rows["top_k"] = top_k_numeric.astype(int)

    expected_ids = set(candidate_ids)
    by_view: dict[str, pd.DataFrame] = {}
    for view_id in priority:
        view = rows.loc[rows["selection_view_id"].eq(view_id)].copy()
        if view["id"].duplicated().any() or set(view["id"]) != expected_ids or len(view) != len(candidate_ids):
            raise OpalError(
                "Allocation preview views must contain exactly the same candidate IDs as candidate_rows.",
                ExitCodes.BAD_ARGS,
            )
        expected_ranks = list(range(1, len(candidate_ids) + 1))
        if sorted(view["rank"].tolist()) != expected_ranks:
            raise OpalError(
                f"Allocation preview view {view_id!r} must provide complete ordinal ranks 1..{len(candidate_ids)}.",
                ExitCodes.BAD_ARGS,
            )
        top_values = view["top_k"].unique().tolist()
        if len(top_values) != 1:
            raise OpalError(f"Allocation preview view {view_id!r} must declare one top_k value.", ExitCodes.BAD_ARGS)
        top_k = int(top_values[0])
        if top_k <= 0 or top_k > len(candidate_ids):
            raise OpalError(
                f"Allocation preview view {view_id!r} top_k must be in [1, {len(candidate_ids)}].",
                ExitCodes.BAD_ARGS,
            )
        ordered = view.sort_values(
            ["score", "id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        canonical_ranks = np.arange(1, len(ordered) + 1, dtype=int)
        if not np.array_equal(ordered["rank"].to_numpy(dtype=int), canonical_ranks):
            raise OpalError(
                f"Allocation preview view {view_id!r} ordinal ranks must match "
                "the production score-descending, candidate-ID-ascending order.",
                ExitCodes.BAD_ARGS,
            )
        by_view[view_id] = ordered.reset_index(drop=True)
    return by_view


def _selection_evaluation(
    view_id: str,
    rows: pd.DataFrame,
    *,
    pool_ids: list[str],
) -> SelectionEvaluation:
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(pool_ids)}
    order_idx = np.asarray([index_by_id[str(value)] for value in rows["id"]], dtype=int)
    scores = np.empty(len(pool_ids), dtype=float)
    for row in rows.itertuples(index=False):
        index = index_by_id[str(row.id)]
        scores[index] = float(row.score)
    top_k = int(rows["top_k"].iloc[0])
    normalized = normalize_selection_result(
        {"order_idx": order_idx},
        ids=np.asarray(pool_ids, dtype=str),
        scores=scores,
        top_k=top_k,
        tie_handling="ordinal",
        objective="maximize",
    )
    ranks_ordinal = np.asarray(normalized["ranks"], dtype=int)
    ranks_competition = np.asarray(normalized["rank_competition"], dtype=int)
    preferred = np.asarray(normalized["selected_bool"], dtype=bool)
    return SelectionEvaluation(
        selection_view_id=view_id,
        y_obj_scalar=scores.copy(),
        diag={},
        obj_summary_stats=None,
        obj_name="allocation_preview",
        obj_params={},
        obj_mode="maximize",
        score_ref=f"{view_id}/preview_score",
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
        ranks_competition=ranks_competition,
        preferred_bool=preferred.copy(),
        selected_bool=preferred.copy(),
        allocation_slots=np.zeros(len(pool_ids), dtype=int),
        selected_effective=int(preferred.sum()),
        top_k=top_k,
        obj_sha="allocation_preview",
        scores=scores.copy(),
        uq_scalar=None,
    )


def _exact_string_series(values: pd.Series, *, field: str) -> pd.Series:
    if any(not isinstance(value, str) for value in values.tolist()):
        raise OpalError(f"Allocation preview {field} must be strings.", ExitCodes.BAD_ARGS)
    if values.map(lambda value: value != value.strip()).any():
        raise OpalError(
            f"Allocation preview {field} must not contain leading or trailing whitespace.",
            ExitCodes.BAD_ARGS,
        )
    if values.eq("").any():
        raise OpalError(f"Allocation preview {field} cannot be blank.", ExitCodes.BAD_ARGS)
    return values.copy()


__all__ = [
    "SELECTION_ALLOCATION_PREVIEW_API_VERSION",
    "SelectionAllocationPreview",
    "preview_round_robin_next_best_unallocated",
]
