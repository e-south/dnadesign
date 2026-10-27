"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/selection_batch_contract.py

Fail-closed validation for persisted OPAL selection-batch evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from ..core.utils import OpalError

SELECTION_BATCH_REQUIRED_COLUMNS = frozenset(
    {
        "run_id",
        "as_of_round",
        "campaign_slug",
        "id",
        "selection_batch_key",
        "deduplicate_by",
        "selection_view_ids",
        "selection_memberships",
        "preferred_view_ids",
        "allocation_view_id",
        "allocation_slot",
    }
)
SELECTION_MEMBERSHIP_REQUIRED_FIELDS = frozenset(
    {
        "selection_view_id",
        "rank",
        "rank_ordinal",
        "score",
        "selection_score",
        "score_ref",
        "allocation_slot",
        "selection_origin",
    }
)
SELECTION_ORIGINS = frozenset({"preferred_top_k", "next_best_unallocated"})


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (Mapping, Sequence, np.ndarray)) and not isinstance(value, (str, bytes)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _required_text(value: Any, *, field: str, candidate_id: str | None = None) -> str:
    text = "" if _is_missing_scalar(value) else str(value).strip()
    if not text:
        scope = "" if candidate_id is None else f" for candidate {candidate_id!r}"
        raise OpalError(f"Selection batch {field}{scope} must be non-empty.")
    return text


def _optional_text(value: Any, *, field: str, candidate_id: str) -> str | None:
    if _is_missing_scalar(value):
        return None
    if not isinstance(value, str):
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} must be text or null.")
    text = str(value).strip()
    if not text:
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} cannot be blank.")
    return text


def _integer(value: Any, *, field: str, candidate_id: str, positive: bool) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} must be an integer.")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real) and np.isfinite(float(value)) and float(value).is_integer():
        result = int(value)
    else:
        qualifier = "positive " if positive else "non-negative "
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} must be a {qualifier}integer.")
    minimum = 1 if positive else 0
    if result < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} must be a {qualifier} integer.")
    return result


def _optional_positive_integer(value: Any, *, field: str, candidate_id: str) -> int | None:
    if _is_missing_scalar(value):
        return None
    return _integer(value, field=field, candidate_id=candidate_id, positive=True)


def _finite_float(value: Any, *, field: str, candidate_id: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} must be finite numeric data.")
    result = float(value)
    if not np.isfinite(result):
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} must be finite numeric data.")
    return result


def _sequence(value: Any, *, field: str, candidate_id: str) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} must be a list.")


def _view_ids(
    value: Any,
    *,
    field: str,
    candidate_id: str,
    configured_view_ids: tuple[str, ...],
    allow_empty: bool,
) -> list[str]:
    raw_ids = _sequence(value, field=field, candidate_id=candidate_id)
    view_ids = [_required_text(item, field=field, candidate_id=candidate_id) for item in raw_ids]
    if not allow_empty and not view_ids:
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} cannot be empty.")
    if len(view_ids) != len(set(view_ids)):
        raise OpalError(f"Selection batch {field} for candidate {candidate_id!r} contains duplicates.")
    unknown = sorted(set(view_ids) - set(configured_view_ids))
    if unknown:
        raise OpalError(
            f"Selection batch candidate {candidate_id!r} has unknown {field}: {unknown}; "
            f"configured={list(configured_view_ids)}."
        )
    return view_ids


def _validate_memberships(
    value: Any,
    *,
    candidate_id: str,
    selection_view_ids: list[str],
    preferred_view_ids: list[str],
) -> list[dict[str, Any]]:
    raw_memberships = _sequence(value, field="selection_memberships", candidate_id=candidate_id)
    if not all(isinstance(raw, Mapping) for raw in raw_memberships):
        raise OpalError(f"Selection batch selection_memberships for candidate {candidate_id!r} must contain mappings.")
    membership_view_ids = [
        _required_text(
            raw["selection_view_id"] if "selection_view_id" in raw else None,
            field="membership selection_view_id",
            candidate_id=candidate_id,
        )
        for raw in raw_memberships
    ]
    if len(membership_view_ids) != len(set(membership_view_ids)):
        raise OpalError(f"Selection batch candidate {candidate_id!r} contains duplicate membership view IDs.")
    if set(membership_view_ids) != set(selection_view_ids):
        raise OpalError(
            f"Selection batch membership view IDs for candidate {candidate_id!r} must exactly match "
            f"selection_view_ids; memberships={membership_view_ids}, selection_view_ids={selection_view_ids}."
        )
    memberships: list[dict[str, Any]] = []
    for raw in raw_memberships:
        missing = sorted(SELECTION_MEMBERSHIP_REQUIRED_FIELDS - set(raw))
        if missing:
            raise OpalError(
                f"Selection batch membership for candidate {candidate_id!r} is missing required fields: {missing}."
            )
        membership = {str(key): _json_value(item) for key, item in raw.items()}
        view_id = _required_text(
            membership["selection_view_id"],
            field="membership selection_view_id",
            candidate_id=candidate_id,
        )
        rank = _integer(membership["rank"], field="rank", candidate_id=candidate_id, positive=True)
        rank_ordinal = _integer(
            membership["rank_ordinal"],
            field="rank_ordinal",
            candidate_id=candidate_id,
            positive=True,
        )
        if rank > rank_ordinal:
            raise OpalError(
                f"Selection batch membership for candidate {candidate_id!r} has competition rank {rank} "
                f"greater than ordinal rank {rank_ordinal}."
            )
        origin = _required_text(
            membership["selection_origin"],
            field="selection_origin",
            candidate_id=candidate_id,
        )
        if origin not in SELECTION_ORIGINS:
            raise OpalError(f"Selection batch candidate {candidate_id!r} has unsupported selection_origin {origin!r}.")
        if (view_id in preferred_view_ids) != (origin == "preferred_top_k"):
            raise OpalError(
                f"Selection batch candidate {candidate_id!r} has selection_origin {origin!r} inconsistent "
                f"with preferred_view_ids for view {view_id!r}."
            )
        membership.update(
            {
                "selection_view_id": view_id,
                "rank": rank,
                "rank_ordinal": rank_ordinal,
                "score": _finite_float(membership["score"], field="score", candidate_id=candidate_id),
                "selection_score": _finite_float(
                    membership["selection_score"],
                    field="selection_score",
                    candidate_id=candidate_id,
                ),
                "score_ref": _required_text(
                    membership["score_ref"],
                    field="score_ref",
                    candidate_id=candidate_id,
                ),
                "allocation_slot": _optional_positive_integer(
                    membership["allocation_slot"],
                    field="membership allocation_slot",
                    candidate_id=candidate_id,
                ),
                "selection_origin": origin,
            }
        )
        memberships.append(membership)
    return memberships


def _validate_provenance(
    frame: pd.DataFrame,
    *,
    campaign_slug: str,
    run_id: str,
    as_of_round: int,
    deduplicate_by: str,
) -> None:
    expected_text = {
        "campaign_slug": campaign_slug,
        "run_id": run_id,
        "deduplicate_by": deduplicate_by,
    }
    for field, expected in expected_text.items():
        values = frame[field].map(lambda value: "" if _is_missing_scalar(value) else str(value).strip())
        if values.eq("").any() or not values.eq(expected).all():
            raise OpalError(
                f"Selection batch contains mixed or unexpected {field}; expected {expected!r}, "
                f"observed={sorted(values.unique().tolist())}."
            )
    rounds = pd.to_numeric(frame["as_of_round"], errors="coerce")
    if rounds.isna().any() or not np.isfinite(rounds.to_numpy(dtype=float)).all():
        raise OpalError("Selection batch as_of_round must contain finite integers.")
    if not (rounds.to_numpy(dtype=float) == np.floor(rounds.to_numpy(dtype=float))).all():
        raise OpalError("Selection batch as_of_round must contain finite integers.")
    if not (rounds.astype(int) == int(as_of_round)).all():
        raise OpalError(f"Selection batch contains mixed or unexpected as_of_round; expected {int(as_of_round)}.")


def validate_selection_batch_rows(
    frame: pd.DataFrame,
    *,
    campaign_slug: str,
    run_id: str,
    as_of_round: int,
    configured_view_ids: Sequence[str],
    deduplicate_by: str,
    allocation_strategy: str | None,
    allocation_view_priority: Sequence[str],
    quota_by_view: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Validate and order persisted selection-batch rows for public inspection."""

    missing = sorted(SELECTION_BATCH_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise OpalError(f"Selection batch is missing required columns: {missing}.")
    if frame.empty:
        raise OpalError("Selection batch cannot be empty.")
    view_ids = tuple(str(value).strip() for value in configured_view_ids)
    if not view_ids or any(not value for value in view_ids) or len(view_ids) != len(set(view_ids)):
        raise OpalError("Configured selection view IDs must be unique and non-empty.")
    _validate_provenance(
        frame,
        campaign_slug=campaign_slug,
        run_id=run_id,
        as_of_round=as_of_round,
        deduplicate_by=deduplicate_by,
    )
    candidate_ids = frame["id"].map(lambda value: "" if _is_missing_scalar(value) else str(value).strip())
    if candidate_ids.eq("").any():
        raise OpalError("Selection batch contains null or blank candidate IDs.")
    if candidate_ids.duplicated().any():
        duplicates = sorted(candidate_ids.loc[candidate_ids.duplicated(keep=False)].unique().tolist())
        raise OpalError(f"Selection batch contains duplicate candidate IDs: {duplicates[:10]}.")
    keys = frame["selection_batch_key"].map(lambda value: "" if _is_missing_scalar(value) else str(value).strip())
    if keys.eq("").any():
        raise OpalError("Selection batch contains blank selection_batch_key values.")
    if keys.duplicated().any():
        raise OpalError("Selection batch contains duplicate selection_batch_key values.")

    normalized: list[dict[str, Any]] = []
    seen_allocation_slots: set[tuple[str, int]] = set()
    for raw_row in frame.to_dict(orient="records"):
        row = {str(key): _json_value(value) for key, value in raw_row.items()}
        candidate_id = _required_text(row["id"], field="id")
        selected_views = _view_ids(
            row["selection_view_ids"],
            field="selection_view_ids",
            candidate_id=candidate_id,
            configured_view_ids=view_ids,
            allow_empty=False,
        )
        preferred_views = _view_ids(
            row["preferred_view_ids"],
            field="preferred_view_ids",
            candidate_id=candidate_id,
            configured_view_ids=view_ids,
            allow_empty=True,
        )
        memberships = _validate_memberships(
            row["selection_memberships"],
            candidate_id=candidate_id,
            selection_view_ids=selected_views,
            preferred_view_ids=preferred_views,
        )
        allocation_view_id = _optional_text(
            row["allocation_view_id"],
            field="allocation_view_id",
            candidate_id=candidate_id,
        )
        allocation_slot = _optional_positive_integer(
            row["allocation_slot"],
            field="allocation_slot",
            candidate_id=candidate_id,
        )
        selection_batch_key = str(row["selection_batch_key"]).strip()
        if deduplicate_by == "id" and selection_batch_key != candidate_id:
            raise OpalError(
                f"Selection batch key for candidate {candidate_id!r} must equal the candidate id "
                "when deduplicate_by='id'."
            )
        if allocation_strategy is None:
            if allocation_view_id is not None or allocation_slot is not None:
                raise OpalError(
                    f"Selection batch logical-union rows cannot declare allocation for candidate {candidate_id!r}."
                )
            if any(membership["allocation_slot"] is not None for membership in memberships):
                raise OpalError(
                    "Selection batch logical-union memberships cannot declare allocation "
                    f"for candidate {candidate_id!r}."
                )
            if set(preferred_views) != set(selected_views):
                raise OpalError(
                    f"Selection batch logical-union candidate {candidate_id!r} must have preferred_view_ids "
                    "equal to selection_view_ids."
                )
        else:
            if allocation_view_id is None or allocation_slot is None:
                raise OpalError(
                    f"Selection batch allocated candidate {candidate_id!r} requires allocation_view_id "
                    "and allocation_slot."
                )
            if selected_views != [allocation_view_id] or len(memberships) != 1:
                raise OpalError(
                    f"Selection batch allocated candidate {candidate_id!r} must have exactly one membership "
                    "owned by allocation_view_id."
                )
            if memberships[0]["allocation_slot"] != allocation_slot:
                raise OpalError(
                    f"Selection batch allocated candidate {candidate_id!r} has inconsistent allocation slots."
                )
            slot_key = (allocation_view_id, allocation_slot)
            if slot_key in seen_allocation_slots:
                raise OpalError(
                    f"Selection batch contains duplicate allocation slot {allocation_slot} "
                    f"for view {allocation_view_id!r}."
                )
            seen_allocation_slots.add(slot_key)
        row.update(
            {
                "run_id": run_id,
                "as_of_round": int(as_of_round),
                "campaign_slug": campaign_slug,
                "id": candidate_id,
                "selection_batch_key": selection_batch_key,
                "deduplicate_by": deduplicate_by,
                "selection_view_ids": selected_views,
                "selection_memberships": memberships,
                "preferred_view_ids": preferred_views,
                "allocation_view_id": allocation_view_id,
                "allocation_slot": allocation_slot,
            }
        )
        normalized.append(row)

    if allocation_strategy is None:
        normalized.sort(
            key=lambda row: (
                min(int(membership["rank"]) for membership in row["selection_memberships"]),
                str(row["id"]),
            )
        )
    else:
        priority = tuple(str(value).strip() for value in allocation_view_priority)
        if (
            len(priority) != len(set(priority))
            or set(priority) != set(view_ids)
            or any(not value for value in priority)
        ):
            raise OpalError(
                "Selection batch allocation view priority must be an exact permutation of configured selection views."
            )
        priority_index = {view_id: index for index, view_id in enumerate(priority)}
        for view_id in priority:
            expected_slots = set(range(1, int(quota_by_view[view_id]) + 1))
            observed_slots = {slot for owner, slot in seen_allocation_slots if owner == view_id}
            if observed_slots != expected_slots:
                raise OpalError(
                    f"Selection batch allocation slots for view {view_id!r} must equal "
                    f"{sorted(expected_slots)}; observed={sorted(observed_slots)}."
                )
        normalized.sort(
            key=lambda row: (
                int(row["allocation_slot"]),
                priority_index[str(row["allocation_view_id"])],
                str(row["id"]),
            )
        )
    return normalized


__all__ = ["SELECTION_BATCH_REQUIRED_COLUMNS", "validate_selection_batch_rows"]
