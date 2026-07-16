"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/selection_batch.py

Notebook presentation contract for a campaign selection batch.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._support import compact_identifier

SELECTION_BATCH_SURFACE_KIND = "selection_batch"
_SELECTION_BATCH_SCHEMA_VERSION = "opal.selection_batch.v3"
_ALLOCATION_STRATEGIES = {"logical_union", "round_robin_next_best_unallocated"}


def build_notebook_selection_batch_choice(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build one campaign-scoped deliverable for the logical selection union."""

    if not payload:
        return None
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != _SELECTION_BATCH_SCHEMA_VERSION:
        raise ValueError(
            f"Selection batch payload requires schema {_SELECTION_BATCH_SCHEMA_VERSION!r}; found {schema_version!r}."
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Selection batch payload requires a rows list.")
    unique_count = int(payload.get("unique_count") or 0)
    if unique_count != len(rows):
        raise ValueError(f"Selection batch unique_count={unique_count} does not match rows={len(rows)}.")
    allocation_strategy = str(payload.get("allocation_strategy") or "").strip()
    if allocation_strategy not in _ALLOCATION_STRATEGIES:
        raise ValueError(f"Selection batch payload has unsupported allocation_strategy {allocation_strategy!r}.")
    deduplicate_by = str(payload.get("deduplicate_by") or "").strip()
    if not deduplicate_by:
        raise ValueError("Selection batch payload requires non-empty deduplicate_by provenance.")
    return {
        "label": "Selection batch proposal",
        "title": "Deduplicated selection batch proposal",
        "surface_kind": SELECTION_BATCH_SURFACE_KIND,
        "review_group": "handoff",
        "review_rank": 0,
        "as_of_round": payload.get("as_of_round"),
        "run_id": payload.get("run_id"),
        "deduplicate_by": deduplicate_by,
        "allocation_strategy": allocation_strategy,
        "unique_count": unique_count,
        "rows": rows,
    }


def build_notebook_selection_batch_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact allocation rows with preference and rank provenance."""

    rows: list[dict[str, Any]] = []
    for raw in choice.get("rows") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("Selection batch rows must be mappings.")
        candidate_id = str(raw.get("id") or "").strip()
        if not candidate_id:
            raise ValueError("Selection batch rows require non-empty candidate ids.")
        memberships = raw.get("selection_memberships")
        if (
            not isinstance(memberships, list)
            or not memberships
            or not all(isinstance(item, Mapping) for item in memberships)
        ):
            raise ValueError(f"Selection batch candidate {candidate_id!r} has no displayable memberships.")
        preferred_view_ids = raw.get("preferred_view_ids")
        if not isinstance(preferred_view_ids, list):
            raise ValueError(f"Selection batch candidate {candidate_id!r} has no preferred-view list.")
        allocation_view_id = str(raw.get("allocation_view_id") or "").strip()
        allocation_slot_value = raw.get("allocation_slot")
        if allocation_view_id:
            allocation_slot = _positive_integer(
                allocation_slot_value,
                field="allocation_slot",
                candidate_id=candidate_id,
            )
            allocated_memberships = [
                item for item in memberships if str(item["selection_view_id"]) == allocation_view_id
            ]
            if len(allocated_memberships) != 1:
                raise ValueError(
                    f"Allocated candidate {candidate_id!r} must have exactly one membership for "
                    f"view {allocation_view_id!r}; found {len(allocated_memberships)}."
                )
            membership = allocated_memberships[0]
            rank: int | str = _positive_integer(
                membership.get("rank"),
                field=f"{allocation_view_id}.rank",
                candidate_id=candidate_id,
            )
            origin = _origin_label(membership["selection_origin"])
            slot: int | str = allocation_slot
            allocated_view = _view_label(allocation_view_id)
        else:
            rank_rows = [
                f"{_view_label(item['selection_view_id'])} "
                f"{_positive_integer(item.get('rank'), field='rank', candidate_id=candidate_id)}"
                for item in memberships
            ]
            rank = "; ".join(rank_rows) or "not recorded"
            origins = list(dict.fromkeys(_origin_label(item["selection_origin"]) for item in memberships))
            origin = ", ".join(origins) or "not recorded"
            slot = "not allocated"
            allocated_view = "Logical union"
        minimum_rank = min(int(item["rank"]) for item in memberships)
        rows.append(
            {
                "candidate": compact_identifier(candidate_id),
                "allocated view": allocated_view,
                "competition rank": rank,
                "preferred by": ", ".join(_view_label(value) for value in preferred_view_ids) or "None",
                "allocation origin": origin,
                "view slot": slot,
                "__sort_slot": int(allocation_slot_value) if allocation_slot_value is not None else 10**9,
                "__sort_rank": minimum_rank,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["__sort_slot"]),
            int(row["__sort_rank"]),
            str(row["allocated view"]),
            str(row["candidate"]),
        )
    )
    for row in rows:
        row.pop("__sort_slot")
        row.pop("__sort_rank")
    return rows


def build_notebook_selection_batch_summary_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return bounded provenance rows for the selected batch."""

    allocation_strategy = str(choice.get("allocation_strategy") or "").strip()
    allocated = allocation_strategy != "logical_union"
    return [
        {"field": "round", "value": choice.get("as_of_round")},
        {"field": "run", "value": choice.get("run_id") or "not recorded"},
        {"field": "unique candidates", "value": int(choice.get("unique_count") or 0)},
        {
            "field": "row order",
            "value": (
                "Allocation slot, then competition rank and view."
                if allocated
                else "Competition rank, then view and candidate."
            ),
        },
        {
            "field": "batch formation",
            "value": "Coordinated unique-slot allocation" if allocated else "Logical union of view selections",
        },
        {
            "field": "deduplicated by",
            "value": str(choice.get("deduplicate_by") or "not recorded"),
        },
        {
            "field": "candidate labels",
            "value": "Compact display only; exact ids remain in the selection-batch artifact.",
        },
        {
            "field": "scope",
            "value": "Physical batch proposal only; synthesis authorization remains study-owned.",
        },
    ]


def _view_label(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() == "and":
        return "AND"
    return text.replace("_", " ").title()


def _origin_label(value: Any) -> str:
    text = str(value or "").strip()
    labels = {
        "preferred_top_k": "Preferred top k",
        "next_best_unallocated": "Next best unallocated",
    }
    return labels.get(text, text.replace("_", " ").title() or "not recorded")


def _positive_integer(value: Any, *, field: str, candidate_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Selection batch candidate {candidate_id!r} requires {field} as a positive integer; found {value!r}."
        )
    return value


__all__ = [
    "SELECTION_BATCH_SURFACE_KIND",
    "build_notebook_selection_batch_choice",
    "build_notebook_selection_batch_rows",
    "build_notebook_selection_batch_summary_rows",
]
