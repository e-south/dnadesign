"""Notebook presentation contract for a campaign selection batch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SELECTION_BATCH_SURFACE_KIND = "selection_batch"


def build_notebook_selection_batch_choice(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build one campaign-scoped deliverable for the logical selection union."""

    if not payload:
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Selection batch payload requires a rows list.")
    unique_count = int(payload.get("unique_count") or 0)
    if unique_count != len(rows):
        raise ValueError(f"Selection batch unique_count={unique_count} does not match rows={len(rows)}.")
    return {
        "label": "Selection batch",
        "title": "The selection batch is the deduplicated union of named selection views",
        "surface_kind": SELECTION_BATCH_SURFACE_KIND,
        "review_group": "handoff",
        "review_rank": 0,
        "as_of_round": payload.get("as_of_round"),
        "run_id": payload.get("run_id"),
        "unique_count": unique_count,
        "rows": rows,
    }


def build_notebook_selection_batch_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact candidate and view-membership rows for notebook review."""

    rows: list[dict[str, Any]] = []
    for raw in choice.get("rows") or []:
        if not isinstance(raw, Mapping):
            raise ValueError("Selection batch rows must be mappings.")
        memberships = raw.get("selection_memberships") or []
        if not isinstance(memberships, list):
            raise ValueError("selection_memberships must be a list.")
        view_ids = raw.get("selection_view_ids") or []
        if not isinstance(view_ids, list):
            raise ValueError("selection_view_ids must be a list.")
        rows.append(
            {
                "candidate": str(raw.get("id") or ""),
                "selection views": ", ".join(_view_label(value) for value in view_ids),
                "view count": len(view_ids),
                "view ranks": "; ".join(
                    f"{_view_label(item.get('selection_view_id'))} {int(item['rank'])}"
                    for item in memberships
                    if isinstance(item, Mapping) and item.get("rank") is not None
                ),
            }
        )
    return rows


def build_notebook_selection_batch_summary_rows(choice: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return bounded provenance rows for the selected batch."""

    return [
        {"field": "round", "value": choice.get("as_of_round")},
        {"field": "run", "value": choice.get("run_id") or "not recorded"},
        {"field": "unique candidates", "value": int(choice.get("unique_count") or 0)},
        {
            "field": "scope",
            "value": "Logical union only; physical synthesis authorization remains study-owned.",
        },
    ]


def _view_label(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() == "and":
        return "AND"
    return text.replace("_", " ").title()


__all__ = [
    "SELECTION_BATCH_SURFACE_KIND",
    "build_notebook_selection_batch_choice",
    "build_notebook_selection_batch_rows",
    "build_notebook_selection_batch_summary_rows",
]
