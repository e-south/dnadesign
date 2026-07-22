"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_candidate_catalog.py

Campaign-scoped candidate evidence for BaseRender notebook lookup.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .baserender_record_selection import build_notebook_selected_baserender_record_sets


def build_notebook_baserender_candidate_catalog(
    selection_batch: Mapping[str, Any] | None,
    labels_df: Any | None,
    *,
    campaign_slug: Any,
    selection_view_ids: Sequence[Any],
    round_value: Any | None,
    run_id: Any | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Union observed labels and the verified active selection batch for lookup."""

    view_ids = _view_ids(selection_view_ids)
    selected_by_view, selection_status = build_notebook_selected_baserender_record_sets(
        selection_batch,
        campaign_slug=campaign_slug,
        selection_view_ids=view_ids,
        round_value=round_value,
        run_id=run_id,
    )
    observed = _observed_evidence(labels_df)
    memberships = _selection_memberships(selected_by_view, selected_round=round_value)
    all_ids = set(observed) | set(memberships)

    catalogs: dict[str, list[dict[str, Any]]] = {}
    statuses: dict[str, list[dict[str, Any]]] = {}
    for view_id in view_ids:
        active_ids = [str(row["record_id"]) for row in selected_by_view[view_id]]
        other_selected = sorted(set(memberships) - set(active_ids))
        observed_only = sorted(set(observed) - set(memberships))
        ordered_ids = [*active_ids, *other_selected, *observed_only]
        if set(ordered_ids) != all_ids:
            raise ValueError("BaseRender candidate catalog failed to preserve the complete campaign evidence union.")
        rows = [
            _candidate_row(
                record_id=record_id,
                active_view_id=view_id,
                memberships=memberships.get(record_id, []),
                observed=observed.get(record_id),
            )
            for record_id in ordered_ids
        ]
        catalogs[view_id] = rows
        statuses[view_id] = [
            *selection_status[view_id],
            {"field": "observed records", "value": len(observed)},
            {"field": "candidate records", "value": len(rows)},
            {
                "field": "candidate scope",
                "value": "verified labels used plus active verified selection batch",
            },
        ]
    return catalogs, statuses


def _view_ids(values: Sequence[Any]) -> list[str]:
    view_ids = [str(value or "").strip() for value in values]
    if any(not value for value in view_ids):
        raise ValueError("BaseRender candidate-catalog selection-view ids must be non-empty.")
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("BaseRender candidate-catalog selection-view ids must be unique.")
    return view_ids


def _observed_evidence(labels_df: Any | None) -> dict[str, dict[str, Any]]:
    if labels_df is None or not hasattr(labels_df, "columns") or not hasattr(labels_df, "is_empty"):
        return {}
    if labels_df.is_empty():
        return {}
    required = {"id", "observed_round"}
    missing = sorted(required - set(labels_df.columns))
    if missing:
        raise ValueError(f"BaseRender observed candidate evidence is missing columns: {', '.join(missing)}.")
    columns = [column for column in ("id", "observed_round", "src") if column in labels_df.columns]
    evidence: dict[str, dict[str, Any]] = {}
    for raw in labels_df.select(columns).to_dicts():
        record_id = str(raw.get("id") or "").strip()
        if not record_id:
            raise ValueError("BaseRender observed candidate evidence contains a blank record id.")
        observed_round = _nonnegative_integer(raw.get("observed_round"), field="observed_round")
        row = evidence.setdefault(record_id, {"observed_rounds": set(), "observed_sources": set()})
        row["observed_rounds"].add(observed_round)
        source = str(raw.get("src") or "").strip()
        if source:
            row["observed_sources"].add(source)
    return {
        record_id: {
            "observed_rounds": sorted(row["observed_rounds"]),
            "observed_sources": sorted(row["observed_sources"]),
        }
        for record_id, row in evidence.items()
    }


def _selection_memberships(
    selected_by_view: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    selected_round: Any | None,
) -> dict[str, list[dict[str, Any]]]:
    memberships: dict[str, list[dict[str, Any]]] = {}
    for view_id, rows in selected_by_view.items():
        for row in rows:
            record_id = str(row.get("record_id") or "").strip()
            membership = {
                "selection_view_id": str(view_id),
                "view_rank": int(row["view_rank"]),
                "selected_round": None if selected_round is None else int(selected_round),
            }
            memberships.setdefault(record_id, []).append(membership)
    for rows in memberships.values():
        rows.sort(key=lambda row: (str(row["selection_view_id"]), int(row["view_rank"])))
    return memberships


def _candidate_row(
    *,
    record_id: str,
    active_view_id: str,
    memberships: Sequence[Mapping[str, Any]],
    observed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    membership_rows = [dict(row) for row in memberships]
    active = [row for row in membership_rows if str(row["selection_view_id"]) == active_view_id]
    if len(active) > 1:
        raise ValueError(f"BaseRender candidate {record_id!r} has duplicate active-view memberships.")
    observed_rounds = list((observed or {}).get("observed_rounds") or [])
    evidence_roles = [
        role for role, present in (("observed", observed_rounds), ("selected", membership_rows)) if present
    ]
    return {
        "record_id": record_id,
        "active_selection_view_id": active_view_id,
        "active_view_rank": int(active[0]["view_rank"]) if active else None,
        "selection_memberships": membership_rows,
        "observed_rounds": observed_rounds,
        "observed_sources": list((observed or {}).get("observed_sources") or []),
        "evidence_roles": evidence_roles,
    }


def _nonnegative_integer(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BaseRender candidate {field} must be an integer.") from exc
    if isinstance(value, bool) or parsed < 0 or float(value) != float(parsed):
        raise ValueError(f"BaseRender candidate {field} has invalid value {value!r}.")
    return parsed


__all__ = ["build_notebook_baserender_candidate_catalog"]
