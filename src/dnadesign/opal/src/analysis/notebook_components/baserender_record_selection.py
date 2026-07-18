"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_record_selection.py

Exact selected-record resolution for BaseRender notebook evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._support import display_name
from .baserender_records import load_notebook_baserender_record_row


def build_notebook_selected_baserender_records(
    campaign_analysis: Any,
    *,
    selection_view_id: str,
    round_value: Any | None,
    run_id: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return selected record identities with their exact selection-view ranks."""

    selected_records: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    if round_value is None:
        return selected_records, [{"field": "selection scope", "value": "no rounds available"}]
    run_text = str(run_id or "").strip()
    if not run_text:
        return selected_records, [{"field": "selection scope", "value": "no run available"}]
    view_text = str(selection_view_id or "").strip()
    if not view_text:
        return selected_records, [{"field": "selection scope", "value": "no selection view available"}]

    try:
        import polars as pl

        round_int = int(round_value)
        pred_df = campaign_analysis.read_selection_view_predictions(
            selection_view_id=selection_view_id,
            columns=[
                "id",
                "as_of_round",
                "run_id",
                "view__rank_competition",
                "view__is_selected",
            ],
            round_selector=[round_int],
            run_id=run_text,
        )
        selected_df = (
            pred_df.filter(pl.col("view__is_selected").fill_null(False)) if not pred_df.is_empty() else pred_df
        )
        sort_columns = [column for column in ("view__rank_competition", "id") if column in selected_df.columns]
        if sort_columns:
            selected_df = selected_df.sort(sort_columns)
        required = {"id", "view__rank_competition"}
        missing = sorted(required - set(selected_df.columns))
        if missing:
            raise ValueError(f"Selected-sequence evidence is missing columns: {', '.join(missing)}.")
        validated_records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for row in selected_df.select("id", "view__rank_competition").to_dicts():
            record_id = str(row.get("id") or "").strip()
            if not record_id:
                raise ValueError("Selected-sequence evidence contains a blank record id.")
            if record_id in seen_ids:
                raise ValueError(f"Selected-sequence evidence contains duplicate record id {record_id!r}.")
            rank_value = row.get("view__rank_competition")
            try:
                rank = int(rank_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Selected record {record_id!r} has no integer competition rank.") from exc
            if rank <= 0 or float(rank_value) != float(rank):
                raise ValueError(f"Selected record {record_id!r} has invalid competition rank {rank_value!r}.")
            seen_ids.add(record_id)
            validated_records.append(
                {
                    "record_id": record_id,
                    "selection_view_id": view_text,
                    "view_rank": rank,
                }
            )
        selected_records = validated_records
        status_rows.extend(
            [
                {"field": "selection view", "value": _selection_view_label(view_text)},
                {"field": "selection round", "value": round_int},
                {"field": "selection run", "value": run_text},
                {"field": "selected records", "value": len(selected_records)},
            ]
        )
    except Exception as exc:
        selected_records = []
        status_rows.append({"field": "selection ledger", "value": f"unavailable: {exc}"})
    return selected_records, status_rows


def build_notebook_selected_baserender_record_sets(
    selection_batch: Mapping[str, Any] | None,
    *,
    campaign_slug: Any,
    selection_view_ids: Sequence[Any],
    round_value: Any | None,
    run_id: Any | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Project selected records from one verified selection-batch payload."""

    view_ids = [str(value or "").strip() for value in selection_view_ids]
    if any(not value for value in view_ids):
        raise ValueError("BaseRender selection-view ids must be non-empty.")
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("BaseRender selection-view ids must be unique.")

    records_by_view: dict[str, list[dict[str, Any]]] = {view_id: [] for view_id in view_ids}
    if round_value is None:
        return records_by_view, _scope_status(view_ids, "no rounds available")
    run_text = str(run_id or "").strip()
    if not run_text:
        return records_by_view, _scope_status(view_ids, "no run available")
    slug = str(campaign_slug or "").strip()
    if not slug:
        raise ValueError("BaseRender selection-batch scope requires a campaign slug.")
    if not isinstance(selection_batch, Mapping) or not selection_batch:
        return records_by_view, _scope_status(view_ids, "unavailable: no verified selection batch available")

    _validate_selection_batch_scope(
        selection_batch,
        campaign_slug=slug,
        run_id=run_text,
        round_value=round_value,
    )
    rows = selection_batch.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Verified selection batch requires a rows list.")
    seen_by_view = {view_id: set() for view_id in view_ids}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("Verified selection-batch rows must be objects.")
        record_id = str(raw_row.get("id") or "").strip()
        if not record_id:
            raise ValueError("Verified selection-batch rows require non-empty record ids.")
        _validate_selection_batch_row_scope(raw_row, campaign_slug=slug, run_id=run_text, round_value=round_value)
        memberships = raw_row.get("selection_memberships")
        if not isinstance(memberships, Sequence) or isinstance(memberships, (str, bytes)):
            raise ValueError(f"Selection-batch record {record_id!r} requires selection_memberships.")
        raw_declared_views = raw_row.get("selection_view_ids")
        if not isinstance(raw_declared_views, Sequence) or isinstance(raw_declared_views, (str, bytes)):
            raise ValueError(f"Selection-batch record {record_id!r} requires selection_view_ids.")
        declared_views = {str(value or "").strip() for value in raw_declared_views}
        if not declared_views or "" in declared_views:
            raise ValueError(f"Selection-batch record {record_id!r} has invalid selection_view_ids.")
        for raw_membership in memberships:
            if not isinstance(raw_membership, Mapping):
                raise ValueError(f"Selection-batch record {record_id!r} has a non-object membership.")
            view_id = str(raw_membership.get("selection_view_id") or "").strip()
            if not view_id or view_id not in declared_views:
                raise ValueError(f"Selection-batch record {record_id!r} has inconsistent view membership.")
            if view_id not in records_by_view:
                continue
            if record_id in seen_by_view[view_id]:
                raise ValueError(f"Selection-batch view {view_id!r} contains duplicate record {record_id!r}.")
            rank = _positive_integer(raw_membership.get("rank"), field="rank", record_id=record_id)
            seen_by_view[view_id].add(record_id)
            records_by_view[view_id].append({"record_id": record_id, "selection_view_id": view_id, "view_rank": rank})

    status_by_view: dict[str, list[dict[str, Any]]] = {}
    for view_id in view_ids:
        records_by_view[view_id].sort(key=lambda row: (int(row["view_rank"]), str(row["record_id"])))
        status_by_view[view_id] = [
            {"field": "selection view", "value": _selection_view_label(view_id)},
            {"field": "selection round", "value": int(round_value)},
            {"field": "selection run", "value": run_text},
            {"field": "selected records", "value": len(records_by_view[view_id])},
            {"field": "selection source", "value": "verified selection batch"},
        ]
    return records_by_view, status_by_view


def resolve_notebook_baserender_selection_batch_scope(
    selection_batch: Mapping[str, Any] | None,
) -> tuple[int | None, str | None]:
    """Resolve the one run and round bound to a campaign's verified selection batch."""

    if not isinstance(selection_batch, Mapping) or not selection_batch:
        return None, None
    round_value = selection_batch.get("as_of_round")
    run_id = str(selection_batch.get("run_id") or "").strip()
    if (round_value is None) != (not run_id):
        raise ValueError("BaseRender selection-batch scope requires both round and run id.")
    if round_value is None:
        return None, None
    return _nonnegative_integer(round_value, field="as_of_round"), run_id


def _scope_status(view_ids: Sequence[str], message: str) -> dict[str, list[dict[str, Any]]]:
    return {view_id: [{"field": "selection scope", "value": message}] for view_id in view_ids}


def _validate_selection_batch_scope(
    selection_batch: Mapping[str, Any],
    *,
    campaign_slug: str,
    run_id: str,
    round_value: Any,
) -> None:
    if selection_batch.get("schema_version") != "opal.selection_batch.v3":
        raise ValueError("BaseRender requires verified opal.selection_batch.v3 evidence.")
    verification = selection_batch.get("verification")
    if not isinstance(verification, Mapping) or verification.get("status") != "pass":
        raise ValueError("BaseRender selection-batch verification must pass.")
    campaign = selection_batch.get("campaign")
    batch_slug = str(campaign.get("slug") or "").strip() if isinstance(campaign, Mapping) else ""
    if batch_slug != campaign_slug:
        raise ValueError(f"Selection-batch campaign {batch_slug!r} does not match {campaign_slug!r}.")
    if str(selection_batch.get("run_id") or "").strip() != run_id:
        raise ValueError("Selection-batch run id does not match the selected run.")
    if _nonnegative_integer(selection_batch.get("as_of_round"), field="as_of_round") != _nonnegative_integer(
        round_value,
        field="round_value",
    ):
        raise ValueError("Selection-batch round does not match the selected round.")


def _validate_selection_batch_row_scope(
    row: Mapping[str, Any],
    *,
    campaign_slug: str,
    run_id: str,
    round_value: Any,
) -> None:
    if str(row.get("campaign_slug") or "").strip() != campaign_slug:
        raise ValueError("Selection-batch row campaign does not match the selected campaign.")
    if str(row.get("run_id") or "").strip() != run_id:
        raise ValueError("Selection-batch row run does not match the selected run.")
    if _nonnegative_integer(row.get("as_of_round"), field="row as_of_round") != _nonnegative_integer(
        round_value,
        field="round_value",
    ):
        raise ValueError("Selection-batch row round does not match the selected round.")


def _positive_integer(value: Any, *, field: str, record_id: str) -> int:
    parsed = _nonnegative_integer(value, field=field)
    if parsed <= 0:
        raise ValueError(f"Selection-batch record {record_id!r} has invalid {field} {value!r}.")
    return parsed


def _nonnegative_integer(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Selection-batch {field} must be an integer.") from exc
    if isinstance(value, bool) or parsed < 0 or float(value) != float(parsed):
        raise ValueError(f"Selection-batch {field} has invalid value {value!r}.")
    return parsed


def _selection_view_label(value: str) -> str:
    return "AND" if value.lower() == "and" else display_name(value)


def resolve_notebook_baserender_record_selection(
    records_path: str | Path,
    selector_value: Any | None,
    selected_records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Bind one selector value to exactly one ledger row and one render record."""

    if selector_value is None:
        return None, None, None
    record_id = str(selector_value).strip()
    matches = [dict(row) for row in selected_records if str(row.get("record_id") or "").strip() == record_id]
    if len(matches) != 1:
        raise ValueError(
            f"Selected sequence {record_id!r} must resolve to exactly one selection record; found {len(matches)}."
        )
    record_row = load_notebook_baserender_record_row(records_path, record_id, contract)
    return record_id, record_row, matches[0]


__all__ = [
    "build_notebook_selected_baserender_record_sets",
    "build_notebook_selected_baserender_records",
    "resolve_notebook_baserender_selection_batch_scope",
    "resolve_notebook_baserender_record_selection",
]
