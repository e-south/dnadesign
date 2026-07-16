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


__all__ = ["build_notebook_selected_baserender_records", "resolve_notebook_baserender_record_selection"]
