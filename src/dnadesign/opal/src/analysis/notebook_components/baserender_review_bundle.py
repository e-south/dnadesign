"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_review_bundle.py

Immutable evidence and scoped controls for BaseRender notebook review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .baserender import build_notebook_baserender_contract_rows
from .baserender_candidate_catalog import build_notebook_baserender_candidate_catalog
from .baserender_diagnostics import render_notebook_baserender_diagnostic_panel
from .baserender_record_memory import build_notebook_baserender_record_memory_key
from .baserender_selector import build_notebook_baserender_selector_model, render_notebook_baserender_selector
from .selection_views import build_notebook_selection_view_options


def build_notebook_baserender_evidence_bundle(
    records_path: str | Path,
    contract: Mapping[str, Any],
    campaign_model: Mapping[str, Any],
    *,
    labels_df: Any,
    run_id: Any,
    round_value: Any,
    mo: Any,
    opal_table: Any,
    pl: Any,
) -> dict[str, dict[str, Any]]:
    """Build immutable campaign-candidate evidence for every selection view."""

    view_options = build_notebook_selection_view_options(campaign_model)
    view_ids = tuple(dict.fromkeys(str(value) for value in view_options.values()))
    campaign = campaign_model.get("campaign") or {}
    campaign_slug = str(campaign.get("slug") or "").strip()
    records_by_view, status_by_view = build_notebook_baserender_candidate_catalog(
        campaign_model.get("selection_batch"),
        labels_df,
        campaign_slug=campaign_slug,
        selection_view_ids=view_ids,
        round_value=round_value,
        run_id=run_id,
    )
    bundle: dict[str, dict[str, Any]] = {}
    for view_id in view_ids:
        records = records_by_view[view_id]
        status_rows = status_by_view[view_id]
        selector_model = build_notebook_baserender_selector_model(records_path, contract, records)
        has_renderable_records = bool(selector_model["has_renderable_records"])
        unrenderable_record_ids = list(selector_model["unrenderable_record_ids"])
        diagnostic_panel = render_notebook_baserender_diagnostic_panel(
            has_renderable_records=has_renderable_records,
            candidate_record_ids=[str(row["record_id"]) for row in records],
            unrenderable_record_ids=unrenderable_record_ids,
            status_rows=status_rows,
            contract=contract,
            build_contract_rows=build_notebook_baserender_contract_rows,
            mo=mo,
            opal_table=opal_table,
            pl=pl,
        )
        bundle[view_id] = {
            "records": records,
            "status_rows": status_rows,
            "selector_model": selector_model,
            "has_renderable_records": has_renderable_records,
            "has_candidate_records": bool(records),
            "unrenderable_record_ids": unrenderable_record_ids,
            "diagnostic_panel": diagnostic_panel,
        }
    return bundle


def build_notebook_baserender_record_controls(
    evidence_bundle: Mapping[str, Mapping[str, Any]],
    *,
    campaign_slug: Any,
    run_id: Any,
    round_value: Any,
    review_group_key: Any,
    deliverable_key: Any,
    memory: Any,
    set_memory: Any,
    mo: Any,
) -> dict[str, Any | None]:
    """Render one stable selector per view without reopening evidence sources."""

    remembered = dict(memory())
    has_memory_scope = round_value is not None and bool(str(run_id or "").strip())
    controls: dict[str, Any | None] = {}
    for view_id, evidence in evidence_bundle.items():
        key = (
            build_notebook_baserender_record_memory_key(
                campaign_slug=campaign_slug,
                run_id=run_id,
                round_value=round_value,
                selection_view_id=view_id,
                review_group_key=review_group_key,
                deliverable_key=deliverable_key,
            )
            if has_memory_scope
            else None
        )

        def remember_record(value: Any, *, _key: str | None = key) -> None:
            if _key is not None:
                set_memory({**dict(memory()), _key: str(value)})

        controls[view_id] = render_notebook_baserender_selector(
            evidence["selector_model"],
            preferred_record_id=remembered.get(key) if key is not None else None,
            on_change=remember_record if key is not None else None,
            mo=mo,
        )
    return controls


__all__ = [
    "build_notebook_baserender_evidence_bundle",
    "build_notebook_baserender_record_controls",
]
