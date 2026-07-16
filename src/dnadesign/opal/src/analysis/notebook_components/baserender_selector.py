"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_selector.py

Presentation model for the selected-sequence BaseRender control.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .baserender import build_notebook_baserender_contract_rows
from .baserender_diagnostics import render_notebook_baserender_diagnostic_panel
from .baserender_records import (
    build_notebook_baserender_record_annotation_counts,
    build_notebook_baserender_record_choices_with_counts,
    build_notebook_baserender_record_options,
    has_notebook_baserender_record_options,
    select_notebook_baserender_default_record_id,
)


def build_notebook_baserender_selector_model(
    records_path: str | Path,
    contract: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a selector model only when selected records satisfy the public adapter."""

    selected_ids = [str(row["record_id"]) for row in selected_records]
    options = build_notebook_baserender_record_options(
        records_path,
        contract,
        record_ids=selected_ids,
    )
    has_renderable_records = has_notebook_baserender_record_options(options)
    if not has_renderable_records:
        return {
            "has_renderable_records": False,
            "record_options": options,
            "annotation_counts": {},
            "record_choices": {},
            "default_label": None,
        }

    counts = build_notebook_baserender_record_annotation_counts(
        records_path,
        contract,
        record_ids=options,
    )
    choice_rows = build_notebook_baserender_record_choices_with_counts(
        options,
        counts,
        annotation_label="annotated elements",
        view_ranks={str(row["record_id"]): int(row["view_rank"]) for row in selected_records},
    )
    choices = {str(choice["label"]): str(choice["record_id"]) for choice in choice_rows}
    default_record_id = select_notebook_baserender_default_record_id(options, counts)
    default_labels = [
        str(choice["label"]) for choice in choice_rows if str(choice["record_id"]) == str(default_record_id)
    ]
    if len(default_labels) != 1:
        raise ValueError(
            f"Default BaseRender record {default_record_id!r} must resolve to exactly one selector label; "
            f"found {len(default_labels)}."
        )
    return {
        "has_renderable_records": True,
        "record_options": options,
        "annotation_counts": counts,
        "record_choices": choices,
        "default_label": default_labels[0],
    }


def render_notebook_baserender_selector(selector_model: Mapping[str, Any], *, mo: Any) -> Any | None:
    """Render the selected-sequence control only when its public adapter is usable."""

    if not bool(selector_model.get("has_renderable_records")):
        return None
    choices = dict(selector_model.get("record_choices") or {})
    default_label = str(selector_model.get("default_label") or "").strip()
    if not choices or default_label not in choices:
        raise ValueError("Renderable BaseRender selector state requires one valid default choice.")
    return mo.ui.dropdown(
        choices,
        value=default_label,
        label="Selected sequence",
        searchable=True,
        full_width=True,
    )


def build_notebook_baserender_review_state(
    records_path: str | Path,
    contract: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
    status_rows: Sequence[Mapping[str, Any]],
    *,
    mo: Any,
    opal_table: Any,
    pl: Any,
) -> tuple[bool, Any | None, Any | None]:
    """Build the selected-sequence control and its unavailable-evidence fallback."""

    selector_model = build_notebook_baserender_selector_model(records_path, contract, selected_records)
    has_renderable_records = bool(selector_model["has_renderable_records"])
    selector = render_notebook_baserender_selector(selector_model, mo=mo)
    diagnostic_panel = render_notebook_baserender_diagnostic_panel(
        has_renderable_records=has_renderable_records,
        selected_record_ids=[str(row["record_id"]) for row in selected_records],
        status_rows=status_rows,
        contract=contract,
        build_contract_rows=build_notebook_baserender_contract_rows,
        mo=mo,
        opal_table=opal_table,
        pl=pl,
    )
    return has_renderable_records, selector, diagnostic_panel


__all__ = [
    "build_notebook_baserender_review_state",
    "build_notebook_baserender_selector_model",
    "render_notebook_baserender_selector",
]
