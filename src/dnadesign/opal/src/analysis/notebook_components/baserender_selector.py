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

from .baserender_record_memory import resolve_notebook_baserender_preferred_record_id
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
            "selected_record_id": None,
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
        "selected_record_id": default_record_id,
    }


def render_notebook_baserender_selector(
    selector_model: Mapping[str, Any],
    *,
    mo: Any,
    preferred_record_id: Any | None = None,
    on_change: Any | None = None,
) -> Any | None:
    """Render the selected-sequence control only when its public adapter is usable."""

    if not bool(selector_model.get("has_renderable_records")):
        return None
    choices = dict(selector_model.get("record_choices") or {})
    selected_record_id = resolve_notebook_baserender_preferred_record_id(
        list(selector_model.get("record_options") or []),
        dict(selector_model.get("annotation_counts") or {}),
        preferred_record_id=preferred_record_id,
    )
    selected_labels = [label for label, record_id in choices.items() if str(record_id) == selected_record_id]
    if not choices or len(selected_labels) != 1:
        raise ValueError("Renderable BaseRender selector state requires one valid default choice.")
    return mo.ui.dropdown(
        choices,
        value=selected_labels[0],
        label="Selected sequence",
        searchable=True,
        full_width=True,
        on_change=on_change,
    )


__all__ = [
    "build_notebook_baserender_selector_model",
    "render_notebook_baserender_selector",
]
