"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/layered_scatter_controls.py

Build persistent controls for manifest-backed notebook scatter layers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping


def build_notebook_layered_scatter_controls(
    contract: Mapping[str, Any] | None,
    *,
    memory: Any,
    set_memory: Any,
    mo: Any,
) -> dict[str, Any]:
    """Build persistent widgets for one manifest-declared layered scatter."""

    if contract is None:
        return {key: None for key in ("prediction_pool", "selected", "observed_batches", "labels")}
    key = str(contract["key"])
    remembered = dict(dict(memory()).get(key) or {})

    def remember(field: str, value: object) -> None:
        current = dict(memory())
        current_view = dict(current.get(key) or {})
        set_memory({**current, key: {**current_view, field: value}})

    batch_options = {str(item["label"]): str(item["id"]) for item in contract["observed_batches"]}
    known_batch_ids = set(batch_options.values())
    remembered_ids = [
        str(value)
        for value in remembered.get("observed_batches", sorted(known_batch_ids))
        if str(value) in known_batch_ids
    ]
    selected_batch_labels = [label for label, value in batch_options.items() if value in remembered_ids]
    label_options = {
        "No labels": "none",
        "Selected labels": "selected",
        "Observed labels": "observed",
        "Selected + observed": "both",
    }
    remembered_scope = str(remembered.get("label_scope", "none"))
    label_key = next((label for label, value in label_options.items() if value == remembered_scope), "No labels")
    return {
        "prediction_pool": mo.ui.switch(
            value=bool(remembered.get("show_prediction_pool", True)),
            label="Prediction pool",
            on_change=lambda value: remember("show_prediction_pool", bool(value)),
        ),
        "selected": mo.ui.switch(
            value=bool(remembered.get("show_selected", True)),
            label="Selected overlay",
            on_change=lambda value: remember("show_selected", bool(value)),
        ),
        "observed_batches": mo.ui.multiselect(
            batch_options,
            value=selected_batch_labels,
            label="Observed batches",
            on_change=lambda value: remember("observed_batches", [str(item) for item in value]),
        ),
        "labels": mo.ui.dropdown(
            label_options,
            value=label_key,
            label="Labels",
            on_change=lambda value: remember("label_scope", str(value)),
        ),
    }


def read_notebook_layered_scatter_state(controls: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read widget values outside their definition cell."""

    if not controls or controls.get("prediction_pool") is None:
        return None
    return {
        "show_prediction_pool": bool(controls["prediction_pool"].value),
        "show_selected": bool(controls["selected"].value),
        "observed_batches": [str(value) for value in controls["observed_batches"].value],
        "label_scope": str(controls["labels"].value),
    }


__all__ = ["build_notebook_layered_scatter_controls", "read_notebook_layered_scatter_state"]
