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
        return {
            key: None
            for key in (
                "figure",
                "prediction_pool",
                "selected",
                "selection_rounds",
                "observed_batches",
                "labels",
            )
        }
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
    interactive = bool(contract.get("interactive"))
    figure_options = {
        "2D publication figure": "publication_2d",
        "3D interactive inspector": "interactive_3d",
    }
    remembered_figure = str(remembered.get("figure", "publication_2d"))
    figure_key = next(
        (label for label, value in figure_options.items() if value == remembered_figure),
        "2D publication figure",
    )
    selection_round_values = [int(value) for value in contract.get("selection_rounds") or []]
    active_selection_round = int(contract["active_selection_round"])
    selection_round_options = {f"Round {round_k}": round_k for round_k in selection_round_values}
    remembered_rounds = {
        int(value)
        for value in remembered.get("selection_rounds", [active_selection_round])
        if not isinstance(value, bool) and str(value).lstrip("-").isdigit()
    }
    selected_round_labels = [
        label for label, round_k in selection_round_options.items() if round_k in remembered_rounds
    ]
    if not selected_round_labels:
        selected_round_labels = [f"Round {active_selection_round}"]
    return {
        "figure": (
            mo.ui.dropdown(
                figure_options,
                value=figure_key,
                label="Figure",
                on_change=lambda value: remember("figure", str(value)),
                full_width=True,
            )
            if interactive
            else None
        ),
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
        "selection_rounds": (
            mo.ui.multiselect(
                selection_round_options,
                value=selected_round_labels,
                label="Selected rounds",
                on_change=lambda value: remember("selection_rounds", [int(item) for item in value]),
                full_width=True,
            )
            if len(selection_round_options) > 1
            else None
        ),
        "observed_batches": mo.ui.multiselect(
            batch_options,
            value=selected_batch_labels,
            label="Observed batches",
            on_change=lambda value: remember("observed_batches", [str(item) for item in value]),
            full_width=True,
        ),
        "labels": mo.ui.dropdown(
            label_options,
            value=label_key,
            label="2D annotations" if interactive else "Labels",
            on_change=lambda value: remember("label_scope", str(value)),
            full_width=True,
        ),
    }


def read_notebook_layered_scatter_state(controls: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read widget values outside their definition cell."""

    if not controls or controls.get("prediction_pool") is None:
        return None
    figure = controls.get("figure")
    selection_rounds = controls.get("selection_rounds")
    return {
        "figure": str(figure.value) if figure is not None else "publication_2d",
        "show_prediction_pool": bool(controls["prediction_pool"].value),
        "show_selected": bool(controls["selected"].value),
        "selection_rounds": (
            [int(value) for value in selection_rounds.value] if selection_rounds is not None else None
        ),
        "observed_batches": [str(value) for value in controls["observed_batches"].value],
        "label_scope": str(controls["labels"].value),
    }


__all__ = ["build_notebook_layered_scatter_controls", "read_notebook_layered_scatter_state"]
