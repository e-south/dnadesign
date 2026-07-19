"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/layered_scatter.py

Build manifest-backed layered-scatter contracts for generated notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping

import pandas as pd

from .layered_scatter_contract import build_notebook_layered_scatter_contract
from .layered_scatter_controls import (
    build_notebook_layered_scatter_controls,
    read_notebook_layered_scatter_state,
)
from .layered_scatter_rendering import render_layered_scatter_figure


def filter_notebook_layered_scatter_rows(
    rows: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply independent layer and annotation controls to one tidy scatter table."""

    state = dict(state or {})
    view = _mapping(contract.get("view"))
    record_column = str(view["record_kind_column"])
    selected_column = str(view["selection_column"])
    batch_column = str(view["batch_column"])
    label_column = str(view["label_column"])
    prediction_value = str(view["prediction_value"])
    observed_value = str(view["observed_value"])
    required = {record_column, selected_column, batch_column, label_column, "id"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Layered-scatter rows are missing columns: {missing}.")

    show_pool = _state_bool(state, "show_prediction_pool", True)
    show_selected = _state_bool(state, "show_selected", True)
    known_batches = {str(item["id"]) for item in contract.get("observed_batches") or []}
    selected_batches = {
        str(value) for value in state.get("observed_batches", sorted(known_batches)) if str(value).strip()
    }
    unknown = sorted(selected_batches - known_batches)
    if unknown:
        raise ValueError(f"Unknown observed batch IDs: {unknown}.")
    label_scope = str(state.get("label_scope", "none")).strip().lower()
    if label_scope not in {"none", "selected", "observed", "both"}:
        raise ValueError("label_scope must be none, selected, observed, or both.")
    effective_label_scope = _effective_label_scope(
        label_scope,
        show_selected=show_selected,
        show_observed=bool(selected_batches),
    )

    kinds = rows[record_column].astype(str)
    selected_flags = rows[selected_column].fillna(False).astype(bool)
    prediction_mask = kinds.eq(prediction_value)
    pool_mask = prediction_mask & show_pool
    selection_mask = kinds.eq(prediction_value) & selected_flags & show_selected
    observed_mask = kinds.eq(observed_value) & rows[batch_column].astype(str).isin(selected_batches)
    filtered = rows.loc[pool_mask | selection_mask | observed_mask].copy()

    annotate = pd.Series(False, index=filtered.index)
    filtered_kinds = filtered[record_column].astype(str)
    filtered_selected = filtered[selected_column].fillna(False).astype(bool)
    if effective_label_scope in {"selected", "both"}:
        annotate |= filtered_kinds.eq(prediction_value) & filtered_selected
    if effective_label_scope in {"observed", "both"}:
        annotate |= filtered_kinds.eq(observed_value)
    visible_labels = filtered.loc[annotate, label_column]
    missing_labels = visible_labels.isna() | visible_labels.astype(str).str.strip().eq("")
    if missing_labels.any():
        ids = filtered.loc[annotate].loc[missing_labels, "id"].astype(str).tolist()[:5]
        raise ValueError(f"Visible annotations are missing display labels (sample IDs: {ids}).")
    result = filtered.reset_index(drop=True)
    result.attrs["requested_label_scope"] = label_scope
    result.attrs["effective_label_scope"] = effective_label_scope
    result.attrs["show_prediction_pool"] = show_pool
    result.attrs["show_selected"] = show_selected
    result.attrs["annotate_row_positions"] = tuple(
        position for position, should_annotate in enumerate(annotate.to_numpy(dtype=bool)) if should_annotate
    )
    if result.empty:
        result.attrs["empty_state"] = "all_layers_hidden"
    return result


def render_notebook_layered_scatter_image(
    choice: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None,
    mo: Any,
) -> Any:
    """Render one server-side layered scatter without browser-side data sprawl."""

    prepared = contract if contract is not None else build_notebook_layered_scatter_contract(choice)
    if prepared is None:
        raise ValueError("Plot choice does not declare a layered-scatter notebook adapter.")
    rows = prepared.get("rows")
    if not isinstance(rows, pd.DataFrame):
        raise ValueError("Layered-scatter contract is missing its verified tidy rows.")
    filtered = filter_notebook_layered_scatter_rows(rows, contract=prepared, state=state)
    if filtered.empty:
        return mo.callout(
            mo.md(
                "**No scatter layers are visible.** Enable the prediction pool, selected candidates, "
                "or at least one observed batch."
            ),
            kind="neutral",
        )
    figure = render_layered_scatter_figure(filtered, contract=prepared)
    payload = BytesIO()
    figure.savefig(payload, format="png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    import matplotlib.pyplot as plt

    plt.close(figure)
    selected_batches = {
        str(value)
        for value in dict(state or {}).get(
            "observed_batches",
            [item["id"] for item in prepared["observed_batches"]],
        )
    }
    batch_labels = [str(item["label"]) for item in prepared["observed_batches"] if str(item["id"]) in selected_batches]
    runtime = _mapping(prepared["runtime"])
    color_context = str(_mapping(runtime.get("color_scale")).get("context") or "").strip()
    x_context = str(runtime.get("x_label") or "").strip()
    y_context = str(runtime.get("y_label") or "").strip()
    return mo.image(
        payload.getvalue(),
        alt=(
            "Layered scatter of the campaign-scoped prediction pool, active-view selections, and measured "
            "observations selected by study batch."
        ),
        caption=(
            f"Horizontal: {x_context}. Vertical: {y_context}. Color: {color_context}. "
            "Interpret all three encodings together; no single axis determines selection. "
            "Observed vectors are measured evidence rescored under the active selection view. "
            f"Visible observed batches: {', '.join(batch_labels) if batch_labels else 'none'}."
        ),
        rounded=True,
        style={
            "width": "auto",
            "max-height": "min(76vh, 860px)",
            "max-width": "100%",
            "height": "auto",
            "object-fit": "contain",
            "margin": "0 auto",
            "display": "block",
            "background": "white",
        },
    )


def _state_bool(state: Mapping[str, Any], key: str, default: bool) -> bool:
    value = state.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _effective_label_scope(label_scope: str, *, show_selected: bool, show_observed: bool) -> str:
    selected = label_scope in {"selected", "both"} and show_selected
    observed = label_scope in {"observed", "both"} and show_observed
    if selected and observed:
        return "both"
    if selected:
        return "selected"
    if observed:
        return "observed"
    return "none"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "build_notebook_layered_scatter_contract",
    "build_notebook_layered_scatter_controls",
    "filter_notebook_layered_scatter_rows",
    "read_notebook_layered_scatter_state",
    "render_notebook_layered_scatter_image",
]
