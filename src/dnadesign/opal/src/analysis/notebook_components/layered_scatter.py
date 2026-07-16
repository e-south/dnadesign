"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/layered_scatter.py

Build manifest-backed layered-scatter contracts for generated notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...plots._mpl_utils import observed_batch_marker_map, pretty_batch_label
from ...plots.manifests import verified_plot_tidy_csv
from .layered_scatter_controls import (
    build_notebook_layered_scatter_controls,
    read_notebook_layered_scatter_state,
)
from .layered_scatter_rendering import render_layered_scatter_figure


def build_notebook_layered_scatter_contract(choice: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build one fail-fast control contract from a plot manifest and tidy table."""

    manifest = _mapping(choice.get("manifest"))
    metadata = _mapping(manifest.get("metadata"))
    view = _mapping(metadata.get("notebook_view"))
    if not view:
        return None
    if view.get("adapter") != "layered_scatter_v1":
        raise ValueError(f"Unsupported notebook plot adapter: {view.get('adapter')!r}.")
    runtime = _mapping(_mapping(manifest.get("artifact_metadata")).get("notebook_view"))
    workdir = str(choice.get("workdir") or "").strip()
    if not workdir:
        raise ValueError("Layered-scatter plot choice requires the campaign workdir.")
    tidy_path = verified_plot_tidy_csv(
        manifest,
        plot_root=Path(workdir) / "outputs" / "plots",
    )

    required_spec = {
        "record_kind_column",
        "prediction_value",
        "observed_value",
        "selection_column",
        "batch_column",
        "label_column",
        "x_column",
        "y_column",
        "color_column",
    }
    missing_spec = sorted(required_spec - set(view))
    if missing_spec:
        raise ValueError(f"Layered-scatter adapter is missing fields: {missing_spec}.")
    required_runtime = {
        "title",
        "context",
        "x_label",
        "y_label",
        "color_label",
        "x_boundary",
        "y_boundary",
        "color_extent",
        "x_limits",
        "y_limits",
    }
    missing_runtime = sorted(required_runtime - set(runtime))
    if missing_runtime:
        raise ValueError(f"Layered-scatter runtime metadata is missing fields: {missing_runtime}.")

    columns = [str(view[key]) for key in required_spec if key.endswith("_column")]
    tidy = pd.read_csv(tidy_path, low_memory=False)
    missing_columns = sorted(set(columns) - set(tidy.columns))
    if missing_columns:
        raise ValueError(f"Layered-scatter tidy table is missing columns: {missing_columns}.")
    _validate_tidy_semantics(tidy, view=view)
    record_column = str(view["record_kind_column"])
    batch_column = str(view["batch_column"])
    observed_value = str(view["observed_value"])
    observed_batches = tidy.loc[tidy[record_column].astype(str).eq(observed_value), batch_column]
    batch_ids = sorted(observed_batches.astype(str).unique().tolist())
    if not batch_ids:
        raise ValueError("Layered-scatter review requires at least one observed batch.")
    observed_batch_marker_map(tuple(batch_ids), universe_batch_ids=tuple(batch_ids))
    batch_labels = _unique_observed_batch_labels(batch_ids)
    return {
        "adapter": "layered_scatter_v1",
        "key": _layered_scatter_memory_key(manifest, workdir=workdir),
        "tidy_path": tidy_path,
        "view": dict(view),
        "runtime": dict(runtime),
        "observed_batches": [{"id": value, "label": batch_labels[value]} for value in batch_ids],
    }


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
    state: Mapping[str, Any] | None,
    mo: Any,
) -> Any:
    """Render one server-side layered scatter without browser-side data sprawl."""

    contract = build_notebook_layered_scatter_contract(choice)
    if contract is None:
        raise ValueError("Plot choice does not declare a layered-scatter notebook adapter.")
    rows = pd.read_csv(contract["tidy_path"], low_memory=False)
    filtered = filter_notebook_layered_scatter_rows(rows, contract=contract, state=state)
    if filtered.empty:
        return mo.callout(
            mo.md(
                "**No scatter layers are visible.** Enable the prediction pool, selected candidates, "
                "or at least one observed batch."
            ),
            kind="neutral",
        )
    figure = render_layered_scatter_figure(filtered, contract=contract)
    payload = BytesIO()
    figure.savefig(payload, format="png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    import matplotlib.pyplot as plt

    plt.close(figure)
    selected_batches = {
        str(value)
        for value in dict(state or {}).get(
            "observed_batches",
            [item["id"] for item in contract["observed_batches"]],
        )
    }
    batch_labels = [str(item["label"]) for item in contract["observed_batches"] if str(item["id"]) in selected_batches]
    return mo.image(
        payload.getvalue(),
        alt=(
            "Layered scatter of the campaign-scoped prediction pool, active-view selections, and measured "
            "observations selected by study batch."
        ),
        caption=(
            "Observed vectors are measured evidence rescored under the active selection view. "
            f"Visible observed batches: {', '.join(batch_labels) if batch_labels else 'none'}."
        ),
        rounded=True,
        style={
            "width": "auto",
            "max-height": "min(62vh, 720px)",
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


def _layered_scatter_memory_key(manifest: Mapping[str, Any], *, workdir: str) -> str:
    identity = {
        "workdir": str(Path(workdir).expanduser().resolve()),
        "plot": manifest.get("plot_id") or manifest.get("name") or manifest.get("kind"),
        "kind": manifest.get("kind"),
        "run_id": manifest.get("run_id"),
        "selection_view_id": manifest.get("selection_view_id"),
        "rounds": manifest.get("rounds"),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"layered_scatter_v1:{hashlib.sha256(payload).hexdigest()}"


def _unique_observed_batch_labels(batch_ids: list[str]) -> dict[str, str]:
    base_labels = {batch_id: pretty_batch_label(batch_id) for batch_id in batch_ids}
    counts: dict[str, int] = {}
    for label in base_labels.values():
        counts[label] = counts.get(label, 0) + 1
    return {
        batch_id: (f"{label} · {batch_id}" if counts[label] > 1 else label) for batch_id, label in base_labels.items()
    }


def _validate_tidy_semantics(rows: pd.DataFrame, *, view: Mapping[str, Any]) -> None:
    record_column = str(view["record_kind_column"])
    selected_column = str(view["selection_column"])
    batch_column = str(view["batch_column"])
    prediction_value = str(view["prediction_value"])
    observed_value = str(view["observed_value"])
    kinds = rows[record_column]
    allowed_kinds = {prediction_value, observed_value}
    invalid_kinds = kinds.isna() | ~kinds.isin(allowed_kinds)
    if invalid_kinds.any():
        sample = ["<null>" if pd.isna(value) else str(value) for value in kinds.loc[invalid_kinds].tolist()[:5]]
        raise ValueError(
            "Layered-scatter record_kind values must match the declared prediction and observed values "
            f"(sample: {sample})."
        )

    numeric_columns = [str(view[key]) for key in ("x_column", "y_column", "color_column")]
    try:
        numeric = rows[numeric_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Layered-scatter x, y, and color columns require finite numeric values.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("Layered-scatter x, y, and color columns require finite numeric values.")

    selected = rows[selected_column]
    valid_selected = selected.map(lambda value: pd.isna(value) or isinstance(value, (bool, np.bool_)))
    if not valid_selected.all():
        raise ValueError("Layered-scatter selected values must be boolean or null.")
    prediction_mask = kinds.eq(prediction_value)
    observed_mask = kinds.eq(observed_value)
    if selected.loc[prediction_mask].isna().any():
        raise ValueError("Layered-scatter prediction rows require boolean selected values.")
    observed_selected = selected.loc[observed_mask].dropna()
    if any(bool(value) for value in observed_selected):
        raise ValueError("Layered-scatter observed rows require selected to be false or null.")

    batches = rows[batch_column]
    if batches.loc[prediction_mask].notna().any():
        raise ValueError("Layered-scatter prediction rows require null batch IDs.")
    observed_batches = batches.loc[observed_mask]
    if observed_batches.isna().any() or observed_batches.astype(str).str.strip().eq("").any():
        raise ValueError("Layered-scatter observed rows require non-empty batch IDs.")


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
