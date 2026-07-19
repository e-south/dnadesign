"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/layered_scatter_contract.py

Load and validate manifest-backed layered-scatter evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ...plots._mpl_utils import observed_batch_marker_map, pretty_batch_label
from ...plots.manifests import verified_plot_tidy_csv


def build_notebook_layered_scatter_contract(choice: Mapping[str, Any]) -> dict[str, Any] | None:
    """Load one verified scatter table and its immutable display contract."""

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

    column_spec_fields = (
        "record_kind_column",
        "selection_column",
        "batch_column",
        "label_column",
        "x_column",
        "y_column",
        "color_column",
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
        "reference_lines",
        "color_scale",
        "x_limits",
        "y_limits",
    }
    missing_runtime = sorted(required_runtime - set(runtime))
    if missing_runtime:
        raise ValueError(f"Layered-scatter runtime metadata is missing fields: {missing_runtime}.")
    _validate_runtime_semantics(runtime)

    columns = list(dict.fromkeys(["id", *(str(view[key]) for key in column_spec_fields)]))
    tidy = pd.read_csv(tidy_path, low_memory=False)
    missing_columns = sorted(set(columns) - set(tidy.columns))
    if missing_columns:
        raise ValueError(f"Layered-scatter tidy table is missing columns: {missing_columns}.")
    _validate_tidy_semantics(tidy, view=view)
    tidy = tidy.loc[:, columns].copy()
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
        "rows": tidy,
        "view": dict(view),
        "runtime": dict(runtime),
        "observed_batches": [{"id": value, "label": batch_labels[value]} for value in batch_ids],
    }


def _layered_scatter_memory_key(manifest: Mapping[str, Any], *, workdir: str) -> str:
    # Selection views change the evidence shown, not the operator's display preferences.
    identity = {
        "workdir": str(Path(workdir).expanduser().resolve()),
        "plot": manifest.get("plot_id") or manifest.get("name") or manifest.get("kind"),
        "kind": manifest.get("kind"),
        "run_id": manifest.get("run_id"),
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


def _validate_runtime_semantics(runtime: Mapping[str, Any]) -> None:
    reference_lines = _mapping(runtime.get("reference_lines"))
    if set(reference_lines) != {"x", "y"}:
        raise ValueError("Layered-scatter reference_lines must contain exactly x and y lists.")
    for axis in ("x", "y"):
        lines = reference_lines[axis]
        if not isinstance(lines, list):
            raise ValueError(f"Layered-scatter reference_lines.{axis} must be a list.")
        for item in lines:
            if not isinstance(item, Mapping) or set(item) != {"value", "label"}:
                raise ValueError(f"Layered-scatter reference_lines.{axis} entries require exactly value and label.")
            value = float(item["value"])
            label = str(item["label"]).strip()
            if not np.isfinite(value) or not label:
                raise ValueError(f"Layered-scatter reference_lines.{axis} values must be finite and labels non-empty.")
    color_scale = _mapping(runtime.get("color_scale"))
    required_color_scale = {"center", "extent", "context"}
    allowed_color_scale = {*required_color_scale, "extend"}
    if missing := sorted(required_color_scale - set(color_scale)):
        raise ValueError(f"Layered-scatter color_scale is missing fields: {missing}.")
    if extra := sorted(set(color_scale) - allowed_color_scale):
        raise ValueError(f"Layered-scatter color_scale contains unsupported fields: {extra}.")
    center = float(color_scale["center"])
    extent = float(color_scale["extent"])
    context = str(color_scale["context"]).strip()
    if not np.isfinite(center) or not np.isfinite(extent) or extent <= 0.0 or not context:
        raise ValueError(
            "Layered-scatter color_scale requires a finite center, positive finite extent, and non-empty context."
        )
    extend = str(color_scale.get("extend") or "neither")
    if extend not in {"neither", "both", "min", "max"}:
        raise ValueError("Layered-scatter color_scale.extend must be neither, both, min, or max.")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["build_notebook_layered_scatter_contract"]
