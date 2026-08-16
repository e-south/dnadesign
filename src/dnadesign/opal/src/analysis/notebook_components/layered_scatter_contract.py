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

from ...plots._mpl_utils import compact_batch_label, observed_batch_marker_map, pretty_batch_label
from ...plots.manifests import verified_plot_tidy_csv
from .layered_scatter_display import invariant_round_display, runtime_limits, shared_colorbar_extend
from .layered_scatter_rounds import resolve_layered_scatter_selection_rounds


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

    interactive = _interactive_spec(view)
    columns = list(
        dict.fromkeys(
            [
                "id",
                *(str(view[key]) for key in column_spec_fields),
                *([str(interactive["score_column"])] if interactive else []),
            ]
        )
    )
    tidy = pd.read_csv(tidy_path, low_memory=False)
    missing_columns = sorted(set(columns) - set(tidy.columns))
    if missing_columns:
        raise ValueError(f"Layered-scatter tidy table is missing columns: {missing_columns}.")
    _validate_tidy_semantics(tidy, view=view, interactive=interactive)
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
    active_selection_round, round_options = resolve_layered_scatter_selection_rounds(choice, active_manifest=manifest)
    selection_rows, shared_display = _load_selection_round_rows(
        round_options,
        active_manifest=manifest,
        view=view,
        runtime=runtime,
        interactive=interactive,
        columns=columns,
    )
    shared_runtime = dict(runtime)
    shared_color_scale = dict(_mapping(runtime["color_scale"]))
    shared_color_scale.update(
        extent=shared_display["color_extent"],
        context=(
            shared_display["color_contexts"][0]
            if len(round_options) == 1
            else "shared across loaded rounds; endpoint values remain in the plotted data"
        ),
        extend=shared_colorbar_extend(
            minimum=shared_display["color_min"],
            maximum=shared_display["color_max"],
            center=float(shared_color_scale["center"]),
            extent=shared_display["color_extent"],
        ),
    )
    shared_runtime["color_scale"] = shared_color_scale
    shared_runtime["x_limits"] = shared_display["x_limits"]
    shared_runtime["y_limits"] = shared_display["y_limits"]
    return {
        "adapter": "layered_scatter_v1",
        "key": _layered_scatter_memory_key(manifest, workdir=workdir),
        "tidy_path": tidy_path,
        "rows": tidy,
        "view": dict(view),
        "runtime": shared_runtime,
        "interactive": dict(interactive),
        "active_selection_round": active_selection_round,
        "selection_rounds": sorted(round_options),
        "selection_rows": selection_rows,
        "observed_batches": [{"id": value, "label": batch_labels[value]} for value in batch_ids],
    }


def _load_selection_round_rows(
    round_options: Mapping[int, Mapping[str, Any]],
    *,
    active_manifest: Mapping[str, Any],
    view: Mapping[str, Any],
    runtime: Mapping[str, Any],
    interactive: Mapping[str, Any],
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_column = str(view["selection_column"])
    record_column = str(view["record_kind_column"])
    prediction_value = str(view["prediction_value"])
    active_kind = str(active_manifest.get("kind") or "")
    active_selection_view = str(active_manifest.get("selection_view_id") or "")
    cohorts: list[pd.DataFrame] = []
    color_extents: list[float] = []
    color_contexts: set[str] = set()
    color_minima: list[float] = []
    color_maxima: list[float] = []
    x_limits: list[tuple[float, float]] = []
    y_limits: list[tuple[float, float]] = []
    for round_k, option in sorted(round_options.items()):
        option_manifest = _mapping(option.get("manifest"))
        if str(option_manifest.get("kind") or "") != active_kind:
            raise ValueError("Layered-scatter round overlays must use the same plot kind.")
        if str(option_manifest.get("selection_view_id") or "") != active_selection_view:
            raise ValueError("Layered-scatter round overlays must use the same selection view.")
        option_view = _mapping(_mapping(option_manifest.get("metadata")).get("notebook_view"))
        if option_view != view:
            raise ValueError("Layered-scatter round overlays must use the same notebook view contract.")
        option_runtime = _mapping(_mapping(option_manifest.get("artifact_metadata")).get("notebook_view"))
        _validate_runtime_semantics(option_runtime)
        if invariant_round_display(option_runtime) != invariant_round_display(runtime):
            raise ValueError("Layered-scatter round overlays must use the same coordinate display contract.")
        option_color_scale = _mapping(option_runtime["color_scale"])
        color_extents.append(float(option_color_scale["extent"]))
        color_contexts.add(str(option_color_scale["context"]))
        x_limits.append(runtime_limits(option_runtime, "x_limits"))
        y_limits.append(runtime_limits(option_runtime, "y_limits"))
        workdir = str(option.get("workdir") or "").strip()
        if not workdir:
            raise ValueError("Layered-scatter round overlay requires the campaign workdir.")
        tidy_path = verified_plot_tidy_csv(
            option_manifest,
            plot_root=Path(workdir) / "outputs" / "plots",
        )
        tidy = pd.read_csv(tidy_path, low_memory=False)
        missing = sorted(set(columns) - set(tidy.columns))
        if missing:
            raise ValueError(f"Layered-scatter round overlay is missing columns: {missing}.")
        _validate_tidy_semantics(tidy, view=view, interactive=interactive)
        color_values = pd.to_numeric(tidy[str(view["color_column"])], errors="raise")
        color_minima.append(float(color_values.min()))
        color_maxima.append(float(color_values.max()))
        selected = tidy.loc[
            tidy[record_column].astype(str).eq(prediction_value) & tidy[selected_column].fillna(False).astype(bool),
            columns,
        ].copy()
        if selected.empty:
            raise ValueError(f"Layered-scatter selection round {round_k} contains no selected candidates.")
        selected["__notebook_selection_round"] = round_k
        cohorts.append(selected)
    return pd.concat(cohorts, ignore_index=True), {
        "color_extent": max(color_extents),
        "color_contexts": sorted(color_contexts),
        "color_min": min(color_minima),
        "color_max": max(color_maxima),
        "x_limits": [min(lower for lower, _ in x_limits), max(upper for _, upper in x_limits)],
        "y_limits": [min(lower for lower, _ in y_limits), max(upper for _, upper in y_limits)],
    }


def _layered_scatter_memory_key(manifest: Mapping[str, Any], *, workdir: str) -> str:
    # Selection views and rounds change the evidence shown, not the operator's display preferences.
    identity = {
        "workdir": str(Path(workdir).expanduser().resolve()),
        "plot": manifest.get("name") or manifest.get("kind"),
        "kind": manifest.get("kind"),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"layered_scatter_v1:{hashlib.sha256(payload).hexdigest()}"


def _unique_observed_batch_labels(batch_ids: list[str]) -> dict[str, str]:
    base_labels = {batch_id: _notebook_batch_label(batch_id) for batch_id in batch_ids}
    counts: dict[str, int] = {}
    for label in base_labels.values():
        counts[label] = counts.get(label, 0) + 1
    return {
        batch_id: (f"{label} · {batch_id}" if counts[label] > 1 else label) for batch_id, label in base_labels.items()
    }


def _notebook_batch_label(batch_id: str) -> str:
    full = pretty_batch_label(batch_id)
    return compact_batch_label(batch_id) if len(full) > 28 else full


def _validate_tidy_semantics(
    rows: pd.DataFrame,
    *,
    view: Mapping[str, Any],
    interactive: Mapping[str, Any],
) -> None:
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
    if interactive:
        numeric_columns.append(str(interactive["score_column"]))
    try:
        numeric = rows[numeric_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Layered-scatter display coordinates and scores require finite numeric values.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("Layered-scatter display coordinates and scores require finite numeric values.")

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


def _interactive_spec(view: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = view.get("interactive")
    if raw is None:
        return {}
    interactive = _mapping(raw)
    required = {
        "adapter",
        "score_column",
        "score_label",
        "prediction_sample_limit",
        "sampling_method",
    }
    if missing := sorted(required - set(interactive)):
        raise ValueError(f"Layered-scatter interactive adapter is missing fields: {missing}.")
    if extra := sorted(set(interactive) - required):
        raise ValueError(f"Layered-scatter interactive adapter contains unsupported fields: {extra}.")
    if interactive["adapter"] != "three_axis_scatter_v1":
        raise ValueError(f"Unsupported layered-scatter interactive adapter: {interactive['adapter']!r}.")
    if not str(interactive["score_column"]).strip() or not str(interactive["score_label"]).strip():
        raise ValueError("Layered-scatter interactive score column and label must be non-empty.")
    limit = interactive["prediction_sample_limit"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Layered-scatter interactive prediction_sample_limit must be a positive integer.")
    if interactive["sampling_method"] != "sha256_id_v1":
        raise ValueError("Layered-scatter interactive sampling_method must be 'sha256_id_v1'.")
    return interactive


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["build_notebook_layered_scatter_contract"]
