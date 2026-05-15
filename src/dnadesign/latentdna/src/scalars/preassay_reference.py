"""Reference-set pre-assay scalar builders."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError
from ..geometry.cohorts import centroid_map, group_indices
from ..geometry.preprocessing import try_l2_normalize_vector
from ..labels import humanize_label
from ..reference_sets import resolve_reference_set_rows
from ..workspaces.loader import WorkspaceContext
from .common import (
    ScalarInputRef,
    _cosine_distance_upper_from_normalized,
    _metric_row,
    _normalized_geometry_rows,
    _optional_param,
    _require_param,
)
from .preassay_common import (
    ScalarBuilderResult,
    _load_candidate_sample,
    _reference_group_panel_title,
    _reference_set_panel_title,
)


def _configured_reference_set_ids(params: dict[str, Any]) -> list[str]:
    configured = _optional_param(params, "reference_sets", default=[])
    reference_set_ids: list[str] = []
    for item in list(configured or []):
        if isinstance(item, dict):
            reference_set_id = str(item.get("reference_set_id") or item.get("id") or "").strip()
        else:
            reference_set_id = str(item or "").strip()
        if reference_set_id:
            reference_set_ids.append(reference_set_id)
    return list(dict.fromkeys(reference_set_ids))


def _reference_indices_for_matched_ids(
    rows: list[dict[str, object]],
    *,
    match_column: str,
    matched_ids: list[str],
) -> list[int]:
    matched = set(matched_ids)
    return [
        index
        for index, row in enumerate(rows)
        if row.get(match_column) is not None and str(row.get(match_column)) in matched
    ]


def _reference_status(
    *,
    missing_columns: list[str],
    expected_ids: list[str],
    matched_ids: list[str],
    selected_count: int,
    min_reference_group_size: int,
) -> str:
    if missing_columns:
        return "missing_columns"
    if not expected_ids:
        return "absent"
    if not matched_ids:
        return "missing_rows"
    if selected_count < min_reference_group_size:
        return "too_small"
    return "ok"


def _reference_distance_summary(
    normalized: np.ndarray,
    indices: list[int],
    *,
    min_reference_group_size: int,
) -> tuple[float, float]:
    if len(indices) < min_reference_group_size:
        return float("nan"), float("nan")
    distances = _cosine_distance_upper_from_normalized(np.asarray(normalized[indices], dtype=np.float32))
    if not distances.size:
        return float("nan"), float("nan")
    return (
        float(np.median(distances)),
        float(np.percentile(distances, 75.0) - np.percentile(distances, 25.0)),
    )


def _append_reference_set_rows(
    rows: list[dict[str, object]],
    *,
    context: WorkspaceContext,
    descriptor: dict[str, object],
    normalized: np.ndarray,
    candidate_rows: list[dict[str, object]],
    reference_set_id: str,
    min_reference_group_size: int,
) -> int:
    if reference_set_id not in context.config.reference_sets:
        raise ContractViolationError(
            f"reference_alignment_summary references unknown reference_set {reference_set_id!r}"
        )
    reference_set = context.config.reference_sets[reference_set_id]
    resolution = resolve_reference_set_rows(reference_set, candidate_rows)
    match_column = str(getattr(reference_set, "match_column"))
    indices = _reference_indices_for_matched_ids(
        candidate_rows,
        match_column=match_column,
        matched_ids=resolution.matched_ids,
    )
    status = _reference_status(
        missing_columns=resolution.missing_columns,
        expected_ids=resolution.expected_ids,
        matched_ids=resolution.matched_ids,
        selected_count=len(indices),
        min_reference_group_size=min_reference_group_size,
    )
    reference_set_label = str(getattr(reference_set, "label", None) or humanize_label(reference_set_id))
    distance_median, distance_iqr = _reference_distance_summary(
        normalized,
        indices,
        min_reference_group_size=min_reference_group_size,
    )
    extra = {
        "reference_group_column": "reference_set",
        "reference_group": reference_set_id,
        "reference_rows": len(indices),
        "reference_set_id": reference_set_id,
        "reference_set_label": reference_set_label,
        "reference_set_status": status,
        "reference_set_complete": bool(resolution.complete),
        "reference_set_missing_columns": ", ".join(resolution.missing_columns),
        "reference_expected_rows": len(resolution.expected_ids),
        "reference_matched_rows": len(resolution.matched_ids),
        "category": f"reference_set: {reference_set_id}",
        "label": reference_set_label,
    }
    rows.extend(
        [
            _metric_row(
                context=context,
                descriptor=descriptor,
                metric_id="reference_group_size",
                metric_value=float(len(indices)),
                category="reference collapse",
                extra={
                    **extra,
                    "display_name": _reference_set_panel_title(
                        metric_id="reference_group_size",
                        reference_set_label=reference_set_label,
                    ),
                },
            ),
            _metric_row(
                context=context,
                descriptor=descriptor,
                metric_id="reference_group_pairwise_cosine_distance_median",
                metric_value=distance_median,
                category="reference collapse",
                extra={
                    **extra,
                    "display_name": _reference_set_panel_title(
                        metric_id="reference_group_pairwise_cosine_distance_median",
                        reference_set_label=reference_set_label,
                    ),
                },
            ),
            _metric_row(
                context=context,
                descriptor=descriptor,
                metric_id="reference_group_pairwise_cosine_distance_iqr",
                metric_value=distance_iqr,
                category="reference collapse",
                extra={
                    **extra,
                    "display_name": _reference_set_panel_title(
                        metric_id="reference_group_pairwise_cosine_distance_iqr",
                        reference_set_label=reference_set_label,
                    ),
                },
            ),
        ]
    )
    return 3


def _reference_alignment_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    reference_group_columns = [
        str(value)
        for value in _optional_param(
            params,
            "reference_group_columns",
            default=[],
        )
    ]
    reference_set_ids = _configured_reference_set_ids(params)
    reference_label_column = str(_optional_param(params, "reference_label_column", default="usr_label__primary"))
    min_reference_group_size = int(_optional_param(params, "min_reference_group_size", default=2))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        design_groups = group_indices(candidate_sample.rows, column="design_family")
        reference_groups = group_indices(candidate_sample.rows, column="usr_label__primary")
        emitted_rows = 0
        if (
            {"background_only", "ethanol", "ciprofloxacin"}.issubset(design_groups)
            and any(label.lower() == "spyp" for label in reference_groups)
            and any(label.lower() == "sulap" for label in reference_groups)
        ):
            centroids = centroid_map(normalized, design_groups)
            reference_centroids = {
                label.lower(): centroid
                for label, indices in reference_groups.items()
                if (centroid := try_l2_normalize_vector(np.asarray(normalized[indices].mean(axis=0), dtype=np.float32)))
                is not None
            }
            ethanol_alignment = float(np.dot(centroids["ethanol"], reference_centroids["spyp"])) - float(
                np.dot(centroids["background_only"], reference_centroids["spyp"])
            )
            cipro_alignment = float(np.dot(centroids["ciprofloxacin"], reference_centroids["sulap"])) - float(
                np.dot(centroids["background_only"], reference_centroids["sulap"])
            )
            rows.extend(
                [
                    _metric_row(
                        context=context,
                        descriptor=candidate_sample.descriptor,
                        metric_id="reference_alignment_ethanol_background_relative",
                        metric_value=ethanol_alignment,
                    ),
                    _metric_row(
                        context=context,
                        descriptor=candidate_sample.descriptor,
                        metric_id="reference_alignment_cipro_background_relative",
                        metric_value=cipro_alignment,
                    ),
                ]
            )
            emitted_rows += 2
        elif not reference_group_columns and not reference_set_ids:
            if not {"background_only", "ethanol", "ciprofloxacin"}.issubset(design_groups):
                raise ContractViolationError(
                    "reference_alignment_summary requires background_only, ethanol, "
                    f"and ciprofloxacin cohorts in {_require_param(candidate, 'view_id')!r}"
                )
            raise ContractViolationError(
                "reference_alignment_summary requires carried SpyP and SulA rows in "
                f"{_require_param(candidate, 'view_id')!r}"
            )
        reference_indices = [
            index
            for index, row in enumerate(candidate_sample.rows)
            if row.get(reference_label_column) is not None and str(row.get(reference_label_column)).strip()
        ]
        for group_column in reference_group_columns:
            grouped: dict[str, list[int]] = {}
            for index in reference_indices:
                value = candidate_sample.rows[index].get(group_column)
                if value is None or not str(value).strip() or str(value).lower() == "nan":
                    continue
                grouped.setdefault(str(value), []).append(index)
            for group_value, indices in sorted(grouped.items()):
                if len(indices) < min_reference_group_size:
                    continue
                distances = _cosine_distance_upper_from_normalized(np.asarray(normalized[indices], dtype=np.float32))
                distance_median = float(np.median(distances)) if distances.size else 0.0
                distance_iqr = (
                    float(np.percentile(distances, 75.0) - np.percentile(distances, 25.0)) if distances.size else 0.0
                )
                extra = {
                    "reference_group_column": group_column,
                    "reference_group": group_value,
                    "reference_rows": len(indices),
                    "category": f"{group_column}: {group_value}",
                    "label": group_value,
                }
                rows.extend(
                    [
                        _metric_row(
                            context=context,
                            descriptor=candidate_sample.descriptor,
                            metric_id="reference_group_size",
                            metric_value=float(len(indices)),
                            category="reference collapse",
                            extra={
                                **extra,
                                "display_name": _reference_group_panel_title(
                                    metric_id="reference_group_size",
                                    group_column=group_column,
                                    group_value=group_value,
                                ),
                            },
                        ),
                        _metric_row(
                            context=context,
                            descriptor=candidate_sample.descriptor,
                            metric_id="reference_group_pairwise_cosine_distance_median",
                            metric_value=distance_median,
                            category="reference collapse",
                            extra={
                                **extra,
                                "display_name": _reference_group_panel_title(
                                    metric_id="reference_group_pairwise_cosine_distance_median",
                                    group_column=group_column,
                                    group_value=group_value,
                                ),
                            },
                        ),
                        _metric_row(
                            context=context,
                            descriptor=candidate_sample.descriptor,
                            metric_id="reference_group_pairwise_cosine_distance_iqr",
                            metric_value=distance_iqr,
                            category="reference collapse",
                            extra={
                                **extra,
                                "display_name": _reference_group_panel_title(
                                    metric_id="reference_group_pairwise_cosine_distance_iqr",
                                    group_column=group_column,
                                    group_value=group_value,
                                ),
                            },
                        ),
                    ]
                )
                emitted_rows += 3
        for reference_set_id in reference_set_ids:
            emitted_rows += _append_reference_set_rows(
                rows,
                context=context,
                descriptor=candidate_sample.descriptor,
                normalized=normalized,
                candidate_rows=candidate_sample.rows,
                reference_set_id=reference_set_id,
                min_reference_group_size=min_reference_group_size,
            )
        if emitted_rows == 0:
            raise ContractViolationError(
                "reference_alignment_summary requires SpyP/SulA alignment rows or at least one "
                f"reference group with >= {min_reference_group_size} rows in {_require_param(candidate, 'view_id')!r}"
            )
    for row in rows:
        row.setdefault("reference_group_column", None)
        row.setdefault("reference_group", None)
        row.setdefault("reference_rows", None)
        row.setdefault("reference_set_id", None)
        row.setdefault("reference_set_label", None)
        row.setdefault("reference_set_status", None)
        row.setdefault("reference_set_complete", None)
        row.setdefault("reference_set_missing_columns", "")
        row.setdefault("reference_expected_rows", None)
        row.setdefault("reference_matched_rows", None)
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {
            "candidate_count": len(candidates),
            "reference_set_count": len(reference_set_ids),
            "rows": len(rows),
        },
    )


def _centroid_axis_groups(axis_config: dict[str, Any]) -> tuple[str, list[tuple[str, str]]]:
    column = str(_require_param(axis_config, "column"))
    raw_groups = _require_param(axis_config, "groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ContractViolationError("reference_to_centroid_similarity centroid_axis.groups must be a non-empty list")
    groups: list[tuple[str, str]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise ContractViolationError("reference_to_centroid_similarity centroid groups must be mappings")
        value = str(raw_group.get("value") or "").strip()
        if not value:
            raise ContractViolationError("reference_to_centroid_similarity centroid groups require value")
        groups.append((value, str(raw_group.get("label") or humanize_label(value))))
    return column, groups


def _reference_set_entries(params: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for raw_entry in list(_optional_param(params, "reference_sets", default=[])):
        if isinstance(raw_entry, dict):
            reference_set_id = str(raw_entry.get("reference_set_id") or raw_entry.get("id") or "").strip()
            aggregation = str(raw_entry.get("aggregation") or "centroid").strip()
        else:
            reference_set_id = str(raw_entry or "").strip()
            aggregation = "centroid"
        if not reference_set_id:
            continue
        if aggregation not in {"centroid", "rows", "both"}:
            raise ContractViolationError(
                "reference_to_centroid_similarity reference_set aggregation must be centroid, rows, or both"
            )
        entries.append({"reference_set_id": reference_set_id, "aggregation": aggregation})
    return entries


def _reference_row_label(reference_set: object, row: dict[str, object], fallback_id: str) -> str:
    display_labels = dict(getattr(reference_set, "display_labels", {}) or {})
    if fallback_id in display_labels:
        return str(display_labels[fallback_id])
    label_column = getattr(reference_set, "label_column", None)
    if label_column:
        label = row.get(str(label_column))
        if label is not None and str(label).strip():
            return str(label)
    return fallback_id


def _append_reference_to_centroid_rows(
    rows: list[dict[str, object]],
    *,
    context: WorkspaceContext,
    descriptor: dict[str, object],
    vector: np.ndarray,
    centroids: dict[str, np.ndarray],
    centroid_labels: dict[str, str],
    reference_set_id: str,
    reference_set_label: str,
    reference_entity_id: str,
    reference_entity_label: str,
    reference_entity_type: str,
    reference_set_status: str,
    reference_set_complete: bool,
    reference_rows: int,
) -> None:
    similarities = {group: float(np.dot(vector, centroid)) for group, centroid in centroids.items()}
    ordered = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    nearest_group = ordered[0][0] if ordered else ""
    nearest_similarity = ordered[0][1] if ordered else float("nan")
    second_similarity = ordered[1][1] if len(ordered) > 1 else float("nan")
    margin = nearest_similarity - second_similarity if np.isfinite(second_similarity) else float("nan")
    for centroid_group, similarity in similarities.items():
        rows.append(
            _metric_row(
                context=context,
                descriptor=descriptor,
                metric_id="reference_to_centroid_similarity",
                metric_value=similarity,
                category="reference_to_centroid_similarity",
                extra={
                    "reference_set_id": reference_set_id,
                    "reference_set_label": reference_set_label,
                    "reference_set_status": reference_set_status,
                    "reference_set_complete": reference_set_complete,
                    "reference_rows": reference_rows,
                    "reference_entity_id": reference_entity_id,
                    "reference_entity_label": reference_entity_label,
                    "reference_entity_type": reference_entity_type,
                    "centroid_group": centroid_group,
                    "centroid_label": centroid_labels[centroid_group],
                    "nearest_centroid_group": nearest_group,
                    "nearest_centroid_label": centroid_labels.get(nearest_group, nearest_group),
                    "nearest_centroid_similarity": nearest_similarity,
                    "nearest_centroid_margin": margin,
                    "label": reference_entity_label,
                },
            )
        )


def _reference_to_centroid_similarity_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    centroid_column, centroid_groups = _centroid_axis_groups(dict(_require_param(params, "centroid_axis")))
    reference_entries = _reference_set_entries(params)
    if not reference_entries:
        raise ContractViolationError("reference_to_centroid_similarity requires at least one reference_set")
    min_reference_group_size = int(_optional_param(params, "min_reference_group_size", default=2))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    centroid_values = [value for value, _ in centroid_groups]
    centroid_labels = {value: label for value, label in centroid_groups}
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        groups = group_indices(
            candidate_sample.rows,
            column=centroid_column,
            allowed_values=set(centroid_values),
        )
        missing_groups = [value for value in centroid_values if value not in groups]
        if missing_groups:
            raise ContractViolationError(
                "reference_to_centroid_similarity missing centroid groups "
                f"for {_require_param(candidate, 'view_id')!r}: {missing_groups}"
            )
        centroids = centroid_map(normalized, groups)
        centroids = {value: centroids[value] for value in centroid_values}
        for entry in reference_entries:
            reference_set_id = entry["reference_set_id"]
            if reference_set_id not in context.config.reference_sets:
                raise ContractViolationError(
                    f"reference_to_centroid_similarity references unknown reference_set {reference_set_id!r}"
                )
            reference_set = context.config.reference_sets[reference_set_id]
            resolution = resolve_reference_set_rows(reference_set, candidate_sample.rows)
            match_column = str(getattr(reference_set, "match_column"))
            indices = _reference_indices_for_matched_ids(
                candidate_sample.rows,
                match_column=match_column,
                matched_ids=resolution.matched_ids,
            )
            status = _reference_status(
                missing_columns=resolution.missing_columns,
                expected_ids=resolution.expected_ids,
                matched_ids=resolution.matched_ids,
                selected_count=len(indices),
                min_reference_group_size=1,
            )
            reference_set_label = str(getattr(reference_set, "label", None) or humanize_label(reference_set_id))
            aggregation = entry["aggregation"]
            if aggregation in {"rows", "both"}:
                for index in indices:
                    row = candidate_sample.rows[index]
                    entity_id = str(row.get(match_column))
                    _append_reference_to_centroid_rows(
                        rows,
                        context=context,
                        descriptor=candidate_sample.descriptor,
                        vector=np.asarray(normalized[index], dtype=np.float32),
                        centroids=centroids,
                        centroid_labels=centroid_labels,
                        reference_set_id=reference_set_id,
                        reference_set_label=reference_set_label,
                        reference_entity_id=entity_id,
                        reference_entity_label=_reference_row_label(reference_set, row, entity_id),
                        reference_entity_type="row",
                        reference_set_status=status,
                        reference_set_complete=bool(resolution.complete),
                        reference_rows=len(indices),
                    )
            if aggregation in {"centroid", "both"} and len(indices) >= min_reference_group_size:
                centroid = try_l2_normalize_vector(np.asarray(normalized[indices].mean(axis=0), dtype=np.float32))
                if centroid is not None:
                    _append_reference_to_centroid_rows(
                        rows,
                        context=context,
                        descriptor=candidate_sample.descriptor,
                        vector=centroid,
                        centroids=centroids,
                        centroid_labels=centroid_labels,
                        reference_set_id=reference_set_id,
                        reference_set_label=reference_set_label,
                        reference_entity_id=reference_set_id,
                        reference_entity_label=reference_set_label,
                        reference_entity_type="reference_set_centroid",
                        reference_set_status=status,
                        reference_set_complete=bool(resolution.complete),
                        reference_rows=len(indices),
                    )
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {"candidate_count": len(candidates), "reference_set_count": len(reference_entries), "rows": len(rows)},
    )
