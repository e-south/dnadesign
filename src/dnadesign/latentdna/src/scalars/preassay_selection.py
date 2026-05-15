"""Selection and ordinal-detail pre-assay scalar builders."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError
from ..geometry.cohorts import centroid_map, group_indices, ordinal_gap_and_distance_vectors
from ..geometry.preprocessing import try_l2_normalize_vector
from ..labels import humanize_label
from ..metadata_axes import axis_display_text
from ..stats.rank import spearman_correlation
from ..workspaces.loader import WorkspaceContext
from .common import (
    ScalarInputRef,
    _candidate_descriptor_from_view,
    _metric_row,
    _normalized_geometry_rows,
    _optional_param,
    _require_param,
)
from .preassay_common import (
    ScalarBuilderResult,
    _load_candidate_sample,
    _load_scalar_rows,
    _rows_by_candidate_and_metric,
)
from .preassay_ordinal import (
    _ORDINAL_AXIS_DEFAULT_METRIC_IDS,
    _axis_style_for_column,
    _coerce_float,
    _ordinal_axis_extra,
    _ordinal_global_statistics,
    _OrdinalAxis,
    _resolve_numeric_ordinal_axis,
    _resolve_ordinal_axis,
)


def _collection_strength_ordinal_audit_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    collection_column = str(_require_param(params, "collection_column"))
    group_column = str(_require_param(params, "group_column"))
    rank_column = str(_require_param(params, "rank_column"))
    collections = [dict(value) for value in _require_param(params, "collections")]
    metric_ids = dict(_ORDINAL_AXIS_DEFAULT_METRIC_IDS)
    configured_metric_ids = dict(_optional_param(params, "metric_ids", default={}) or {})
    metric_ids.update({str(key): str(value) for key, value in configured_metric_ids.items()})
    permutations = int(_optional_param(params, "permutations", default=200))
    seed = int(_optional_param(params, "seed", default=context.config.defaults.random_seed))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for candidate_offset, candidate in enumerate(candidates):
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        for collection_offset, collection in enumerate(collections):
            collection_id = str(_require_param(collection, "collection_id"))
            collection_label = str(_optional_param(collection, "label", default=humanize_label(collection_id)))
            collection_filters = [dict(value) for value in _optional_param(collection, "where", default=[])]
            collection_indices = [
                index
                for index, row in enumerate(candidate_sample.rows)
                if str(row.get(collection_column) or "") == collection_id
                and _row_matches_filters(row, collection_filters)
                and str(row.get(group_column) or "").strip()
                and _coerce_float(row.get(rank_column)) is not None
            ]
            if len(collection_indices) < 3:
                continue
            collection_rows = [candidate_sample.rows[index] for index in collection_indices]
            collection_matrix = normalized[np.asarray(collection_indices, dtype=np.int64)]
            axis = _resolve_numeric_ordinal_axis(
                axis_id=f"{collection_id}_strength",
                axis={
                    "axis_id": f"{collection_id}_strength",
                    "label": f"{collection_label} strength",
                    "rank_column": rank_column,
                },
                rows=collection_rows,
                group_column=group_column,
                exclude_values=set(),
            )
            axis_extra = {
                **_ordinal_axis_extra(axis),
                "reference_collection_id": collection_id,
                "reference_collection_label": collection_label,
                "reference_collection_column": collection_column,
            }
            spearman, kendall = _ordinal_global_statistics(collection_matrix, collection_rows, axis=axis)
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id=metric_ids["spearman"],
                    metric_value=spearman,
                    extra={**axis_extra, "display_name": f"{collection_label} strength Spearman"},
                )
            )
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id=metric_ids["kendall"],
                    metric_value=kendall,
                    extra={**axis_extra, "display_name": f"{collection_label} strength Kendall"},
                )
            )
            rng = np.random.default_rng(seed + candidate_offset * 997 + collection_offset)
            groups = group_indices(collection_rows, column=axis.column, allowed_values=set(axis.ranks))
            observed = spearman
            permutation_values: list[float] = []
            variants = sorted(set(axis.ranks), key=str.casefold)
            if np.isfinite(observed) and len(variants) >= 3:
                centroids = centroid_map(collection_matrix, groups)
                for _ in range(permutations):
                    shuffled = rng.permutation([axis.ranks[variant] for variant in variants]).tolist()
                    shuffled_ranks = {variant: rank for variant, rank in zip(variants, shuffled, strict=True)}
                    gaps, distances = ordinal_gap_and_distance_vectors(centroids=centroids, ranks=shuffled_ranks)
                    if gaps.size:
                        permutation_values.append(spearman_correlation(gaps, distances))
            permutation_pvalue = (
                float(
                    (1 + np.sum(np.abs(np.asarray(permutation_values, dtype=np.float64)) >= abs(observed)))
                    / (len(permutation_values) + 1)
                )
                if permutation_values and np.isfinite(observed)
                else float("nan")
            )
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id=metric_ids["permutation_pvalue"],
                    metric_value=permutation_pvalue,
                    extra={**axis_extra, "display_name": f"{collection_label} strength permutation p-value"},
                )
            )
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {"candidate_count": len(candidates), "collection_count": len(collections), "rows": len(rows)},
    )


def _row_matches_filters(row: dict[str, object], filters: list[dict[str, Any]]) -> bool:
    for selector in filters:
        column = str(_require_param(selector, "column"))
        value = row.get(column)
        if "equals" in selector and str(value) != str(selector["equals"]):
            return False
        if "in_values" in selector and str(value) not in {str(item) for item in list(selector["in_values"])}:
            return False
        if "regex" in selector and re.search(str(selector["regex"]), str(value or "")) is None:
            return False
    return True


def _aggregate_values(values: list[float], aggregation: str) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if aggregation == "median":
        return float(np.median(array))
    if aggregation == "mean":
        return float(np.mean(array))
    if aggregation == "max":
        return float(np.max(array))
    if aggregation == "min":
        return float(np.min(array))
    raise ContractViolationError(f"candidate_x_selection_scorecard unsupported aggregation: {aggregation!r}")


def _scorecard_descriptor(
    context: WorkspaceContext,
    candidate_id: str,
    fallback_row: dict[str, object] | None,
) -> dict[str, object]:
    if candidate_id in context.config.views:
        return _candidate_descriptor_from_view(context, view_id=candidate_id)
    if fallback_row is not None:
        return {
            "candidate_id": candidate_id,
            "candidate_family": fallback_row.get("candidate_family", ""),
            "candidate_model": fallback_row.get("candidate_model", ""),
            "candidate_scope": fallback_row.get("candidate_scope", ""),
            "candidate_label": fallback_row.get("candidate_label", candidate_id),
        }
    return {
        "candidate_id": candidate_id,
        "candidate_family": "",
        "candidate_model": "",
        "candidate_scope": "",
        "candidate_label": candidate_id,
    }


def _candidate_x_selection_scorecard_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    metric_sources = [dict(value) for value in _optional_param(params, "metric_sources", default=[])]
    aggregate_sources = [dict(value) for value in _optional_param(params, "aggregate_sources", default=[])]
    candidate_ids = [str(value) for value in _optional_param(params, "candidate_ids", default=[])]
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    seen_inputs: set[tuple[str, str]] = set()
    descriptor_rows: dict[str, dict[str, object]] = {}

    def add_inputs(source_inputs: list[ScalarInputRef]) -> None:
        for source_input in source_inputs:
            key = (source_input.kind, source_input.artifact_id)
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            inputs.append(source_input)

    if not candidate_ids:
        discovered: list[str] = []
        for source_spec in metric_sources:
            source_rows, source_inputs = _load_scalar_rows(
                context,
                scalar_id=str(_require_param(source_spec, "scalar")),
            )
            add_inputs(source_inputs)
            for row in source_rows:
                candidate_id = str(row.get("candidate_id") or "").strip()
                if candidate_id and candidate_id not in discovered:
                    discovered.append(candidate_id)
                    descriptor_rows[candidate_id] = row
        candidate_ids = discovered

    for source_spec in metric_sources:
        source_rows, source_inputs = _load_scalar_rows(context, scalar_id=str(_require_param(source_spec, "scalar")))
        add_inputs(source_inputs)
        grouped = _rows_by_candidate_and_metric(source_rows)
        source_metric_id = str(_require_param(source_spec, "metric_id"))
        output_metric_id = str(_optional_param(source_spec, "output_metric_id", default=source_metric_id))
        scorecard_section = str(_optional_param(source_spec, "section", default="scorecard"))
        for candidate_id in candidate_ids:
            source_row = grouped.get(candidate_id, {}).get(source_metric_id)
            if source_row is None:
                continue
            descriptor_rows.setdefault(candidate_id, source_row)
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=_scorecard_descriptor(context, candidate_id, source_row),
                    metric_id=output_metric_id,
                    metric_value=float(source_row["metric_value"]),
                    category=scorecard_section,
                    extra={
                        "source_scalar": str(_require_param(source_spec, "scalar")),
                        "source_metric_id": source_metric_id,
                        "scorecard_section": scorecard_section,
                        "label": _scorecard_descriptor(context, candidate_id, source_row)["candidate_label"],
                    },
                )
            )

    for aggregate_spec in aggregate_sources:
        source_rows, source_inputs = _load_scalar_rows(context, scalar_id=str(_require_param(aggregate_spec, "scalar")))
        add_inputs(source_inputs)
        source_column = str(_optional_param(aggregate_spec, "source_column", default="metric_value"))
        output_metric_id = str(_require_param(aggregate_spec, "output_metric_id"))
        aggregation = str(_optional_param(aggregate_spec, "aggregation", default="median"))
        filters = [dict(value) for value in _optional_param(aggregate_spec, "where", default=[])]
        scorecard_section = str(_optional_param(aggregate_spec, "section", default="scorecard"))
        for candidate_id in candidate_ids:
            values = [
                float(row[source_column])
                for row in source_rows
                if str(row.get("candidate_id") or "") == candidate_id
                and source_column in row
                and _coerce_float(row.get(source_column)) is not None
                and _row_matches_filters(row, filters)
            ]
            source_row = next((row for row in source_rows if str(row.get("candidate_id") or "") == candidate_id), None)
            descriptor = _scorecard_descriptor(
                context,
                candidate_id,
                source_row or descriptor_rows.get(candidate_id),
            )
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=descriptor,
                    metric_id=output_metric_id,
                    metric_value=_aggregate_values(values, aggregation),
                    category=scorecard_section,
                    extra={
                        "source_scalar": str(_require_param(aggregate_spec, "scalar")),
                        "source_column": source_column,
                        "aggregation": aggregation,
                        "scorecard_section": scorecard_section,
                        "label": descriptor["candidate_label"],
                    },
                )
            )
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(candidate_ids), "rows": len(rows)}


def _context_pair_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    comparisons = [dict(value) for value in _require_param(params, "comparisons")]
    metric_specs = [
        ("context_self_cosine", "context_self_cosine_median"),
        ("context_shift_l2", "context_shift_l2_median"),
    ]
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for comparison in comparisons:
        scalar_id = str(_require_param(comparison, "scalar_id"))
        comparison_id = str(_require_param(comparison, "comparison_id"))
        comparison_label = str(_require_param(comparison, "comparison_label"))
        comparison_role = str(_require_param(comparison, "comparison_role"))
        source_rows, source_inputs = _load_scalar_rows(context, scalar_id=scalar_id)
        inputs.extend(source_inputs)
        for source_column, metric_id in metric_specs:
            values = [
                float(value)
                for row in source_rows
                if (value := row.get(source_column)) is not None and np.isfinite(float(value))
            ]
            metric_value = float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")
            rows.append(
                _metric_row(
                    context=context,
                    descriptor={
                        "comparison_id": comparison_id,
                        "comparison_label": comparison_label,
                        "comparison_role": comparison_role,
                    },
                    metric_id=metric_id,
                    metric_value=metric_value,
                    extra={
                        "label": comparison_label,
                        "source_scalar_id": scalar_id,
                    },
                )
            )
    return pa.Table.from_pylist(rows), inputs, {"comparison_count": len(comparisons), "rows": len(rows)}


def _candidate_decision_frontier_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    health_scalar = str(_require_param(params, "health_scalar"))
    design_scalar = str(_require_param(params, "design_scalar"))
    ordinal_scalar = str(_require_param(params, "ordinal_scalar"))
    context_scalar = str(_require_param(params, "context_scalar"))
    health_metric_id = str(_optional_param(params, "health_metric_id", default="effective_rank"))
    design_metric_id = str(
        _optional_param(params, "design_metric_id", default="design_family_balanced_separation_ratio")
    )
    ordinal_metric_id = str(_optional_param(params, "ordinal_metric_id", default="ordinal_axis_spearman"))
    ordinal_output_column = str(_optional_param(params, "ordinal_output_column", default=ordinal_metric_id))
    context_metric_id = str(_optional_param(params, "context_metric_id", default="context_self_cosine_median"))
    candidate_ids = [str(value) for value in _optional_param(params, "candidate_ids", default=[])]
    context_pairs = {
        str(_require_param(entry, "candidate_id")): str(_require_param(entry, "pair_id"))
        for entry in _optional_param(params, "context_pairs", default=[])
    }
    candidate_roles = {
        str(_require_param(entry, "candidate_id")): str(_require_param(entry, "role"))
        for entry in _optional_param(params, "candidate_roles", default=[])
    }
    annotation_labels = {
        str(_require_param(entry, "candidate_id")): str(_require_param(entry, "label"))
        for entry in _optional_param(params, "annotation_labels", default=[])
    }

    health_rows, health_inputs = _load_scalar_rows(context, scalar_id=health_scalar)
    design_rows, design_inputs = _load_scalar_rows(context, scalar_id=design_scalar)
    ordinal_rows, ordinal_inputs = _load_scalar_rows(context, scalar_id=ordinal_scalar)
    context_rows, context_inputs = _load_scalar_rows(context, scalar_id=context_scalar)
    inputs = [*health_inputs, *design_inputs, *ordinal_inputs, *context_inputs]

    health_map = _rows_by_candidate_and_metric(health_rows)
    design_map = _rows_by_candidate_and_metric(design_rows)
    ordinal_map = _rows_by_candidate_and_metric(ordinal_rows)
    context_map = _rows_by_candidate_and_metric(context_rows)

    ordered_candidate_ids = candidate_ids or list(health_map)
    rows: list[dict[str, object]] = []
    for index, candidate_id in enumerate(ordered_candidate_ids):
        health_metrics = health_map.get(candidate_id, {})
        descriptor_source = (
            health_metrics.get(health_metric_id)
            or next(iter(health_metrics.values()), None)
            or next(iter(design_map.get(candidate_id, {}).values()), None)
            or next(iter(ordinal_map.get(candidate_id, {}).values()), None)
        )
        if descriptor_source is None:
            raise ContractViolationError(f"candidate_decision_frontier is missing descriptor rows for {candidate_id!r}")
        context_pair_id = context_pairs.get(candidate_id)
        context_metric_row = (
            context_map.get(context_pair_id, {}).get(context_metric_id) if context_pair_id is not None else None
        )
        health_metric_row = health_metrics.get(health_metric_id)
        design_metric_row = design_map.get(candidate_id, {}).get(design_metric_id)
        ordinal_metric_row = ordinal_map.get(candidate_id, {}).get(ordinal_metric_id)
        row = {
            "candidate_id": candidate_id,
            "candidate_label": descriptor_source["candidate_label"],
            "candidate_family": descriptor_source["candidate_family"],
            "candidate_model": descriptor_source["candidate_model"],
            "candidate_scope": descriptor_source["candidate_scope"],
            "candidate_order": index,
            "selection_role": candidate_roles.get(candidate_id, "candidate"),
            "frontier_label": annotation_labels.get(candidate_id, str(descriptor_source["candidate_label"])),
            "health_status": str(descriptor_source.get("health_status") or "unknown"),
            "collapse_flag": bool(descriptor_source.get("collapse_flag", False)),
            "effective_rank": (
                float(health_metric_row["metric_value"]) if health_metric_row is not None else float("nan")
            ),
            "design_family_balanced_separation_ratio": (
                float(design_metric_row["metric_value"]) if design_metric_row is not None else float("nan")
            ),
            ordinal_output_column: (
                float(ordinal_metric_row["metric_value"]) if ordinal_metric_row is not None else float("nan")
            ),
            "context_self_cosine_median": (
                float(context_metric_row["metric_value"]) if context_metric_row is not None else float("nan")
            ),
            "context_pair_id": context_pair_id,
            "context_validation_status": "direct" if context_pair_id is not None else "not_applicable",
            "x_display_name": "Balanced design-family separation ratio",
            "y_display_name": str(_optional_param(params, "ordinal_display_name", default="Ordinal-axis Spearman")),
        }
        rows.append(row)
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(rows), "rows": len(rows)}


def _ordinal_extreme_values(axis: _OrdinalAxis, *, stronger_rank: str) -> tuple[list[str], list[str]]:
    if stronger_rank not in {"min", "max"}:
        raise ContractViolationError("ordinal_ladder_rows stronger_rank must be 'min' or 'max'")
    rank_values = list(axis.ranks.values())
    if not rank_values:
        raise ContractViolationError(f"ordinal axis {axis.axis_id!r} has no rank values")
    strong_rank = min(rank_values) if stronger_rank == "min" else max(rank_values)
    weak_rank = max(rank_values) if stronger_rank == "min" else min(rank_values)
    target_values = [value for value, rank in axis.ranks.items() if rank == strong_rank]
    control_values = [value for value, rank in axis.ranks.items() if rank == weak_rank]
    return target_values, control_values


def _ordinal_plot_order_map(axis: _OrdinalAxis, *, stronger_rank: str) -> dict[str, int]:
    rank_values = sorted(set(axis.ranks.values()), reverse=stronger_rank == "min")
    rank_to_order = {rank: index + 1 for index, rank in enumerate(rank_values)}
    return {value: rank_to_order[rank] for value, rank in axis.ranks.items()}


def _ordinal_reference_centroid(
    normalized: np.ndarray,
    groups: dict[str, list[int]],
    values: list[str],
    *,
    role: str,
) -> tuple[np.ndarray, int]:
    indices = sorted({index for value in values for index in groups.get(value, [])})
    if not indices:
        raise ContractViolationError(f"ordinal_ladder_rows {role} values matched no rows: {values}")
    centroid = try_l2_normalize_vector(
        np.asarray(normalized[np.asarray(indices, dtype=np.int64)].mean(axis=0), dtype=np.float32)
    )
    if centroid is None:
        raise ContractViolationError(f"ordinal_ladder_rows {role} centroid is degenerate for values: {values}")
    return centroid, len(indices)


def _ordinal_row_label(
    context: WorkspaceContext,
    *,
    row: dict[str, object],
    axis: _OrdinalAxis,
    group_config: dict[str, Any],
    axis_value: str,
) -> str:
    label_column = str(_optional_param(group_config, "label_column", default="") or "")
    if label_column:
        label = str(row.get(label_column) or "").strip()
        if label:
            return label
    label_overrides = {
        str(key): str(value)
        for key, value in dict(_optional_param(group_config, "label_overrides", default={}) or {}).items()
    }
    if axis_value in label_overrides:
        return label_overrides[axis_value]
    core60_label = re.sub(r"_core60(?:_context1kb_(?:forward|rc))?$", "", axis_value)
    if core60_label != axis_value:
        return core60_label
    return axis_display_text(_axis_style_for_column(context, axis.column), axis_value, compact=True)


def _ordinal_ladder_rows_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    group_configs = [dict(value) for value in _require_param(params, "groups")]
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    group_counts: dict[str, int] = {}
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        descriptor = candidate_sample.descriptor
        for group_config in group_configs:
            group_id = str(_require_param(group_config, "group_id"))
            group_label = str(_optional_param(group_config, "label", default=humanize_label(group_id)))
            filters = [dict(value) for value in _optional_param(group_config, "where", default=[])]
            filtered_pairs = [
                (index, row) for index, row in enumerate(candidate_sample.rows) if _row_matches_filters(row, filters)
            ]
            if not filtered_pairs:
                raise ContractViolationError(f"ordinal_ladder_rows group {group_id!r} matched no rows")
            axis_config = dict(_require_param(group_config, "axis"))
            stronger_rank = str(_optional_param(axis_config, "stronger_rank", default="max"))
            axis = _resolve_ordinal_axis(
                context,
                axis=axis_config,
                rows=[row for _, row in filtered_pairs],
            )
            if axis.input_ref is not None and axis.input_ref not in inputs:
                inputs.append(axis.input_ref)
            groups: dict[str, list[int]] = {}
            row_records: list[tuple[int, dict[str, object], str]] = []
            for index, row in filtered_pairs:
                axis_value = str(row.get(axis.column) or "").strip()
                if not axis_value or axis_value in axis.exclude_values or axis_value not in axis.ranks:
                    continue
                groups.setdefault(axis_value, []).append(index)
                row_records.append((index, row, axis_value))
            if len(groups) < 2:
                raise ContractViolationError(
                    f"ordinal_ladder_rows group {group_id!r} requires at least two ranked classes"
                )
            target_values = [str(value) for value in _optional_param(group_config, "target_values", default=[]) or []]
            control_values = [str(value) for value in _optional_param(group_config, "control_values", default=[]) or []]
            if not target_values or not control_values:
                target_values, control_values = _ordinal_extreme_values(axis, stronger_rank=stronger_rank)
            target_reference, target_members = _ordinal_reference_centroid(
                normalized,
                groups,
                target_values,
                role="target",
            )
            control_reference, control_members = _ordinal_reference_centroid(
                normalized,
                groups,
                control_values,
                role="control",
            )
            plot_order = _ordinal_plot_order_map(axis, stronger_rank=stronger_rank)
            for index, row, axis_value in row_records:
                vector = np.asarray(normalized[index], dtype=np.float32)
                target_similarity = float(vector @ target_reference)
                control_similarity = float(vector @ control_reference)
                ordinal_margin = target_similarity - control_similarity
                rows.append(
                    {
                        **row,
                        **descriptor,
                        "ordinal_group_id": group_id,
                        "ordinal_group_label": group_label,
                        "ordinal_axis_id": axis.axis_id,
                        "ordinal_axis_label": axis.label,
                        "ordinal_axis_column": axis.column,
                        "ordinal_axis_value": axis_value,
                        "ordinal_label": _ordinal_row_label(
                            context,
                            row=row,
                            axis=axis,
                            group_config=group_config,
                            axis_value=axis_value,
                        ),
                        "ordinal_rank_value": float(axis.ranks[axis_value]),
                        "ordinal_plot_order": int(plot_order[axis_value]),
                        "ordinal_margin": ordinal_margin,
                        "ordinal_target_values": ",".join(target_values),
                        "ordinal_control_values": ",".join(control_values),
                        "ordinal_target_similarity": target_similarity,
                        "ordinal_control_similarity": control_similarity,
                        "ordinal_stronger_rank": stronger_rank,
                        "ordinal_order_source": axis.order_source,
                        "ordinal_order_exploratory": axis.exploratory,
                        "ordinal_group_count": len(groups),
                        "ordinal_target_members": target_members,
                        "ordinal_control_members": control_members,
                    }
                )
            group_counts[group_id] = group_counts.get(group_id, 0) + len(row_records)
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {
            "candidate_count": len(candidates),
            "ordinal_group_count": len(group_configs),
            "rows": len(rows),
            "rows_by_group": group_counts,
        },
    )


def _axis_centroid_distance_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    axis_config = dict(_require_param(params, "axis"))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        axis = _resolve_ordinal_axis(context, axis=axis_config, rows=candidate_sample.rows)
        if axis.input_ref is not None and axis.input_ref not in inputs:
            inputs.append(axis.input_ref)
        style = _axis_style_for_column(context, axis.column)
        groups = group_indices(
            candidate_sample.rows,
            column=axis.column,
            exclude_values=axis.exclude_values,
        )
        unranked_values = sorted(set(groups) - set(axis.ranks), key=str.casefold)
        ordered_values = [
            value for value, _ in sorted(axis.ranks.items(), key=lambda item: float(item[1])) if value in groups
        ]
        ordered_values.extend(unranked_values)
        value_labels = {
            value: (
                f"{axis_display_text(style, value)} (unranked)"
                if value in unranked_values
                else axis_display_text(style, value)
            )
            for value in ordered_values
        }
        centroids = centroid_map(normalized, groups)
        for row_value in ordered_values:
            for column_value in ordered_values:
                value = float("nan")
                if row_value in centroids and column_value in centroids:
                    value = 1.0 - float(np.dot(centroids[row_value], centroids[column_value]))
                rows.append(
                    {
                        **candidate_sample.descriptor,
                        "row_axis_value": row_value,
                        "column_axis_value": column_value,
                        "row_variant": value_labels[row_value],
                        "column_variant": value_labels[column_value],
                        "metric_value": value,
                        **_ordinal_axis_extra(axis),
                    }
                )
    return pa.Table.from_pylist(rows), inputs, {"candidate_count": len(candidates), "rows": len(rows)}
