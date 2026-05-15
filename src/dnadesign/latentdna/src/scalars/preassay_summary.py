"""Summary pre-assay scalar builders."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError
from ..geometry.cohorts import (
    aligned_cohort_distance_vectors,
    balanced_group_indices,
    group_indices,
    resample_groups,
    separation_ratio_from_groups,
)
from ..io.json_io import read_json
from ..stats.rank import pearson_correlation
from ..workspaces.loader import WorkspaceContext
from .common import (
    ScalarInputRef,
    _candidate_descriptor_from_view,
    _effective_rank,
    _load_view_scope_table,
    _metric_row,
    _normalized_geometry_rows,
    _optional_param,
    _pairwise_cosine_distance_summary,
    _reducer_summary_path,
    _require_param,
)
from .preassay_common import (
    _REPRESENTATION_HEALTH_METRIC_IDS,
    ScalarBuilderResult,
    _bootstrap_ci_with_values,
    _cohort_metric_axes,
    _load_candidate_sample,
)


def _representation_health_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    omitted_candidates = [dict(value) for value in _optional_param(params, "omitted_candidates", default=[])]
    collapse_rules = {
        str(key): float(value) for key, value in dict(_optional_param(params, "collapse_rules", default={})).items()
    }
    pairwise_max_rows = int(_optional_param(params, "pairwise_max_rows", default=4096))
    pairwise_seed = int(_optional_param(params, "pairwise_seed", default=17))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        reducer_id = str(_require_param(candidate, "reducer_id"))
        inputs.extend(candidate_sample.inputs)
        reducer_path = _reducer_summary_path(context, reducer_id)
        inputs.append(ScalarInputRef(kind="reducer", artifact_id=reducer_id, path=reducer_path))
        reducer_summary = read_json(reducer_path)
        explained = [float(value) for value in reducer_summary.get("explained_variance_ratio", [])]
        explained_variance_captured = float(sum(explained))
        pc1_fraction = (
            float(explained[0]) / explained_variance_captured
            if explained and explained_variance_captured > 0.0
            else float("nan")
        )
        distance_summary = _pairwise_cosine_distance_summary(
            candidate_sample.matrix,
            max_rows=pairwise_max_rows,
            seed=pairwise_seed,
        )
        effective_rank = _effective_rank(explained)
        failures = sum(
            [
                effective_rank < float(collapse_rules.get("effective_rank_min", 2.0)),
                pc1_fraction > float(collapse_rules.get("pc1_fraction_max", 0.80)),
                distance_summary.iqr < float(collapse_rules.get("pairwise_distance_iqr_min", 0.01)),
            ]
        )
        health_status = "fail" if failures >= 2 else "warn" if failures == 1 else "pass"
        extra = {
            "health_status": health_status,
            "collapse_flag": health_status != "pass",
            "effective_rank_basis": "retained_pca_components",
            "effective_rank_component_count": len([value for value in explained if value > 0.0]),
            "explained_variance_captured": explained_variance_captured,
            "pca_fit_rows": int(reducer_summary.get("fit_rows", 0) or 0),
            "pca_input_dims": int(reducer_summary.get("input_dims", 0) or 0),
            "pca_output_dims": int(reducer_summary.get("output_dims", len(explained)) or len(explained)),
            "pca_fit_scope_kind": str(reducer_summary.get("scope_kind") or ""),
            "pca_fit_scope_id": str(reducer_summary.get("scope_id") or ""),
            "pca_method": str(reducer_summary.get("pca_method") or reducer_summary.get("method") or ""),
            "pairwise_distance_method": distance_summary.method,
            "pairwise_distance_source_rows": distance_summary.source_rows,
            "pairwise_distance_evaluated_rows": distance_summary.evaluated_rows,
            "pairwise_distance_pair_count": distance_summary.pair_count,
            "pairwise_distance_max_rows": distance_summary.max_rows,
            "pairwise_distance_seed": distance_summary.seed,
            "candidate_status": "materialized",
            "candidate_materialized": True,
            "omitted_from_ranking": False,
            "omission_reason": "",
        }
        rows.extend(
            [
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id="effective_rank",
                    metric_value=effective_rank,
                    extra=extra,
                ),
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id="pc1_variance_fraction",
                    metric_value=pc1_fraction,
                    extra=extra,
                ),
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id="pairwise_cosine_distance_median",
                    metric_value=distance_summary.median,
                    extra=extra,
                ),
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id="pairwise_cosine_distance_iqr",
                    metric_value=distance_summary.iqr,
                    extra=extra,
                ),
            ]
        )

    for candidate in omitted_candidates:
        view_id = str(_require_param(candidate, "view_id"))
        status = _omitted_candidate_status(context, candidate, view_id=view_id)
        reason = str(_optional_param(candidate, "reason", default=status) or status)
        descriptor = _candidate_descriptor_from_view(context, view_id=view_id)
        extra = {
            "health_status": status,
            "collapse_flag": False,
            "effective_rank_basis": "unavailable",
            "effective_rank_component_count": 0,
            "explained_variance_captured": float("nan"),
            "pca_fit_rows": 0,
            "pca_input_dims": 0,
            "pca_output_dims": 0,
            "pca_fit_scope_kind": "",
            "pca_fit_scope_id": "",
            "pca_method": "",
            "pairwise_distance_method": "unavailable",
            "pairwise_distance_source_rows": 0,
            "pairwise_distance_evaluated_rows": 0,
            "pairwise_distance_pair_count": 0,
            "pairwise_distance_max_rows": pairwise_max_rows,
            "pairwise_distance_seed": pairwise_seed,
            "candidate_status": status,
            "candidate_materialized": False,
            "omitted_from_ranking": True,
            "omission_reason": reason,
        }
        rows.extend(
            _metric_row(
                context=context,
                descriptor=descriptor,
                metric_id=metric_id,
                metric_value=float("nan"),
                extra=extra,
            )
            for metric_id in _REPRESENTATION_HEALTH_METRIC_IDS
        )

    return (
        pa.Table.from_pylist(rows),
        inputs,
        {
            "candidate_count": len(candidates) + len(omitted_candidates),
            "ranked_candidate_count": len(candidates),
            "omitted_candidate_count": len(omitted_candidates),
            "pairwise_max_rows": pairwise_max_rows,
            "rows": len(rows),
        },
    )


def _omitted_candidate_status(context: WorkspaceContext, candidate: dict[str, Any], *, view_id: str) -> str:
    explicit_status = str(_optional_param(candidate, "status", default="") or "").strip().lower()
    if explicit_status:
        return explicit_status
    role = str(getattr(context.require_view(view_id), "role", "") or "").strip().lower()
    return role or "unavailable"


def _design_structure_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    bootstrap_iterations = int(_optional_param(params, "bootstrap_iterations", default=200))
    seed = int(_optional_param(params, "seed", default=context.config.defaults.random_seed))
    axes = _cohort_metric_axes(params, key="axes", builder_kind="design_structure_summary")
    balanced_axis = _optional_param(params, "balanced_axis", default=None)
    if balanced_axis is not None and not isinstance(balanced_axis, dict):
        raise ContractViolationError("design_structure_summary balanced_axis must be a mapping when provided")
    balanced_axis = dict(balanced_axis or {})
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for offset, candidate in enumerate(candidates):
        rng = np.random.default_rng(seed + offset)
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        for axis in axes:
            groups = group_indices(candidate_sample.rows, column=axis.column, exclude_values=axis.exclude_values)
            value = separation_ratio_from_groups(normalized, groups)
            ci_lower, ci_upper, bootstrap_replicates = _bootstrap_ci_with_values(
                lambda groups=groups, rng=rng: separation_ratio_from_groups(
                    normalized,
                    resample_groups(groups, rng=rng),
                ),
                iterations=bootstrap_iterations,
            )
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id=axis.metric_id,
                    metric_value=value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    category=axis.axis_id,
                    extra={
                        "cohort_axis_id": axis.axis_id,
                        "cohort_column": axis.column,
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_iterations": bootstrap_iterations,
                        **({"display_name": axis.display_name} if axis.display_name is not None else {}),
                    },
                )
            )

        if balanced_axis:
            balanced_column = str(_require_param(balanced_axis, "column"))
            balanced_metric_id = str(_require_param(balanced_axis, "metric_id"))
            balance_columns = [str(value) for value in _require_param(balanced_axis, "balance_columns")]
            balanced_groups = balanced_group_indices(
                candidate_sample.rows,
                group_column=balanced_column,
                balance_columns=balance_columns,
                required_group_values={
                    str(value) for value in _optional_param(balanced_axis, "required_group_values", default=[])
                }
                or None,
                exclude_group_values={
                    str(value) for value in _optional_param(balanced_axis, "exclude_values", default=[])
                }
                or None,
                rng=rng,
            )
            balanced_value = separation_ratio_from_groups(normalized, balanced_groups)
            ci_lower, ci_upper, bootstrap_replicates = _bootstrap_ci_with_values(
                lambda rng=rng: separation_ratio_from_groups(
                    normalized,
                    balanced_group_indices(
                        candidate_sample.rows,
                        group_column=balanced_column,
                        balance_columns=balance_columns,
                        required_group_values={
                            str(value) for value in _optional_param(balanced_axis, "required_group_values", default=[])
                        }
                        or None,
                        exclude_group_values={
                            str(value) for value in _optional_param(balanced_axis, "exclude_values", default=[])
                        }
                        or None,
                        rng=rng,
                    ),
                ),
                iterations=bootstrap_iterations,
            )
            display_name = str(
                _optional_param(balanced_axis, "display_name", default=_optional_param(balanced_axis, "label")) or ""
            ).strip()
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id=balanced_metric_id,
                    metric_value=balanced_value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    category=str(_optional_param(balanced_axis, "axis_id", default=balanced_column) or balanced_column),
                    extra={
                        "cohort_axis_id": str(
                            _optional_param(balanced_axis, "axis_id", default=balanced_column) or balanced_column
                        ),
                        "cohort_column": balanced_column,
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_iterations": bootstrap_iterations,
                        **({"display_name": display_name} if display_name else {}),
                    },
                )
            )
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {"candidate_count": len(candidates), "axis_count": len(axes) + int(bool(balanced_axis)), "rows": len(rows)},
    )


def _cohort_structure_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    axes = [dict(value) for value in _require_param(params, "axes")]
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    skipped_axes: list[str] = []
    for candidate in candidates:
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        for axis in axes:
            column = str(_require_param(axis, "column"))
            axis_id = str(_optional_param(axis, "axis_id", default=column) or column)
            label = str(_optional_param(axis, "label", default=axis_id) or axis_id)
            min_group_size = int(_optional_param(axis, "min_group_size", default=2))
            exclude_values = _optional_param(axis, "exclude_values", default=None)
            allowed_values = _optional_param(axis, "allowed_values", default=None)
            groups = group_indices(
                candidate_sample.rows,
                column=column,
                exclude_values={str(value) for value in exclude_values} if exclude_values else None,
                allowed_values={str(value) for value in allowed_values} if allowed_values else None,
            )
            groups = {key: value for key, value in groups.items() if len(value) >= min_group_size}
            usable_row_count = sum(len(value) for value in groups.values())
            if len(groups) < 2:
                skipped_axes.append(axis_id)
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id="cohort_separation_ratio",
                    metric_value=separation_ratio_from_groups(normalized, groups),
                    category=axis_id,
                    extra={
                        "display_name": label,
                        "cohort_axis_id": axis_id,
                        "cohort_column": column,
                        "cohort_group_count": len(groups),
                        "cohort_usable_row_count": usable_row_count,
                        "cohort_min_group_size": min_group_size,
                    },
                )
            )
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {
            "candidate_count": len(candidates),
            "axis_count": len(axes),
            "rows": len(rows),
            "skipped_axes": sorted(set(skipped_axes)),
        },
    )


def _context_robustness_summary_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    pairs = [dict(value) for value in _require_param(params, "pairs")]
    sample_size = int(_optional_param(params, "sample_size", default=4096))
    sample_group_column = _optional_param(params, "sample_group_column", default="design_family")
    seed = int(_optional_param(params, "seed", default=context.config.defaults.random_seed))
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    skipped_metric_ids: list[str] = []
    axes = _cohort_metric_axes(params, key="retention_axes", builder_kind="context_robustness_summary")
    for offset, pair in enumerate(pairs):
        alignment_id = str(_require_param(pair, "alignment_id"))
        left_view_id = str(_require_param(pair, "anchor_view_id"))
        right_view_id = str(_require_param(pair, "context_view_id"))
        pair_id = str(_optional_param(pair, "pair_id", default=f"{left_view_id}_to_{right_view_id}"))
        descriptor = _candidate_descriptor_from_view(
            context,
            view_id=left_view_id,
            candidate_id=pair_id,
            scope_override="anchor_vs_context",
            label_override=_optional_param(pair, "label", default=None),
        )
        left_matrix, _, left_inputs = _load_view_scope_table(
            context,
            view_id=left_view_id,
            alignment_id=alignment_id,
        )
        right_matrix, metadata_rows, right_inputs = _load_view_scope_table(
            context,
            view_id=right_view_id,
            alignment_id=alignment_id,
        )
        inputs.extend(left_inputs)
        inputs.extend(right_inputs)
        if left_matrix.shape != right_matrix.shape:
            raise ContractViolationError("context robustness summary requires aligned anchor/context matrices")
        if sample_size > 0 and sample_size < len(metadata_rows):
            sampled_indices = _sample_metadata_indices(
                metadata_rows,
                sample_size=sample_size,
                group_column=sample_group_column,
                seed=seed + offset,
            )
            index_array = np.asarray(sampled_indices, dtype=np.int64)
            left_matrix = left_matrix[index_array]
            right_matrix = right_matrix[index_array]
            metadata_rows = [metadata_rows[index] for index in sampled_indices]
        left_norm = _normalized_geometry_rows(left_matrix)
        right_norm = _normalized_geometry_rows(right_matrix)
        self_cosine = np.asarray(np.sum(left_norm * right_norm, axis=1), dtype=np.float64)
        rows.append(
            _metric_row(
                context=context,
                descriptor=descriptor,
                metric_id="context_self_cosine_median",
                metric_value=float(np.median(self_cosine)),
            )
        )
        for axis in axes:
            anchor_vector, context_vector = aligned_cohort_distance_vectors(
                left_norm,
                right_norm,
                metadata_rows,
                column=axis.column,
                exclude_values=axis.exclude_values,
            )
            if anchor_vector.size == 0 or context_vector.size == 0:
                skipped_metric_ids.append(axis.metric_id)
                continue
            retention = pearson_correlation(anchor_vector, context_vector)
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=descriptor,
                    metric_id=axis.metric_id,
                    metric_value=retention,
                    category=axis.axis_id,
                    extra={
                        "cohort_axis_id": axis.axis_id,
                        "cohort_column": axis.column,
                        **({"display_name": axis.display_name} if axis.display_name is not None else {}),
                    },
                )
            )
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {
            "pair_count": len(pairs),
            "rows": len(rows),
            "skipped_metric_ids": skipped_metric_ids,
        },
    )


def _sample_metadata_indices(
    rows: list[dict[str, object]],
    *,
    sample_size: int,
    group_column: str | None,
    seed: int,
) -> list[int]:
    row_count = len(rows)
    if sample_size >= row_count:
        return list(range(row_count))
    rng = np.random.default_rng(seed)
    if group_column is None:
        return sorted(rng.choice(row_count, size=sample_size, replace=False).tolist())
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = str(row.get(group_column))
        groups.setdefault(key, []).append(index)
    total_rows = sum(len(indices) for indices in groups.values())
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for key, indices in groups.items():
        raw = (len(indices) / total_rows) * sample_size
        count = min(len(indices), int(raw))
        quotas[key] = count
        assigned += count
        remainders.append((raw - count, key))
    for _, key in sorted(remainders, reverse=True):
        if assigned >= sample_size:
            break
        if quotas[key] >= len(groups[key]):
            continue
        quotas[key] += 1
        assigned += 1
    selected: list[int] = []
    for key in sorted(groups):
        candidates = np.asarray(groups[key], dtype=np.int64)
        order = rng.permutation(len(candidates))
        selected.extend(sorted(candidates[order][: quotas[key]].tolist()))
    return sorted(selected)
