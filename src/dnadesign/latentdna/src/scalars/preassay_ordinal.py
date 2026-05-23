"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/scalars/preassay_ordinal.py

Ordinal pre-assay scalar builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError
from ..geometry.cohorts import (
    balanced_group_indices,
    centroid_map,
    group_indices,
    ordinal_gap_and_distance_vectors,
    resample_groups,
)
from ..io.json_io import read_json
from ..metadata.axes import AxisStyle, axis_style_map_from_config
from ..stats.rank import kendall_tau_b, spearman_correlation
from ..workspaces.loader import WorkspaceContext
from .common import (
    ScalarInputRef,
    _metric_row,
    _normalized_geometry_rows,
    _optional_param,
    _require_param,
    _workspace_input_path,
)
from .preassay_common import (
    _ORDINAL_AXIS_DEFAULT_METRIC_IDS,
    _ORDINAL_AXIS_DEFAULT_WITHIN_GROUP_METRIC_ID,
    ScalarBuilderResult,
    _bootstrap_ci_with_values,
    _load_candidate_sample,
    _OrdinalAxis,
)


def _coerce_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _resolve_ordinal_metric_ids(axis: dict[str, Any]) -> dict[str, str]:
    metric_ids = dict(_ORDINAL_AXIS_DEFAULT_METRIC_IDS)
    configured = _optional_param(axis, "metric_ids", default={})
    if configured:
        metric_ids.update({str(key): str(value) for key, value in dict(configured).items()})
    return metric_ids


def _load_ordinal_axis_order(
    context: WorkspaceContext,
    *,
    axis_id: str,
    relative_path: str,
) -> tuple[dict[str, float], str, bool, ScalarInputRef]:
    path = _workspace_input_path(context, relative_path)
    payload = read_json(path) if path.suffix == ".json" else None
    if payload is None:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ContractViolationError("ordinal axis order config must decode to a mapping")
    order = payload.get("order")
    if not isinstance(order, list) or not order:
        raise ContractViolationError("ordinal axis order config requires a non-empty order list")
    ranks: dict[str, float] = {}
    for entry in order:
        if not isinstance(entry, dict):
            raise ContractViolationError("ordinal axis order entries must be mappings")
        raw_value = entry.get("value", entry.get("variant_id"))
        value = str(raw_value or "").strip()
        if not value:
            raise ContractViolationError("ordinal axis order entries require value or variant_id")
        rank = _coerce_float(entry.get("rank"))
        if rank is None:
            raise ContractViolationError(f"ordinal axis order entry {value!r} requires a finite rank")
        ranks[value] = rank
    return (
        ranks,
        str(payload.get("source") or "").strip(),
        bool(payload.get("exploratory", False)),
        ScalarInputRef(kind="workspace_input", artifact_id=f"{axis_id}_order", path=path),
    )


def _resolve_numeric_ordinal_axis(
    *,
    axis_id: str,
    axis: dict[str, Any],
    rows: list[dict[str, object]],
    group_column: str,
    exclude_values: set[str],
) -> _OrdinalAxis:
    rank_column = str(_require_param(axis, "rank_column"))
    label = str(_optional_param(axis, "label", default=axis_id) or axis_id)
    grouped_values: dict[str, list[float]] = {}
    for row in rows:
        group_value = str(row.get(group_column) or "").strip()
        if not group_value or group_value in exclude_values:
            continue
        rank_value = _coerce_float(row.get(rank_column))
        if rank_value is None:
            continue
        grouped_values.setdefault(group_value, []).append(rank_value)
    ranks = {
        group_value: float(np.median(np.asarray(values, dtype=np.float64)))
        for group_value, values in grouped_values.items()
        if values
    }
    if len(ranks) < 2:
        raise ContractViolationError(
            f"ordinal axis {axis_id!r} requires at least two finite ranked groups from {rank_column!r}"
        )
    return _OrdinalAxis(
        axis_id=axis_id,
        label=label,
        column=group_column,
        exclude_values=exclude_values,
        ranks=ranks,
        order_source=rank_column,
        exploratory=bool(_optional_param(axis, "exploratory", default=True)),
    )


def _resolve_ordinal_axis(
    context: WorkspaceContext,
    *,
    axis: dict[str, Any],
    rows: list[dict[str, object]],
) -> _OrdinalAxis:
    axis_id = str(_require_param(axis, "axis_id"))
    label = str(_optional_param(axis, "label", default=axis_id) or axis_id)
    group_column = str(_require_param(axis, "column"))
    exclude_values = {str(value) for value in _optional_param(axis, "exclude_values", default=[])}
    order_path = _optional_param(axis, "order_path", default=None)
    rank_column = _optional_param(axis, "rank_column", default=None)
    if bool(order_path) == bool(rank_column):
        raise ContractViolationError("ordinal axis requires exactly one of order_path or rank_column")
    if rank_column:
        return _resolve_numeric_ordinal_axis(
            axis_id=axis_id,
            axis=axis,
            rows=rows,
            group_column=group_column,
            exclude_values=exclude_values,
        )
    ranks, order_source, exploratory, input_ref = _load_ordinal_axis_order(
        context,
        axis_id=axis_id,
        relative_path=str(order_path),
    )
    return _OrdinalAxis(
        axis_id=axis_id,
        label=label,
        column=group_column,
        exclude_values=exclude_values,
        ranks=ranks,
        order_source=order_source,
        exploratory=exploratory,
        input_ref=input_ref,
    )


def _ordinal_statistics_from_groups(
    matrix: np.ndarray,
    groups: dict[str, list[int]],
    *,
    ranks: dict[str, float],
) -> tuple[float, float]:
    if len(groups) < 3:
        return float("nan"), float("nan")
    centroids = centroid_map(matrix, groups)
    if len(centroids) < 3:
        return float("nan"), float("nan")
    gaps, distances = ordinal_gap_and_distance_vectors(centroids=centroids, ranks=ranks)
    if gaps.size == 0:
        return float("nan"), float("nan")
    return spearman_correlation(gaps, distances), kendall_tau_b(gaps, distances)


def _ordinal_global_statistics(
    matrix: np.ndarray,
    rows: list[dict[str, object]],
    *,
    axis: _OrdinalAxis,
) -> tuple[float, float]:
    groups = group_indices(
        rows,
        column=axis.column,
        exclude_values=axis.exclude_values,
        allowed_values=set(axis.ranks),
    )
    return _ordinal_statistics_from_groups(matrix, groups, ranks=axis.ranks)


def _ordinal_mean_statistic_from_outer_groups(
    matrix: np.ndarray,
    rows: list[dict[str, object]],
    *,
    outer_groups: dict[str, list[int]],
    axis: _OrdinalAxis,
) -> float:
    statistics: list[float] = []
    for _, outer_indices in outer_groups.items():
        outer_rows = [rows[index] for index in outer_indices]
        outer_matrix = matrix[np.asarray(outer_indices, dtype=np.int64)]
        spearman, _ = _ordinal_global_statistics(outer_matrix, outer_rows, axis=axis)
        if np.isfinite(spearman):
            statistics.append(float(spearman))
    if not statistics:
        return float("nan")
    return float(np.mean(np.asarray(statistics, dtype=np.float64)))


def _ordinal_axis_extra(axis: _OrdinalAxis) -> dict[str, object]:
    return {
        "ordinal_axis_id": axis.axis_id,
        "ordinal_axis_label": axis.label,
        "ordinal_axis_column": axis.column,
        "ordinal_order_source": axis.order_source,
        "ordinal_order_exploratory": axis.exploratory,
        "ordinal_ranked_group_count": len(axis.ranks),
    }


def _axis_style_for_column(context: WorkspaceContext, column: str) -> AxisStyle | None:
    return axis_style_map_from_config(context.config).get(column)


def _axis_metric_label(
    context: WorkspaceContext,
    *,
    column: str,
    metric_id: str,
    axis_config: dict[str, Any],
) -> str | None:
    configured = dict(_optional_param(axis_config, "metric_labels", default={}) or {})
    style = _axis_style_for_column(context, column)
    labels = {**(style.metric_labels if style is not None else {}), **{str(k): str(v) for k, v in configured.items()}}
    return labels.get(metric_id)


def _ordinal_axis_audit_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    candidates = [dict(value) for value in _require_param(params, "candidates")]
    axis_config = dict(_require_param(params, "axis"))
    metric_ids = _resolve_ordinal_metric_ids(axis_config)
    within_groups = [dict(value) for value in _optional_param(axis_config, "within_groups", default=[])]
    bootstrap_iterations = int(_optional_param(params, "bootstrap_iterations", default=200))
    permutations = int(_optional_param(params, "permutations", default=200))
    seed = int(_optional_param(params, "seed", default=context.config.defaults.random_seed))
    balance_columns = [
        str(value) for value in _optional_param(params, "balance_columns", default=["design_family", "spacer_length"])
    ]
    rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    for offset, candidate in enumerate(candidates):
        rng = np.random.default_rng(seed + offset)
        candidate_sample = _load_candidate_sample(context, candidate)
        inputs.extend(candidate_sample.inputs)
        normalized = _normalized_geometry_rows(candidate_sample.matrix)
        axis = _resolve_ordinal_axis(context, axis=axis_config, rows=candidate_sample.rows)
        if axis.input_ref is not None and axis.input_ref not in inputs:
            inputs.append(axis.input_ref)
        axis_extra = _ordinal_axis_extra(axis)

        def metric_label(metric_id: str) -> str | None:
            return _axis_metric_label(context, column=axis.column, metric_id=metric_id, axis_config=axis_config)

        spearman_display_name = metric_label(metric_ids["spearman"])
        kendall_display_name = metric_label(metric_ids["kendall"])
        global_spearman, global_kendall = _ordinal_global_statistics(normalized, candidate_sample.rows, axis=axis)
        global_groups = group_indices(
            candidate_sample.rows,
            column=axis.column,
            exclude_values=axis.exclude_values,
            allowed_values=set(axis.ranks),
        )
        ci_lower, ci_upper, bootstrap_replicates = _bootstrap_ci_with_values(
            lambda groups=global_groups, rng=rng: _ordinal_statistics_from_groups(
                normalized,
                resample_groups(groups, rng=rng),
                ranks=axis.ranks,
            )[0],
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                context=context,
                descriptor=candidate_sample.descriptor,
                metric_id=metric_ids["spearman"],
                metric_value=global_spearman,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    **axis_extra,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_iterations": bootstrap_iterations,
                    **({"display_name": spearman_display_name} if spearman_display_name else {}),
                },
            )
        )
        rows.append(
            _metric_row(
                context=context,
                descriptor=candidate_sample.descriptor,
                metric_id=metric_ids["kendall"],
                metric_value=global_kendall,
                extra={
                    **axis_extra,
                    **({"display_name": kendall_display_name} if kendall_display_name else {}),
                },
            )
        )

        balanced_groups = balanced_group_indices(
            candidate_sample.rows,
            group_column=axis.column,
            balance_columns=balance_columns,
            required_group_values=set(axis.ranks),
            exclude_group_values=axis.exclude_values,
            rng=rng,
        )
        balanced_spearman = float("nan")
        if balanced_groups:
            centroids = centroid_map(normalized, balanced_groups)
            gaps, distances = ordinal_gap_and_distance_vectors(centroids=centroids, ranks=axis.ranks)
            balanced_spearman = spearman_correlation(gaps, distances) if gaps.size else float("nan")
        ci_lower, ci_upper, bootstrap_replicates = _bootstrap_ci_with_values(
            lambda groups=balanced_groups, rng=rng: _ordinal_statistics_from_groups(
                normalized,
                resample_groups(groups, rng=rng),
                ranks=axis.ranks,
            )[0],
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                context=context,
                descriptor=candidate_sample.descriptor,
                metric_id=metric_ids["balanced_spearman"],
                metric_value=balanced_spearman,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    **axis_extra,
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_iterations": bootstrap_iterations,
                    **(
                        {"display_name": metric_label(metric_ids["balanced_spearman"])}
                        if metric_label(metric_ids["balanced_spearman"])
                        else {}
                    ),
                },
            )
        )

        for within_group in within_groups:
            outer_column = str(_require_param(within_group, "column"))
            metric_id = str(
                _optional_param(
                    within_group,
                    "metric_id",
                    default=_ORDINAL_AXIS_DEFAULT_WITHIN_GROUP_METRIC_ID,
                )
            )
            outer_exclude_values = {
                str(value)
                for value in _optional_param(within_group, "exclude_values", default=axis.exclude_values or [])
            }
            outer_groups = group_indices(
                candidate_sample.rows,
                column=outer_column,
                exclude_values=outer_exclude_values,
            )
            within_mean = _ordinal_mean_statistic_from_outer_groups(
                normalized,
                candidate_sample.rows,
                outer_groups=outer_groups,
                axis=axis,
            )
            ci_lower, ci_upper, bootstrap_replicates = _bootstrap_ci_with_values(
                lambda groups=outer_groups, rng=rng: _ordinal_mean_statistic_from_outer_groups(
                    normalized,
                    candidate_sample.rows,
                    outer_groups=resample_groups(groups, rng=rng),
                    axis=axis,
                ),
                iterations=bootstrap_iterations,
            )
            rows.append(
                _metric_row(
                    context=context,
                    descriptor=candidate_sample.descriptor,
                    metric_id=metric_id,
                    metric_value=within_mean,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    extra={
                        **axis_extra,
                        "ordinal_within_group_column": outer_column,
                        "bootstrap_replicates": bootstrap_replicates,
                        "bootstrap_iterations": bootstrap_iterations,
                        **({"display_name": metric_label(metric_id)} if metric_label(metric_id) else {}),
                    },
                )
            )

        observed = global_spearman
        permutation_values: list[float] = []
        variants = sorted(set(axis.ranks) - axis.exclude_values, key=str.casefold)
        if np.isfinite(observed) and len(variants) >= 3:
            centroids = centroid_map(normalized, global_groups)
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
                extra={
                    **axis_extra,
                    **(
                        {"display_name": metric_label(metric_ids["permutation_pvalue"])}
                        if metric_label(metric_ids["permutation_pvalue"])
                        else {}
                    ),
                },
            )
        )
    return (
        pa.Table.from_pylist(rows),
        inputs,
        {
            "candidate_count": len(candidates),
            "axis_id": str(_require_param(axis_config, "axis_id")),
            "rows": len(rows),
        },
    )


def _ordinal_axes_audit_table(
    context: WorkspaceContext,
    params: dict[str, Any],
) -> ScalarBuilderResult:
    axes = [dict(value) for value in _require_param(params, "axes")]
    tables: list[pa.Table] = []
    inputs: list[ScalarInputRef] = []
    input_keys: set[tuple[str, str, str]] = set()
    axis_ids: list[str] = []
    total_rows = 0
    for axis in axes:
        axis_params = dict(params)
        axis_params.pop("axes", None)
        axis_params["axis"] = axis
        table, axis_inputs, axis_stats = _ordinal_axis_audit_table(context, axis_params)
        tables.append(table)
        axis_ids.append(str(axis_stats.get("axis_id") or _require_param(axis, "axis_id")))
        total_rows += int(axis_stats.get("rows") or table.num_rows)
        for input_ref in axis_inputs:
            key = (input_ref.kind, input_ref.artifact_id, input_ref.path.as_posix())
            if key not in input_keys:
                inputs.append(input_ref)
                input_keys.add(key)
    return (
        pa.concat_tables(tables, promote_options="default") if tables else pa.Table.from_pylist([]),
        inputs,
        {
            "axis_ids": axis_ids,
            "axis_count": len(axis_ids),
            "rows": total_rows,
        },
    )
