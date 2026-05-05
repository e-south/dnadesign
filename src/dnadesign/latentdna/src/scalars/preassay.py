"""Pre-assay scalar builders for representation triage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError
from ..geometry.cohorts import (
    aligned_cohort_distance_vectors,
    balanced_group_indices,
    bootstrap_ci,
    centroid_map,
    group_indices,
    ordinal_gap_and_distance_vectors,
    resample_groups,
    separation_ratio_from_groups,
)
from ..geometry.preprocessing import try_l2_normalize_vector
from ..io.json_io import read_json
from ..io.parquet_io import write_table
from ..labels import humanize_label
from ..metadata_axes import AxisStyle, axis_display_text, axis_style_map_from_config
from ..reference_sets import resolve_reference_set_rows
from ..workspaces.loader import WorkspaceContext
from .common import (
    BuiltScalarArtifact,
    ScalarInputRef,
    _candidate_descriptor_from_view,
    _cosine_distance_upper_from_normalized,
    _effective_rank,
    _kendall_tau,
    _load_view_scope_table,
    _metric_row,
    _normalized_geometry_rows,
    _optional_param,
    _pairwise_cosine_distance_summary,
    _pearson_correlation,
    _reducer_summary_path,
    _require_param,
    _spearman_correlation,
    _workspace_input_path,
)

ScalarBuilderResult = tuple[pa.Table, list[ScalarInputRef], dict[str, object]]
ScalarTableBuilder = Callable[[WorkspaceContext, dict[str, Any]], ScalarBuilderResult]


_REFERENCE_GROUP_COLUMN_LABELS = {
    "source_family": "Src",
    "selection_basis": "Basis",
    "promoter_standard__collection_id": "Std",
}

_REFERENCE_GROUP_VALUE_LABELS = {
    "anderson_igem": "Anderson iGEM",
    "archive_backed_insert": "Archive Insert",
    "construct_derived": "Construct-Derived",
    "legacy_construct_seed": "Legacy Seed",
    "legacy_reference_control": "Legacy Reference",
    "native_source_length": "Native Length",
    "reference_source": "Reference Source",
    "sfxi_archive": "SFXI Archive",
    "sigma_site_pair_midpoint": "Sigma Midpoint",
    "t7_w_collection": "T7 W collection",
    "template_window_center": "Template Window",
}

_REFERENCE_GROUP_METRIC_LABELS = {
    "reference_group_size": "Reference group size",
    "reference_group_pairwise_cosine_distance_median": "Reference group median distance",
    "reference_group_pairwise_cosine_distance_iqr": "Reference group distance IQR",
}


@dataclass(frozen=True, slots=True)
class _CandidateSample:
    descriptor: dict[str, object]
    matrix: np.ndarray
    rows: list[dict[str, object]]
    inputs: list[ScalarInputRef]


@dataclass(frozen=True, slots=True)
class _OrdinalAxis:
    axis_id: str
    label: str
    column: str
    exclude_values: set[str]
    ranks: dict[str, float]
    order_source: str
    exploratory: bool
    input_ref: ScalarInputRef | None = None


@dataclass(frozen=True, slots=True)
class _CohortMetricAxis:
    axis_id: str
    column: str
    metric_id: str
    exclude_values: set[str]
    display_name: str | None = None


_REPRESENTATION_HEALTH_METRIC_IDS = (
    "effective_rank",
    "pc1_variance_fraction",
    "pairwise_cosine_distance_median",
    "pairwise_cosine_distance_iqr",
)

_ORDINAL_AXIS_DEFAULT_METRIC_IDS = {
    "spearman": "ordinal_axis_spearman",
    "kendall": "ordinal_axis_kendall",
    "balanced_spearman": "ordinal_axis_balanced_spearman",
    "permutation_pvalue": "ordinal_axis_label_permutation_pvalue",
}

_ORDINAL_AXIS_DEFAULT_WITHIN_GROUP_METRIC_ID = "ordinal_axis_within_group_mean_spearman"


def _cohort_metric_axes(params: dict[str, Any], *, key: str, builder_kind: str) -> list[_CohortMetricAxis]:
    raw_axes = params.get(key)
    if not isinstance(raw_axes, list) or not raw_axes:
        raise ContractViolationError(f"{builder_kind} requires a non-empty {key!r} axis list")
    axes: list[_CohortMetricAxis] = []
    for index, raw_axis in enumerate(raw_axes):
        if not isinstance(raw_axis, dict):
            raise ContractViolationError(f"{builder_kind} {key}[{index}] must be a mapping")
        column = str(raw_axis.get("column") or "").strip()
        metric_id = str(raw_axis.get("metric_id") or "").strip()
        if not column or not metric_id:
            raise ContractViolationError(f"{builder_kind} {key}[{index}] requires column and metric_id")
        axis_id = str(raw_axis.get("axis_id") or column).strip()
        display_name = str(raw_axis.get("display_name") or raw_axis.get("label") or "").strip() or None
        axes.append(
            _CohortMetricAxis(
                axis_id=axis_id,
                column=column,
                metric_id=metric_id,
                exclude_values={str(value) for value in raw_axis.get("exclude_values", [])},
                display_name=display_name,
            )
        )
    return axes


def _load_candidate_sample(
    context: WorkspaceContext,
    candidate: dict[str, Any],
) -> _CandidateSample:
    view_id = str(_require_param(candidate, "view_id"))
    sample_id = _optional_param(candidate, "sample_id", default=None)
    descriptor = _candidate_descriptor_from_view(context, view_id=view_id)
    matrix, rows, inputs = _load_view_scope_table(context, view_id=view_id, sample_id=sample_id)
    return _CandidateSample(
        descriptor=descriptor,
        matrix=matrix,
        rows=rows,
        inputs=inputs,
    )


def _reference_group_label(value: object) -> str:
    text = " ".join(str(value or "").replace("__", " ").replace("_", " ").split()).strip()
    if not text:
        return ""
    words = []
    for word in text.split(" "):
        lowered = word.lower()
        if lowered in {"igem", "t7", "w"}:
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


def _reference_group_panel_title(*, metric_id: str, group_column: str, group_value: str) -> str:
    metric_label = _REFERENCE_GROUP_METRIC_LABELS.get(metric_id, _reference_group_label(metric_id))
    column_label = _REFERENCE_GROUP_COLUMN_LABELS.get(group_column, _reference_group_label(group_column))
    value_label = _REFERENCE_GROUP_VALUE_LABELS.get(group_value, _reference_group_label(group_value))
    return f"{metric_label}\n{column_label}: {value_label}"


def _reference_set_panel_title(*, metric_id: str, reference_set_label: str) -> str:
    metric_label = _REFERENCE_GROUP_METRIC_LABELS.get(metric_id, humanize_label(metric_id))
    return f"{metric_label}\nReference set: {reference_set_label}"


def _load_scalar_rows(
    context: WorkspaceContext,
    *,
    scalar_id: str,
) -> tuple[list[dict[str, object]], list[ScalarInputRef]]:
    path = context.output_root / "scalars" / scalar_id / "table.parquet"
    if not path.is_file():
        raise ContractViolationError(f"pre-assay scalar source is missing: {scalar_id}")
    table = pq.read_table(path)
    return table.to_pylist(), [ScalarInputRef(kind="scalar_table", artifact_id=scalar_id, path=path)]


def _rows_by_candidate_and_metric(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        metric_id = str(row.get("metric_id") or "").strip()
        if not candidate_id or not metric_id:
            continue
        grouped.setdefault(candidate_id, {})[metric_id] = row
    return grouped


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
                    descriptor=candidate_sample.descriptor,
                    metric_id="effective_rank",
                    metric_value=effective_rank,
                    extra=extra,
                ),
                _metric_row(
                    descriptor=candidate_sample.descriptor,
                    metric_id="pc1_variance_fraction",
                    metric_value=pc1_fraction,
                    extra=extra,
                ),
                _metric_row(
                    descriptor=candidate_sample.descriptor,
                    metric_id="pairwise_cosine_distance_median",
                    metric_value=distance_summary.median,
                    extra=extra,
                ),
                _metric_row(
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
            ci_lower, ci_upper = bootstrap_ci(
                lambda groups=groups, rng=rng: separation_ratio_from_groups(
                    normalized,
                    resample_groups(groups, rng=rng),
                ),
                iterations=bootstrap_iterations,
            )
            rows.append(
                _metric_row(
                    descriptor=candidate_sample.descriptor,
                    metric_id=axis.metric_id,
                    metric_value=value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    category=axis.axis_id,
                    extra={
                        "cohort_axis_id": axis.axis_id,
                        "cohort_column": axis.column,
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
            ci_lower, ci_upper = bootstrap_ci(
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
    if len(ranks) < 3:
        raise ContractViolationError(
            f"ordinal axis {axis_id!r} requires at least three finite ranked groups from {rank_column!r}"
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
    return _spearman_correlation(gaps, distances), _kendall_tau(gaps, distances)


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
        ci_lower, ci_upper = bootstrap_ci(
            lambda groups=global_groups, rng=rng: _ordinal_statistics_from_groups(
                normalized,
                resample_groups(groups, rng=rng),
                ranks=axis.ranks,
            )[0],
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id=metric_ids["spearman"],
                metric_value=global_spearman,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    **axis_extra,
                    **({"display_name": spearman_display_name} if spearman_display_name else {}),
                },
            )
        )
        rows.append(
            _metric_row(
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
            balanced_spearman = _spearman_correlation(gaps, distances) if gaps.size else float("nan")
        ci_lower, ci_upper = bootstrap_ci(
            lambda groups=balanced_groups, rng=rng: _ordinal_statistics_from_groups(
                normalized,
                resample_groups(groups, rng=rng),
                ranks=axis.ranks,
            )[0],
            iterations=bootstrap_iterations,
        )
        rows.append(
            _metric_row(
                descriptor=candidate_sample.descriptor,
                metric_id=metric_ids["balanced_spearman"],
                metric_value=balanced_spearman,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                extra={
                    **axis_extra,
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
            ci_lower, ci_upper = bootstrap_ci(
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
                    descriptor=candidate_sample.descriptor,
                    metric_id=metric_id,
                    metric_value=within_mean,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    extra={
                        **axis_extra,
                        "ordinal_within_group_column": outer_column,
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
                    permutation_values.append(_spearman_correlation(gaps, distances))
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
            retention = _pearson_correlation(anchor_vector, context_vector)
            rows.append(
                _metric_row(
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
                        descriptor=candidate_sample.descriptor,
                        metric_id="reference_alignment_ethanol_background_relative",
                        metric_value=ethanol_alignment,
                    ),
                    _metric_row(
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


_PREASSAY_BUILDERS: dict[str, ScalarTableBuilder] = {
    "representation_health_summary": _representation_health_summary_table,
    "design_structure_summary": _design_structure_summary_table,
    "cohort_structure_summary": _cohort_structure_summary_table,
    "ordinal_axis_audit": _ordinal_axis_audit_table,
    "context_robustness_summary": _context_robustness_summary_table,
    "context_pair_summary": _context_pair_summary_table,
    "reference_alignment_summary": _reference_alignment_summary_table,
    "candidate_decision_frontier": _candidate_decision_frontier_table,
    "axis_centroid_distance": _axis_centroid_distance_table,
}

PREASSAY_BUILDER_KINDS = frozenset(_PREASSAY_BUILDERS)


def build_preassay_scalar_artifact(
    context: WorkspaceContext,
    *,
    scalar_id: str,
    builder_kind: str,
    params: dict[str, Any],
) -> BuiltScalarArtifact | None:
    builder = _PREASSAY_BUILDERS.get(builder_kind)
    if builder is None:
        return None
    table, inputs, stats = builder(context, params)
    artifact_dir = context.output_root / "scalars" / scalar_id
    write_table(table, artifact_dir / "table.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )
