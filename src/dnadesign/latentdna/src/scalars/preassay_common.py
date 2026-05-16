"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/scalars/preassay_common.py

Pre-assay scalar builders for representation triage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError
from ..presentation.labels import humanize_label
from ..workspaces.loader import WorkspaceContext
from .common import (
    ScalarInputRef,
    _candidate_descriptor_from_view,
    _load_view_scope_table,
    _optional_param,
    _require_param,
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


def _bootstrap_values(
    metric_fn: Callable[[], float],
    *,
    iterations: int,
) -> list[float]:
    values: list[float] = []
    for _ in range(iterations):
        value = float(metric_fn())
        if np.isfinite(value):
            values.append(value)
    return values


def _bootstrap_ci_from_values(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def _bootstrap_ci_with_values(
    metric_fn: Callable[[], float],
    *,
    iterations: int,
) -> tuple[float | None, float | None, list[float]]:
    values = _bootstrap_values(metric_fn, iterations=iterations)
    ci_lower, ci_upper = _bootstrap_ci_from_values(values)
    return ci_lower, ci_upper, values


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
