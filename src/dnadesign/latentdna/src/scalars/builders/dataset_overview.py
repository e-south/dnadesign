"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/scalars/builders/dataset_overview.py

Scalar builder for dataset cohort-overview count tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa

from ...contracts.errors import ContractViolationError
from ...io.parquet_io import write_table
from ...metadata.derivations import derive_metadata_value
from ...presentation.labels import humanize_label
from ...sources.resolver import inspect_source_schema, read_records_table, resolve_source
from ...views.row_contracts import derivation_dependency_columns
from ...workspaces.loader import WorkspaceContext
from ..common import BuiltScalarArtifact, ScalarInputRef, _optional_param


def _ordered_dataset_overview_categories(
    counts: Counter[str],
    category_order: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    ordered = list(category_order)
    known = {category for category, _ in ordered}
    next_order = max((order for _, order in ordered), default=0) + 1
    for offset, category in enumerate(sorted(set(counts) - known, key=str.casefold)):
        ordered.append((category, next_order + offset))
    return ordered


def _category_order(raw_order: object) -> list[tuple[str, int]]:
    if not isinstance(raw_order, list):
        return []
    ordered: list[tuple[str, int]] = []
    for index, item in enumerate(raw_order, start=1):
        if isinstance(item, dict):
            category = str(item.get("category") or item.get("value") or "").strip()
            order = int(item.get("order", index))
        else:
            category = str(item).strip()
            order = index
        if category:
            ordered.append((category, order))
    return ordered


def _dimension_specs(params: dict[str, Any]) -> list[dict[str, object]]:
    raw_dimensions = params.get("dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise ContractViolationError("dataset_overview requires configured dimensions")

    specs: list[dict[str, object]] = []
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            raise ContractViolationError("dataset_overview dimensions must be mapping objects")
        dimension = str(raw_dimension.get("dimension") or raw_dimension.get("id") or "").strip()
        column = str(raw_dimension.get("column") or "").strip()
        if not dimension or not column:
            raise ContractViolationError("dataset_overview dimensions must declare dimension and column")
        specs.append(
            {
                "dimension": dimension,
                "label": str(raw_dimension.get("label") or humanize_label(dimension)).strip(),
                "column": column,
                "category_order": _category_order(raw_dimension.get("category_order", [])),
                "include_unlisted_categories": bool(raw_dimension.get("include_unlisted_categories", True)),
                "category_labels": {
                    str(key): str(value) for key, value in dict(raw_dimension.get("category_labels", {}) or {}).items()
                },
            }
        )
    return specs


def _dataset_category_label(spec: dict[str, object], category: str) -> str:
    labels = spec.get("category_labels", {})
    if isinstance(labels, dict) and category in labels:
        return str(labels[category])
    return humanize_label(category)


def _metadata_value(context: WorkspaceContext, row: dict[str, object], column: str) -> object:
    if column in row:
        return row[column]
    derivation = (context.config.metadata.derivations or {}).get(column)
    if derivation is None:
        raise ContractViolationError(
            f"dataset_overview dimension column {column!r} is missing and has no metadata derivation"
        )
    if derivation.kind == "lookup":
        raise ContractViolationError(
            f"dataset_overview dimension column {column!r} uses a lookup derivation; "
            "materialize a row table or use a source-native column instead"
        )
    return derive_metadata_value(row, derivation)


def build_dataset_overview_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, Any],
) -> BuiltScalarArtifact:
    """Build a shared-denominator cohort overview table from workspace sources."""

    source_ids = [str(value) for value in _optional_param(params, "source_ids", default=list(context.config.sources))]
    if not source_ids:
        raise ContractViolationError("dataset_overview requires at least one source")

    dimension_specs = _dimension_specs(params)
    dimension_columns = [str(spec["column"]) for spec in dimension_specs]
    output_rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    canonical_subject_keys: set[object] | None = None
    denominator: int | None = None
    canonical_counts: dict[str, Counter[str]] = {}

    for source_id in source_ids:
        source = context.require_source(source_id)
        resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
        available_columns = set(inspect_source_schema(resolved)["columns"])
        population_key = source.subject_key or source.record_key
        dependency_columns, missing_dependency_columns = derivation_dependency_columns(
            context,
            columns=dimension_columns,
            available_columns=available_columns,
        )
        if missing_dependency_columns:
            raise ContractViolationError(
                f"dataset_overview metadata derivation inputs are missing from source "
                f"{source_id!r}: {missing_dependency_columns}"
            )
        required_columns = list(
            dict.fromkeys(
                [
                    source.record_key,
                    population_key,
                    *dimension_columns,
                    *dependency_columns,
                ]
            )
        )
        required_columns = [column for column in required_columns if column in available_columns]
        table = read_records_table(resolved, columns=required_columns)
        inputs.append(ScalarInputRef(kind="source", artifact_id=source_id, path=resolved.records_path))
        row_dicts = table.to_pylist()
        source_subject_keys = {row[population_key] for row in row_dicts}
        if denominator is None:
            denominator = len(row_dicts)
            canonical_subject_keys = source_subject_keys
        else:
            if len(row_dicts) != denominator:
                raise ContractViolationError(
                    f"dataset_overview requires one shared denominator; {source_id!r} has {len(row_dicts)} rows "
                    f"but the canonical source has {denominator}"
                )
            if source_subject_keys != canonical_subject_keys:
                raise ContractViolationError(
                    f"dataset_overview requires aligned promoter populations; {source_id!r} does not match "
                    "the canonical source row set"
                )
        source_counts = {
            str(spec["dimension"]): Counter(
                str(_metadata_value(context, row, str(spec["column"]))) for row in row_dicts
            )
            for spec in dimension_specs
        }
        if not canonical_counts:
            canonical_counts = source_counts
            continue
        for dimension, counts in source_counts.items():
            if counts != canonical_counts[dimension]:
                raise ContractViolationError(
                    f"dataset_overview requires matching cohort partitions across sources; {source_id!r} "
                    f"does not match the canonical counts for {dimension!r}"
                )

    assert denominator is not None
    for spec in dimension_specs:
        dimension = str(spec["dimension"])
        counts = canonical_counts[dimension]
        category_order = spec["category_order"]
        assert isinstance(category_order, list)
        include_unlisted_categories = bool(spec.get("include_unlisted_categories", True))
        if include_unlisted_categories:
            ordered_categories = _ordered_dataset_overview_categories(counts, category_order)
        else:
            if not category_order:
                raise ContractViolationError(
                    f"dataset_overview dimension {dimension!r} disables unlisted categories "
                    "but declares no category_order"
                )
            ordered_categories = list(category_order)
        dimension_total = sum(int(counts.get(category, 0)) for category, _ in ordered_categories)
        if include_unlisted_categories and dimension_total != denominator:
            raise ContractViolationError(
                f"dataset_overview dimension {dimension!r} sums to {dimension_total}, expected {denominator}"
            )
        if not include_unlisted_categories and dimension_total > denominator:
            raise ContractViolationError(
                f"dataset_overview dimension {dimension!r} sums to {dimension_total}, expected at most {denominator}"
            )
        for category, order in ordered_categories:
            count = int(counts.get(category, 0))
            fraction = count / float(denominator)
            output_rows.append(
                {
                    "dimension": dimension,
                    "dimension_label": str(spec["label"]),
                    "category": category,
                    "category_label": _dataset_category_label(spec, category),
                    "count": count,
                    "denominator": denominator,
                    "fraction": fraction,
                    "percent": fraction * 100.0,
                    "order": order,
                }
            )
    table = pa.Table.from_pylist(output_rows)
    write_table(table, artifact_dir / "table.parquet")
    stats = {"rows": table.num_rows, "sources": len(source_ids), "denominator": denominator}
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )
