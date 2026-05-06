"""
Semantic scalar-table builders for artifact-first latentdna workflows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..alignments.aggregators import aggregate_rows
from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..geometry.cohorts import balanced_group_indices
from ..geometry.preprocessing import try_l2_normalize_vector
from ..io.json_io import read_json, write_json
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table, write_table
from ..labels import humanize_label
from ..metadata.derivations import derive_metadata_value
from ..metrics.definitions import resolve_metric_definition, validate_metric_registry
from ..sources.resolver import inspect_source_schema, read_records_table, resolve_source
from ..views.row_contracts import derivation_dependency_columns
from ..views.scopes import resolve_view_scope
from ..workspaces.loader import WorkspaceContext
from .classification_metrics import binary_metrics, dual_joint_margin
from .common import (
    BuiltScalarArtifact,
    ScalarInputRef,
    _effective_rank,
    _load_view_scope_table,
    _normalized_geometry_rows,
    _optional_param,
    _reducer_summary_path,
    _require_param,
    _spearman_correlation,
)
from .preassay import build_preassay_scalar_artifact


def _view_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path]:
    matrix_path = context.output_root / "views" / view_id / "matrix.npy"
    rows_path = context.output_root / "views" / view_id / "rows.parquet"
    if not matrix_path.is_file() or not rows_path.is_file():
        raise MissingArtifactError(f"view artifact is missing for scalar.build: {view_id}")
    return matrix_path, rows_path


def _alignment_paths(context: WorkspaceContext, alignment_id: str) -> tuple[Path, Path, Path]:
    artifact_dir = context.output_root / "alignments" / alignment_id
    manifest_path = artifact_dir / "manifest.json"
    rows_path = artifact_dir / "rows.parquet"
    mapping_path = artifact_dir / "mapping.parquet"
    for required in (manifest_path, rows_path, mapping_path):
        if not required.is_file():
            raise MissingArtifactError(f"alignment artifact is missing for scalar.build: {required}")
    return manifest_path, rows_path, mapping_path


def _scalar_table_path(context: WorkspaceContext, scalar_id: str) -> Path:
    path = context.output_root / "scalars" / scalar_id / "table.parquet"
    if not path.is_file():
        raise MissingArtifactError(f"scalar artifact is missing for scalar.build: {scalar_id}")
    return path


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


def _neighbor_paths(context: WorkspaceContext, neighbor_id: str) -> tuple[Path, Path, Path]:
    artifact_dir = context.output_root / "neighbors" / neighbor_id
    rows_path = artifact_dir / "rows.parquet"
    indices_path = artifact_dir / "indices.npy"
    manifest_path = artifact_dir / "manifest.json"
    for required in (rows_path, indices_path, manifest_path):
        if not required.is_file():
            raise MissingArtifactError(f"neighbor artifact is missing for scalar.build: {required}")
    return rows_path, indices_path, manifest_path


def _agreement_summary_path(context: WorkspaceContext, agreement_id: str) -> Path:
    path = context.output_root / "agreements" / agreement_id / "summary.json"
    if not path.is_file():
        raise MissingArtifactError(f"agreement artifact is missing for scalar.build: {agreement_id}")
    return path


def _select_indices(rows: list[dict[str, Any]], where: dict[str, Any]) -> list[int]:
    column = where.get("column")
    if not isinstance(column, str):
        raise ContractViolationError("landmark where clause requires a 'column' field")
    if "equals" in where:
        target = where["equals"]
        return [index for index, row in enumerate(rows) if row.get(column) == target]
    if "in" in where:
        targets = set(where["in"])
        return [index for index, row in enumerate(rows) if row.get(column) in targets]
    raise ContractViolationError("landmark where clause requires either 'equals' or 'in'")


def _matching_source_keys(source_rows: list[dict[str, Any]], *, key_column: str, where: dict[str, Any]) -> set[object]:
    matched_indices = _select_indices(source_rows, where)
    return {source_rows[index][key_column] for index in matched_indices}


def _source_column_value_map(
    context: WorkspaceContext,
    *,
    source_id: str,
    column: str,
) -> tuple[dict[object, object], object]:
    source = context.require_source(source_id)
    resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
    source_rows = read_records_table(
        resolved,
        columns=[source.record_key, column],
    ).to_pylist()
    return (
        {row[source.record_key]: row.get(column) for row in source_rows},
        source,
    )


def _project_source_column_to_view_rows(
    context: WorkspaceContext,
    *,
    view_id: str,
    rows_table: pa.Table,
    source_id: str,
    column: str,
) -> pa.Array:
    values_by_key, source = _source_column_value_map(
        context,
        source_id=source_id,
        column=column,
    )
    view = context.require_source_view(view_id)
    row_key_column = source.record_key if source.record_key in rows_table.column_names else None
    if row_key_column is None and view.source == source_id and source.subject_key in rows_table.column_names:
        row_key_column = source.subject_key
    if row_key_column is None:
        raise ContractViolationError(
            f"view {view_id!r} does not expose a key column to project source column {column!r}"
        )
    return pa.array([values_by_key.get(key) for key in rows_table[row_key_column].to_pylist()])


def _project_source_column_via_alignment(
    context: WorkspaceContext,
    *,
    alignment_id: str,
    view_id: str,
    rows_table: pa.Table,
    source_id: str,
    column: str,
) -> pa.Array:
    alignment = context.require_alignment(alignment_id)
    left_matches = _alignment_input_uses_source(context, alignment.left, source_id)
    right_matches = _alignment_input_uses_source(context, alignment.right, source_id)
    if left_matches == right_matches:
        raise ContractViolationError(
            f"alignment {alignment_id!r} cannot project source {source_id!r} onto view {view_id!r}"
        )

    matched_side = "left" if left_matches else "right"
    target_side = "left" if alignment.left == view_id else "right"
    matched_ref = alignment.left if matched_side == "left" else alignment.right
    matched_rows_path = context.output_root / "views" / matched_ref / "rows.parquet"
    if not matched_rows_path.is_file():
        raise MissingArtifactError(f"view artifact is missing for scalar.build: {matched_ref}")

    matched_rows = read_table(matched_rows_path)
    values_by_key, source = _source_column_value_map(
        context,
        source_id=source_id,
        column=column,
    )
    row_key_column = source.record_key if source.record_key in matched_rows.column_names else None
    if row_key_column is None and source.subject_key in matched_rows.column_names:
        row_key_column = source.subject_key
    if row_key_column is None:
        raise ContractViolationError(
            f"view {matched_ref!r} does not expose a key column to project source column {column!r}"
        )

    matched_values = [values_by_key.get(key) for key in matched_rows[row_key_column].to_pylist()]
    _, _, mapping_path = _alignment_paths(context, alignment_id)
    mapping_rows = read_table(mapping_path).to_pylist()
    matched_column = f"{matched_side}_indices"
    target_column = f"{target_side}_indices"
    projected_values: list[object | None] = [None] * int(rows_table.num_rows)
    for row in mapping_rows:
        values = {
            matched_values[int(index)]
            for index in row.get(matched_column, [])
            if matched_values[int(index)] is not None
        }
        if len(values) > 1:
            raise ContractViolationError(
                f"alignment {alignment_id!r} projects conflicting {column!r} values onto {view_id!r}"
            )
        value = next(iter(values), None)
        for index in row.get(target_column, []):
            target_index = int(index)
            existing = projected_values[target_index]
            if existing is not None and value is not None and existing != value:
                raise ContractViolationError(
                    f"alignment {alignment_id!r} projects conflicting {column!r} values onto one target row"
                )
            if value is not None:
                projected_values[target_index] = value
    return pa.array(projected_values)


def _alignment_input_uses_source(context: WorkspaceContext, ref_id: str, source_id: str) -> bool:
    if ref_id == source_id:
        return True
    view = context.config.views.get(ref_id)
    return bool(view is not None and getattr(view, "source", None) == source_id)


def _alignment_projected_indices(
    context: WorkspaceContext,
    *,
    alignment_id: str,
    view_id: str,
    landmark_id: str,
) -> list[int]:
    alignment = context.require_alignment(alignment_id)
    if view_id not in {alignment.left, alignment.right}:
        raise ContractViolationError(f"alignment {alignment_id} does not include view {view_id!r}")

    landmark = context.require_landmark(landmark_id)
    left_matches = _alignment_input_uses_source(context, alignment.left, landmark.source)
    right_matches = _alignment_input_uses_source(context, alignment.right, landmark.source)
    if left_matches == right_matches:
        raise ContractViolationError(
            f"alignment {alignment_id} cannot resolve landmark source {landmark.source!r} for view {view_id!r}"
        )

    matched_side = "left" if left_matches else "right"
    target_side = "left" if alignment.left == view_id else "right"
    matched_ref = alignment.left if matched_side == "left" else alignment.right

    matched_rows_path = context.output_root / "views" / matched_ref / "rows.parquet"
    selector_column = str(landmark.where["column"])
    source = context.require_source(landmark.source)
    resolved = resolve_source(landmark.source, source, workspace_dir=context.workspace_dir)
    if matched_rows_path.is_file():
        matched_rows = read_table(matched_rows_path)
        if selector_column in matched_rows.column_names:
            matched_indices = _select_indices(matched_rows.to_pylist(), landmark.where)
        else:
            source_rows = read_records_table(
                resolved,
                columns=[source.record_key, selector_column],
            ).to_pylist()
            matched_keys = _matching_source_keys(
                source_rows,
                key_column=source.record_key,
                where=landmark.where,
            )
            row_key_column = source.record_key if source.record_key in matched_rows.column_names else None
            if row_key_column is None and source.subject_key in matched_rows.column_names:
                row_key_column = source.subject_key
            if row_key_column is None:
                raise ContractViolationError(
                    f"view {matched_ref!r} does not expose a key column to project landmark {landmark_id!r} membership"
                )
            matched_indices = [
                index for index, key in enumerate(matched_rows[row_key_column].to_pylist()) if key in matched_keys
            ]
    else:
        matched_rows = read_records_table(resolved, columns=[selector_column])
        matched_indices = _select_indices(matched_rows.to_pylist(), landmark.where)
    if not matched_indices:
        raise ContractViolationError(f"landmark {landmark_id} matched no rows in projected alignment scope")

    _, _, mapping_path = _alignment_paths(context, alignment_id)
    mapping_rows = read_table(mapping_path).to_pylist()
    matched_index_set = set(matched_indices)
    matched_column = f"{matched_side}_indices"
    target_column = f"{target_side}_indices"
    projected_indices: set[int] = set()
    for row in mapping_rows:
        source_indices = {int(index) for index in row.get(matched_column, [])}
        if source_indices.intersection(matched_index_set):
            projected_indices.update(int(index) for index in row.get(target_column, []))
    if not projected_indices:
        raise ContractViolationError(
            f"landmark {landmark_id} matched no aligned rows in {alignment_id!r} for view {view_id!r}"
        )
    return sorted(projected_indices)


def _same_source_landmark_indices(
    context: WorkspaceContext,
    *,
    view_id: str,
    rows_table: pa.Table,
    landmark_id: str,
) -> list[int]:
    landmark = context.require_landmark(landmark_id)
    selector_column = str(landmark.where["column"])
    if selector_column in rows_table.column_names:
        return _select_indices(rows_table.to_pylist(), landmark.where)

    view = context.require_source_view(view_id)
    source = context.require_source(landmark.source)
    resolved = resolve_source(landmark.source, source, workspace_dir=context.workspace_dir)
    source_table = read_records_table(
        resolved,
        columns=[source.record_key, selector_column],
    )
    matched_keys = _matching_source_keys(
        source_table.to_pylist(),
        key_column=source.record_key,
        where=landmark.where,
    )
    if not matched_keys:
        return []

    row_key_column = source.record_key if source.record_key in rows_table.column_names else None
    if row_key_column is None and view.source == landmark.source and source.subject_key in rows_table.column_names:
        row_key_column = source.subject_key
    if row_key_column is None:
        raise ContractViolationError(
            f"view {view_id!r} does not expose a key column to project landmark {landmark_id!r} membership"
        )
    row_keys = rows_table[row_key_column].to_pylist()
    return [index for index, key in enumerate(row_keys) if key in matched_keys]


def _replace_or_append_column(table: pa.Table, name: str, values: np.ndarray | list[float]) -> pa.Table:
    array = pa.array(values)
    if name in table.column_names:
        index = table.column_names.index(name)
        return table.set_column(index, name, array)
    return table.append_column(name, array)


def _similarity_margin_table(
    context: WorkspaceContext,
    *,
    view_id: str,
    margin_pairs: list[dict[str, Any]],
    alignment_id: str | None,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    matrix_path, rows_path = _view_paths(context, view_id)
    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    rows_table = read_table(rows_path)
    normalized_rows = _normalized_geometry_rows(matrix)
    inputs = [
        ScalarInputRef(kind="view_matrix", artifact_id=view_id, path=matrix_path),
        ScalarInputRef(kind="view_rows", artifact_id=view_id, path=rows_path),
    ]
    if alignment_id is not None:
        _, _, mapping_path = _alignment_paths(context, alignment_id)
        inputs.append(ScalarInputRef(kind="alignment_set", artifact_id=alignment_id, path=mapping_path))
    seen_landmark_sources: set[str] = set()
    table = rows_table
    margin_stats: dict[str, object] = {"view_id": view_id, "margin_count": len(margin_pairs)}
    selector_landmarks: dict[str, str] = {}
    for pair in margin_pairs:
        target_landmark = str(_require_param(pair, "target_landmark"))
        control_landmark = str(_require_param(pair, "control_landmark"))
        output_column = str(_require_param(pair, "output_column"))
        for landmark_id in (target_landmark, control_landmark):
            selector_column = str(context.require_landmark(landmark_id).where["column"])
            selected_landmark = selector_landmarks.get(selector_column)
            if (
                selected_landmark is not None
                and context.require_landmark(selected_landmark).source != context.require_landmark(landmark_id).source
            ):
                raise ContractViolationError(
                    f"similarity_margin cannot populate selector column {selector_column!r} from multiple sources"
                )
            selector_landmarks.setdefault(selector_column, landmark_id)
        target_indices = (
            _same_source_landmark_indices(
                context,
                view_id=view_id,
                rows_table=rows_table,
                landmark_id=target_landmark,
            )
            if context.require_landmark(target_landmark).source == context.require_source_view(view_id).source
            else _alignment_projected_indices(
                context,
                alignment_id=str(alignment_id),
                view_id=view_id,
                landmark_id=target_landmark,
            )
            if alignment_id is not None
            else []
        )
        control_indices = (
            _same_source_landmark_indices(
                context,
                view_id=view_id,
                rows_table=rows_table,
                landmark_id=control_landmark,
            )
            if context.require_landmark(control_landmark).source == context.require_source_view(view_id).source
            else _alignment_projected_indices(
                context,
                alignment_id=str(alignment_id),
                view_id=view_id,
                landmark_id=control_landmark,
            )
            if alignment_id is not None
            else []
        )
        if not target_indices:
            raise ContractViolationError(f"landmark {target_landmark} matched no rows for similarity margin")
        if not control_indices:
            raise ContractViolationError(f"landmark {control_landmark} matched no rows for similarity margin")
        target_reference = try_l2_normalize_vector(
            np.asarray(normalized_rows[target_indices].mean(axis=0), dtype=np.float32)
        )
        control_reference = try_l2_normalize_vector(
            np.asarray(normalized_rows[control_indices].mean(axis=0), dtype=np.float32)
        )
        if target_reference is None or control_reference is None:
            margin = np.full(len(rows_table), np.nan, dtype=np.float32)
            margin_stats[f"{output_column}_degenerate_reference"] = True
        else:
            target_similarity = np.asarray(normalized_rows @ target_reference, dtype=np.float32)
            control_similarity = np.asarray(normalized_rows @ control_reference, dtype=np.float32)
            margin = np.asarray(target_similarity - control_similarity, dtype=np.float32)
        table = _replace_or_append_column(table, output_column, margin)
        margin_stats[f"{output_column}_target_members"] = len(target_indices)
        margin_stats[f"{output_column}_control_members"] = len(control_indices)
        for landmark_id in (target_landmark, control_landmark):
            source_id = context.require_landmark(landmark_id).source
            if source_id in seen_landmark_sources:
                continue
            seen_landmark_sources.add(source_id)
            source = context.require_source(source_id)
            resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
            if resolved.records_path is not None:
                inputs.append(ScalarInputRef(kind="landmark_source", artifact_id=source_id, path=resolved.records_path))
    for selector_column, landmark_id in selector_landmarks.items():
        if selector_column in table.column_names:
            continue
        landmark = context.require_landmark(landmark_id)
        if landmark.source == context.require_source_view(view_id).source:
            projected = _project_source_column_to_view_rows(
                context,
                view_id=view_id,
                rows_table=rows_table,
                source_id=landmark.source,
                column=selector_column,
            )
        else:
            if alignment_id is None:
                raise ContractViolationError(
                    f"similarity_margin cannot project selector column {selector_column!r} without alignment_id"
                )
            projected = _project_source_column_via_alignment(
                context,
                alignment_id=str(alignment_id),
                view_id=view_id,
                rows_table=rows_table,
                source_id=landmark.source,
                column=selector_column,
            )
        table = _replace_or_append_column(table, selector_column, projected)
    if {
        "wildtype_margin_ethanol_vs_control",
        "wildtype_margin_cipro_vs_control",
    }.issubset(table.column_names):
        wildtype_dual_margin = dual_joint_margin(
            np.asarray(table["wildtype_margin_ethanol_vs_control"].to_pylist(), dtype=np.float32),
            np.asarray(table["wildtype_margin_cipro_vs_control"].to_pylist(), dtype=np.float32),
        )
        table = _replace_or_append_column(table, "dual_joint_margin", wildtype_dual_margin)
    if {
        "synthetic_margin_ethanol_vs_background",
        "synthetic_margin_cipro_vs_background",
    }.issubset(table.column_names):
        synthetic_dual_margin = dual_joint_margin(
            np.asarray(table["synthetic_margin_ethanol_vs_background"].to_pylist(), dtype=np.float32),
            np.asarray(table["synthetic_margin_cipro_vs_background"].to_pylist(), dtype=np.float32),
        )
        table = _replace_or_append_column(table, "synthetic_dual_joint_margin", synthetic_dual_margin)
    return table, inputs, margin_stats


def _cohort_similarity_margin_table(
    context: WorkspaceContext,
    *,
    view_id: str,
    sample_id: str | None = None,
    cohort_column: str,
    margin_pairs: list[dict[str, Any]],
    leave_one_out: bool = False,
    balance_group_column: str | None = None,
    balance_columns: list[str] | None = None,
    balance_reference_only: bool = False,
    required_group_values: set[str] | None = None,
    exclude_group_values: set[str] | None = None,
    seed: int | None = None,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    if sample_id is None:
        matrix_path, rows_path = _view_paths(context, view_id)
        matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
        rows_table = read_table(rows_path)
        inputs = [
            ScalarInputRef(kind="view_matrix", artifact_id=view_id, path=matrix_path),
            ScalarInputRef(kind="view_rows", artifact_id=view_id, path=rows_path),
        ]
    else:
        matrix, rows, inputs = _load_view_scope_table(context, view_id=view_id, sample_id=sample_id)
        rows_table = pa.Table.from_pylist(rows)
    rows = rows_table.to_pylist()
    normalized_rows = _normalized_geometry_rows(matrix)
    table = rows_table
    stats: dict[str, object] = {"view_id": view_id, "margin_count": len(margin_pairs)}
    if sample_id is not None:
        stats["sample_id"] = sample_id
    balanced_groups: dict[str, list[int]] = {}
    if balance_group_column is not None:
        resolved_balance_columns = list(balance_columns or [])
        if not resolved_balance_columns:
            raise ContractViolationError("cohort_similarity_margin balance_group_column requires balance_columns")
        missing_columns = [
            column
            for column in [balance_group_column, *resolved_balance_columns]
            if column not in rows_table.column_names
        ]
        if missing_columns:
            raise ContractViolationError(
                f"balanced cohort_similarity_margin columns are missing from {view_id!r}: {missing_columns}"
            )
        rng = np.random.default_rng(seed if seed is not None else context.config.defaults.random_seed)
        balanced_groups = balanced_group_indices(
            rows,
            group_column=balance_group_column,
            balance_columns=resolved_balance_columns,
            required_group_values=required_group_values,
            exclude_group_values=exclude_group_values,
            rng=rng,
        )
        selected_indices = sorted({index for indices in balanced_groups.values() for index in indices})
        if not selected_indices:
            raise ContractViolationError(
                f"cohort_similarity_margin balancing matched no rows for {view_id!r} on {balance_group_column!r}"
            )
        if not balance_reference_only:
            index_array = np.asarray(selected_indices, dtype=np.int64)
            rows = [rows[index] for index in selected_indices]
            normalized_rows = normalized_rows[index_array]
            table = rows_table.take(pa.array(selected_indices, type=pa.int64()))
        stats["balanced_group_column"] = balance_group_column
        stats["balanced_columns"] = resolved_balance_columns
        stats["balanced_row_count"] = len(selected_indices)
        stats["balanced_group_sizes"] = {group: len(indices) for group, indices in sorted(balanced_groups.items())}
        stats["balanced_reference_only"] = bool(balance_reference_only)
        if required_group_values is not None:
            stats["balanced_required_group_values"] = sorted(required_group_values)
        if exclude_group_values is not None:
            stats["balanced_exclude_group_values"] = sorted(exclude_group_values)
    for pair in margin_pairs:
        pair_cohort_column = str(pair.get("cohort_column") or cohort_column)
        if pair_cohort_column not in rows_table.column_names:
            raise ContractViolationError(f"cohort column {pair_cohort_column!r} is missing from {view_id!r}")
        target_values = {str(value) for value in _require_param(pair, "target_values")}
        control_values = {str(value) for value in _require_param(pair, "control_values")}
        output_column = str(_require_param(pair, "output_column"))
        target_indices = [index for index, row in enumerate(rows) if str(row.get(pair_cohort_column)) in target_values]
        control_indices = [
            index for index, row in enumerate(rows) if str(row.get(pair_cohort_column)) in control_values
        ]
        if not target_indices:
            raise ContractViolationError(f"cohort target matched no rows for {output_column!r}")
        if not control_indices:
            raise ContractViolationError(f"cohort control matched no rows for {output_column!r}")
        target_reference_indices = target_indices
        control_reference_indices = control_indices
        if balance_reference_only and pair_cohort_column == balance_group_column:
            target_reference_indices = sorted(
                {index for value in target_values for index in balanced_groups.get(str(value), [])}
            )
            control_reference_indices = sorted(
                {index for value in control_values for index in balanced_groups.get(str(value), [])}
            )
            if not target_reference_indices:
                raise ContractViolationError(f"balanced cohort target matched no reference rows for {output_column!r}")
            if not control_reference_indices:
                raise ContractViolationError(f"balanced cohort control matched no reference rows for {output_column!r}")
        # Synthetic cohort centroids are EDA-only companions, not benchmark
        # features. Build their directions from already normalized row vectors so
        # cohorts near the global standardized mean do not collapse to an invalid
        # zero-norm cosine reference.
        target_sum = np.asarray(normalized_rows[target_reference_indices].sum(axis=0), dtype=np.float32)
        control_sum = np.asarray(normalized_rows[control_reference_indices].sum(axis=0), dtype=np.float32)
        target_reference = try_l2_normalize_vector(
            np.asarray(target_sum / len(target_reference_indices), dtype=np.float32)
        )
        control_reference = try_l2_normalize_vector(
            np.asarray(control_sum / len(control_reference_indices), dtype=np.float32)
        )
        if target_reference is None or control_reference is None:
            margin = np.full(len(rows), np.nan, dtype=np.float32)
            stats[f"{output_column}_degenerate_reference"] = True
            table = _replace_or_append_column(table, output_column, margin)
            stats[f"{output_column}_target_members"] = len(target_indices)
            stats[f"{output_column}_control_members"] = len(control_indices)
            if balance_reference_only and pair_cohort_column == balance_group_column:
                stats[f"{output_column}_target_reference_members"] = len(target_reference_indices)
                stats[f"{output_column}_control_reference_members"] = len(control_reference_indices)
            stats[f"{output_column}_leave_one_out"] = bool(leave_one_out)
            continue
        target_similarity = np.asarray(normalized_rows @ target_reference, dtype=np.float32)
        control_similarity = np.asarray(normalized_rows @ control_reference, dtype=np.float32)
        if leave_one_out:
            if len(target_reference_indices) <= 1 or len(control_reference_indices) <= 1:
                raise ContractViolationError(
                    f"leave_one_out cohort margins require at least two rows in each cohort for {output_column!r}"
                )
            target_reference_set = set(target_reference_indices)
            control_reference_set = set(control_reference_indices)
            for index in target_reference_indices:
                adjusted = try_l2_normalize_vector(
                    np.asarray(
                        (target_sum - normalized_rows[index]) / (len(target_reference_indices) - 1),
                        dtype=np.float32,
                    )
                )
                target_similarity[index] = float(normalized_rows[index] @ adjusted) if adjusted is not None else np.nan
            for index in control_reference_indices:
                adjusted = try_l2_normalize_vector(
                    np.asarray(
                        (control_sum - normalized_rows[index]) / (len(control_reference_indices) - 1),
                        dtype=np.float32,
                    )
                )
                control_similarity[index] = float(normalized_rows[index] @ adjusted) if adjusted is not None else np.nan
            if balance_reference_only and pair_cohort_column == balance_group_column:
                for index in target_indices:
                    if index in target_reference_set:
                        continue
                    target_similarity[index] = float(normalized_rows[index] @ target_reference)
                for index in control_indices:
                    if index in control_reference_set:
                        continue
                    control_similarity[index] = float(normalized_rows[index] @ control_reference)
        margin = np.asarray(target_similarity - control_similarity, dtype=np.float32)
        table = _replace_or_append_column(table, output_column, margin)
        stats[f"{output_column}_target_members"] = len(target_indices)
        stats[f"{output_column}_control_members"] = len(control_indices)
        if balance_reference_only and pair_cohort_column == balance_group_column:
            stats[f"{output_column}_target_reference_members"] = len(target_reference_indices)
            stats[f"{output_column}_control_reference_members"] = len(control_reference_indices)
        stats[f"{output_column}_leave_one_out"] = bool(leave_one_out)
        stats[f"{output_column}_cohort_column"] = pair_cohort_column
    if {
        "synthetic_margin_ethanol_vs_background",
        "synthetic_margin_cipro_vs_background",
    }.issubset(table.column_names):
        synthetic_best_stress_margin = np.maximum(
            np.asarray(table["synthetic_margin_ethanol_vs_background"].to_pylist(), dtype=np.float32),
            np.asarray(table["synthetic_margin_cipro_vs_background"].to_pylist(), dtype=np.float32),
        )
        table = _replace_or_append_column(table, "synthetic_best_stress_margin", synthetic_best_stress_margin)
    return table, inputs, stats


def _alignment_projection(context: WorkspaceContext, alignment_id: str) -> tuple[pa.Table, list[str], list[str]]:
    manifest_path, rows_path, _ = _alignment_paths(context, alignment_id)
    manifest = context.read_manifest(manifest_path)
    left_key_columns = [str(value) for value in manifest["params"]["key_columns"]]
    right_key_columns = [str(value) for value in manifest["params"].get("right_key_columns", left_key_columns)]
    return read_table(rows_path), left_key_columns, right_key_columns


def _sampled_alignment_scope_matrix(
    context: WorkspaceContext,
    *,
    alignment_id: str,
    view_id: str,
    alignment_indices: np.ndarray,
) -> np.ndarray:
    alignment = context.require_alignment(alignment_id)
    manifest_path, _, mapping_path = _alignment_paths(context, alignment_id)
    manifest = context.read_manifest(manifest_path)
    if view_id == alignment.left:
        index_field = "left_indices"
        mode = str(manifest["params"].get("left_aggregation", "error"))
    elif view_id == alignment.right:
        index_field = "right_indices"
        mode = str(manifest["params"].get("right_aggregation", "error"))
    else:
        raise ContractViolationError(f"alignment {alignment_id} does not include view {view_id!r}")

    matrix_path, _ = _view_paths(context, view_id)
    source_matrix = np.load(matrix_path, mmap_mode="r")
    mapping_rows = read_table(mapping_path).take(pa.array(alignment_indices, type=pa.int64())).to_pylist()
    aligned_matrix = np.vstack(
        [aggregate_rows(source_matrix, [int(index) for index in row[index_field]], mode=mode) for row in mapping_rows]
    ).astype(np.float32, copy=False)
    return np.ascontiguousarray(aligned_matrix)


def _project_table_to_alignment(
    *,
    alignment_rows: pa.Table,
    candidate_table: pa.Table,
    left_key_columns: list[str],
    right_key_columns: list[str],
    candidate_label: str,
) -> pa.Table:
    if set(left_key_columns).issubset(set(candidate_table.column_names)):
        candidate_key_columns = left_key_columns
    elif set(right_key_columns).issubset(set(candidate_table.column_names)):
        candidate_key_columns = right_key_columns
    else:
        raise ContractViolationError(
            f"{candidate_label} shares neither alignment keys {left_key_columns} nor {right_key_columns}"
        )

    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    for row in candidate_table.to_pylist():
        key = tuple(row[column] for column in candidate_key_columns)
        if key in grouped:
            raise ContractViolationError(f"{candidate_label} is non-unique on alignment projection keys")
        grouped[key] = row

    projected_rows: list[dict[str, object]] = []
    missing: list[tuple[object, ...]] = []
    for row in alignment_rows.select(left_key_columns).to_pylist():
        key = tuple(row[column] for column in left_key_columns)
        candidate = grouped.get(key)
        if candidate is None:
            missing.append(key)
            continue
        projected_rows.append(candidate)
    if missing:
        raise ContractViolationError(f"{candidate_label} is missing aligned keys: {missing[:5]}")
    return pa.Table.from_pylist(projected_rows)


def _row_metadata_from_alignment(
    context: WorkspaceContext,
    *,
    alignment_id: str,
    metadata_view_id: str,
) -> tuple[pa.Table, list[ScalarInputRef]]:
    alignment_rows, left_key_columns, right_key_columns = _alignment_projection(context, alignment_id)
    _, rows_path = _view_paths(context, metadata_view_id)
    view_rows = read_table(rows_path)
    projected = _project_table_to_alignment(
        alignment_rows=alignment_rows,
        candidate_table=view_rows,
        left_key_columns=left_key_columns,
        right_key_columns=right_key_columns,
        candidate_label=f"alignment metadata source {metadata_view_id}",
    )
    return projected, [ScalarInputRef(kind="view_rows", artifact_id=metadata_view_id, path=rows_path)]


def _allocate_stratified_counts(groups: dict[str, list[int]], total: int) -> dict[str, int]:
    total_rows = sum(len(indices) for indices in groups.values())
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for key, indices in groups.items():
        raw = (len(indices) / total_rows) * total
        count = min(len(indices), int(raw))
        quotas[key] = count
        assigned += count
        remainders.append((raw - count, key))
    for _, key in sorted(remainders, reverse=True):
        if assigned >= total:
            break
        if quotas[key] >= len(groups[key]):
            continue
        quotas[key] += 1
        assigned += 1
    return quotas


def _sample_alignment_indices(
    rows_table: pa.Table,
    *,
    sample_size: int,
    group_column: str | None,
    seed: int,
) -> list[int]:
    row_count = int(rows_table.num_rows)
    if sample_size >= row_count:
        return list(range(row_count))
    rng = np.random.default_rng(seed)
    if group_column is None or group_column not in rows_table.column_names:
        return sorted(rng.choice(row_count, size=sample_size, replace=False).tolist())
    groups: dict[str, list[int]] = defaultdict(list)
    values = rows_table[group_column].combine_chunks().to_pylist()
    for index, value in enumerate(values):
        groups[str(value)].append(index)
    quotas = _allocate_stratified_counts(groups, sample_size)
    selected: list[int] = []
    for key in sorted(groups):
        candidates = np.asarray(groups[key], dtype=np.int64)
        order = rng.permutation(len(candidates))
        selected.extend(sorted(candidates[order][: quotas[key]].tolist()))
    return sorted(selected)


def _cosine_distance_correlation(left_matrix: np.ndarray, right_matrix: np.ndarray) -> float:
    left = _normalized_geometry_rows(left_matrix)
    right = _normalized_geometry_rows(right_matrix)
    left_distance = 1.0 - np.asarray(left @ left.T, dtype=np.float32)
    right_distance = 1.0 - np.asarray(right @ right.T, dtype=np.float32)
    upper = np.triu_indices(left_distance.shape[0], k=1)
    left_values = np.asarray(left_distance[upper], dtype=np.float64)
    right_values = np.asarray(right_distance[upper], dtype=np.float64)
    if left_values.size == 0:
        return float("nan")
    return _spearman_correlation(left_values, right_values)


def _alignment_metrics_table(
    context: WorkspaceContext,
    *,
    alignment_id: str,
    left_view_id: str,
    right_view_id: str,
    metadata_view_id: str,
    left_margin_source: str | None,
    right_margin_source: str | None,
    margin_deltas: list[dict[str, Any]],
    sample_size: int,
    sample_group_column: str | None,
    where: dict[str, Any] | None,
    table_sample_only: bool,
    seed: int,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object], list[dict[str, object]]]:
    metadata_table, metadata_inputs = _row_metadata_from_alignment(
        context,
        alignment_id=alignment_id,
        metadata_view_id=metadata_view_id,
    )
    metadata_rows = metadata_table.to_pylist()
    selected_indices = np.arange(metadata_table.num_rows, dtype=np.int64)
    if where is not None:
        matched_indices = _select_indices(metadata_rows, where)
        selected_indices = np.asarray(matched_indices, dtype=np.int64)
        metadata_rows = [metadata_rows[int(index)] for index in selected_indices]
        metadata_table = metadata_table.take(pa.array(selected_indices, type=pa.int64()))
    sampled_indices = _sample_alignment_indices(
        metadata_table,
        sample_size=min(sample_size, metadata_table.num_rows),
        group_column=sample_group_column,
        seed=seed,
    )
    sample_index_array = np.asarray(sampled_indices, dtype=np.int64)
    if table_sample_only:
        selected_indices = selected_indices[sample_index_array]
        metadata_rows = [metadata_rows[int(index)] for index in sample_index_array]
        metadata_table = metadata_table.take(pa.array(sample_index_array, type=pa.int64()))
        left_matrix = _sampled_alignment_scope_matrix(
            context,
            alignment_id=alignment_id,
            view_id=left_view_id,
            alignment_indices=selected_indices,
        )
        right_matrix = _sampled_alignment_scope_matrix(
            context,
            alignment_id=alignment_id,
            view_id=right_view_id,
            alignment_indices=selected_indices,
        )
        sampled_left = left_matrix
        sampled_right = right_matrix
    else:
        left_matrix, _, _, _ = resolve_view_scope(
            context, view_id=left_view_id, sample_id=None, alignment_id=alignment_id
        )
        right_matrix, _, _, _ = resolve_view_scope(
            context, view_id=right_view_id, sample_id=None, alignment_id=alignment_id
        )
        if left_matrix.shape != right_matrix.shape:
            raise ContractViolationError(
                "alignment_metrics requires left and right aligned matrices to share one shape"
            )
        if metadata_table.num_rows != left_matrix.shape[0]:
            raise ContractViolationError("alignment metadata rows are not aligned to the aligned matrix row count")
        if where is not None:
            left_matrix = np.asarray(left_matrix[selected_indices], dtype=np.float32)
            right_matrix = np.asarray(right_matrix[selected_indices], dtype=np.float32)
        sampled_left = np.asarray(left_matrix[sampled_indices], dtype=np.float32)
        sampled_right = np.asarray(right_matrix[sampled_indices], dtype=np.float32)
    if left_matrix.shape != right_matrix.shape:
        raise ContractViolationError("alignment_metrics requires left and right aligned matrices to share one shape")
    geometry_distance_correlation = _cosine_distance_correlation(sampled_left, sampled_right)
    left_norm = _normalized_geometry_rows(left_matrix)
    right_norm = _normalized_geometry_rows(right_matrix)
    self_cosine = np.asarray(np.sum(left_norm * right_norm, axis=1), dtype=np.float32)
    shift_l2 = np.asarray(np.linalg.norm(left_norm - right_norm, axis=1), dtype=np.float32)
    table = metadata_table
    table = _replace_or_append_column(table, "context_self_cosine", self_cosine)
    table = _replace_or_append_column(table, "context_shift_l2", shift_l2)
    inputs = list(metadata_inputs)
    left_matrix_path, _ = _view_paths(context, left_view_id)
    right_matrix_path, _ = _view_paths(context, right_view_id)
    inputs.extend(
        [
            ScalarInputRef(kind="view_matrix", artifact_id=left_view_id, path=left_matrix_path),
            ScalarInputRef(kind="view_matrix", artifact_id=right_view_id, path=right_matrix_path),
        ]
    )
    _, _, mapping_path = _alignment_paths(context, alignment_id)
    inputs.append(ScalarInputRef(kind="alignment_set", artifact_id=alignment_id, path=mapping_path))

    if margin_deltas:
        if left_margin_source is None or right_margin_source is None:
            raise ContractViolationError(
                "alignment_metrics margin_deltas require left_margin_source and right_margin_source"
            )
        alignment_rows, left_key_columns, right_key_columns = _alignment_projection(context, alignment_id)
        left_margin_path = _scalar_table_path(context, left_margin_source)
        right_margin_path = _scalar_table_path(context, right_margin_source)
        inputs.extend(
            [
                ScalarInputRef(kind="scalar_table", artifact_id=left_margin_source, path=left_margin_path),
                ScalarInputRef(kind="scalar_table", artifact_id=right_margin_source, path=right_margin_path),
            ]
        )
        aligned_left_margin = _project_table_to_alignment(
            alignment_rows=alignment_rows,
            candidate_table=read_table(left_margin_path),
            left_key_columns=left_key_columns,
            right_key_columns=right_key_columns,
            candidate_label=left_margin_source,
        )
        aligned_right_margin = _project_table_to_alignment(
            alignment_rows=alignment_rows,
            candidate_table=read_table(right_margin_path),
            left_key_columns=left_key_columns,
            right_key_columns=right_key_columns,
            candidate_label=right_margin_source,
        )
        for delta in margin_deltas:
            left_column = str(_require_param(delta, "left_column"))
            right_column = str(_require_param(delta, "right_column"))
            output_column = str(_require_param(delta, "output_column"))
            if left_column not in aligned_left_margin.column_names:
                raise ContractViolationError(f"left margin source is missing {left_column!r}")
            if right_column not in aligned_right_margin.column_names:
                raise ContractViolationError(f"right margin source is missing {right_column!r}")
            left_values = np.asarray(aligned_left_margin[left_column].to_pylist(), dtype=np.float32)[selected_indices]
            right_values = np.asarray(aligned_right_margin[right_column].to_pylist(), dtype=np.float32)[
                selected_indices
            ]
            table = _replace_or_append_column(
                table, output_column, np.asarray(left_values - right_values, dtype=np.float32)
            )
    repeated_correlation = np.repeat(np.float32(geometry_distance_correlation), table.num_rows)
    table = _replace_or_append_column(table, "geometry_distance_correlation", repeated_correlation)
    if table_sample_only:
        sample_rows = list(metadata_rows)
    else:
        sample_rows = metadata_table.take(pa.array(sampled_indices, type=pa.int64())).to_pylist()
    stats = {
        "alignment_id": alignment_id,
        "rows": int(table.num_rows),
        "geometry_distance_correlation": geometry_distance_correlation,
        "sample_size": len(sample_rows),
    }
    if where is not None:
        stats["where"] = dict(where)
    if table_sample_only:
        stats["table_sample_only"] = True
    return table, inputs, stats, sample_rows


def _join_tables_on_keys(
    *,
    tables: list[tuple[str, pa.Table]],
    key_columns: list[str],
) -> pa.Table:
    mappings: list[dict[tuple[object, ...], dict[str, object]]] = []
    shared: set[tuple[object, ...]] | None = None
    first_order: list[tuple[object, ...]] = []
    seen_columns: set[str] = set()
    ordered_columns: list[str] = []
    for index, (label, table) in enumerate(tables):
        missing = [column for column in key_columns if column not in table.column_names]
        if missing:
            raise ContractViolationError(f"{label} is missing join keys {missing}")
        mapping: dict[tuple[object, ...], dict[str, object]] = {}
        order: list[tuple[object, ...]] = []
        for row in table.to_pylist():
            key = tuple(row[column] for column in key_columns)
            if key in mapping:
                raise ContractViolationError(f"{label} is non-unique on join keys {key_columns}")
            mapping[key] = row
            order.append(key)
        mappings.append(mapping)
        shared = set(mapping) if shared is None else shared.intersection(mapping)
        if index == 0:
            first_order = order
            ordered_columns.extend(table.column_names)
            seen_columns.update(table.column_names)
        else:
            duplicate_columns = [
                column for column in table.column_names if column not in key_columns and column in seen_columns
            ]
            if duplicate_columns:
                raise ContractViolationError(f"{label} reuses non-key columns in join: {duplicate_columns}")
            new_columns = [column for column in table.column_names if column not in key_columns]
            ordered_columns.extend(new_columns)
            seen_columns.update(new_columns)
    if not shared:
        raise ContractViolationError("join_view_columns produced an empty key intersection")
    output_rows: list[dict[str, object]] = []
    for key in first_order:
        if key not in shared:
            continue
        merged: dict[str, object] = {}
        for mapping in mappings:
            row = mapping[key]
            for column in ordered_columns:
                if column in merged:
                    continue
                if column in row:
                    merged[column] = row[column]
        output_rows.append(merged)
    return pa.Table.from_pylist(output_rows)


def _join_view_columns_table(
    context: WorkspaceContext,
    *,
    sources: list[dict[str, Any]],
    key_columns: list[str],
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    tables: list[tuple[str, pa.Table]] = []
    inputs: list[ScalarInputRef] = []
    for source_spec in sources:
        view_id = str(_require_param(source_spec, "view_id"))
        columns = [str(column) for column in _require_param(source_spec, "columns")]
        rename = {
            str(key): str(value) for key, value in dict(_optional_param(source_spec, "rename", default={})).items()
        }
        _, rows_path = _view_paths(context, view_id)
        rows_table = read_table(rows_path)
        missing = [column for column in columns if column not in rows_table.column_names]
        if missing:
            raise ContractViolationError(f"view {view_id} is missing requested columns: {missing}")
        selected = rows_table.select(columns)
        if rename:
            missing_rename = [column for column in rename if column not in selected.column_names]
            if missing_rename:
                raise ContractViolationError(f"view {view_id} cannot rename missing columns: {missing_rename}")
            renamed_columns = [rename.get(name, name) for name in selected.column_names]
            selected = selected.rename_columns(renamed_columns)
        tables.append((view_id, selected))
        inputs.append(ScalarInputRef(kind="view_rows", artifact_id=view_id, path=rows_path))
    joined = _join_tables_on_keys(tables=tables, key_columns=key_columns)
    return joined, inputs, {"source_count": len(sources), "join_keys": key_columns}


def _neighbor_label_enrichment(rows_table: pa.Table, indices: np.ndarray, *, label_column: str) -> float:
    if label_column not in rows_table.column_names:
        raise ContractViolationError(f"neighbor rows are missing label column {label_column!r}")
    labels = [str(value) for value in rows_table[label_column].combine_chunks().to_pylist()]
    global_counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return float("nan")
    enrichments: list[float] = []
    k = int(indices.shape[1])
    for row_index, label in enumerate(labels):
        same_hits = sum(1 for index in indices[row_index] if labels[int(index)] == label)
        neighbor_fraction = same_hits / max(k, 1)
        background_fraction = global_counts[label] / total
        enrichments.append(neighbor_fraction - background_fraction)
    return float(np.mean(np.asarray(enrichments, dtype=np.float64)))


def _candidate_descriptor(candidate_id: str) -> dict[str, object]:
    family = "unknown"
    if "intermediate_embedding" in candidate_id:
        family = "intermediate_embedding"
    elif "output_layer_mean" in candidate_id:
        family = "output_layer_mean"
    elif candidate_id.startswith("log_likelihood_per_token_"):
        family = "log_likelihood"
    model = "20b" if "_20b_" in candidate_id else "7b" if "_7b_" in candidate_id else None
    scope = (
        "full_context_1kb"
        if "full_context_1kb" in candidate_id
        else "merged_anchor_insert_seq_mean"
        if "anchor_60bp" in candidate_id
        else None
    )
    label_parts = [part for part in [model.upper() if model is not None else None, family, scope] if part]
    return {
        "candidate_family": family,
        "candidate_model": model,
        "candidate_scope": scope,
        "candidate_label": " ".join(str(part).replace("_", " ") for part in label_parts) or candidate_id,
    }


def _task_reference_neighbor_metrics(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    label_column: str,
    positive_values: set[str],
    reference_labels: set[str],
) -> tuple[list[float], list[float]] | None:
    normalized_reference_labels = {label.lower() for label in reference_labels}
    row_labels = [str(row.get("usr_label__primary") or "").strip().lower() for row in rows]
    label_values = [str(row.get(label_column) or "") for row in rows]
    reference_indices = {
        index for index, reference_label in enumerate(row_labels) if reference_label in normalized_reference_labels
    }
    positive_indices = [index for index, value in enumerate(label_values) if value in positive_values]
    if not reference_indices or not positive_indices:
        return None
    hit_values: list[float] = []
    ranks: list[float] = []
    neighbor_count = int(indices.shape[1]) if indices.ndim == 2 else 0
    for row_index in positive_indices:
        neighbor_indices = [int(value) for value in indices[row_index].tolist()]
        hits = [rank + 1 for rank, neighbor_index in enumerate(neighbor_indices) if neighbor_index in reference_indices]
        hit_values.append(1.0 if hits else 0.0)
        ranks.append(float(hits[0] if hits else neighbor_count + 1))
    return hit_values, ranks


def _reference_neighbor_metrics(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    label_column: str,
    ethanol_values: set[str],
    cipro_values: set[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ethanol_summary = _task_reference_neighbor_metrics(
        rows,
        indices,
        label_column=label_column,
        positive_values=ethanol_values,
        reference_labels={"spyp"},
    )
    cipro_summary = _task_reference_neighbor_metrics(
        rows,
        indices,
        label_column=label_column,
        positive_values=cipro_values,
        reference_labels={"sulap"},
    )
    summaries = [summary for summary in [ethanol_summary, cipro_summary] if summary is not None]
    if summaries:
        hit_values = [value for summary in summaries for value in summary[0]]
        rank_values = [value for summary in summaries for value in summary[1]]
        metrics["reference_in_knn_rate"] = float(np.mean(np.asarray(hit_values, dtype=np.float64)))
        metrics["reference_neighbor_topk_censored_rank_median"] = float(
            np.median(np.asarray(rank_values, dtype=np.float64))
        )
    return metrics


def _candidate_metric_rows_by_candidate(source_table: pa.Table) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in source_table.to_pylist():
        candidate_id = str(row["candidate_id"])
        metric_id = str(row["metric_id"])
        if metric_id in grouped[candidate_id]:
            raise ContractViolationError(
                f"candidate metric source is non-unique on candidate_id={candidate_id!r}, metric_id={metric_id!r}"
            )
        grouped[candidate_id][metric_id] = row
    return grouped


def _candidate_metric_pairs_table(
    context: WorkspaceContext,
    *,
    source_scalar: str,
    x_metric_id: str,
    y_metric_id: str,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    source_path = _scalar_table_path(context, source_scalar)
    grouped = _candidate_metric_rows_by_candidate(read_table(source_path))
    rows: list[dict[str, object]] = []
    for candidate_id, metrics in sorted(grouped.items()):
        if x_metric_id not in metrics or y_metric_id not in metrics:
            continue
        descriptor = _candidate_descriptor(candidate_id)
        x_row = metrics[x_metric_id]
        y_row = metrics[y_metric_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                **descriptor,
                "x_metric_id": x_metric_id,
                "y_metric_id": y_metric_id,
                "x_display_name": x_row["display_name"],
                "y_display_name": y_row["display_name"],
                "x_metric_value": float(x_row["metric_value"]),
                "y_metric_value": float(y_row["metric_value"]),
                "x_direction": x_row["direction"],
                "y_direction": y_row["direction"],
                "x_unit": x_row["unit"],
                "y_unit": y_row["unit"],
            }
        )
    return (
        pa.Table.from_pylist(rows),
        [ScalarInputRef(kind="scalar_table", artifact_id=source_scalar, path=source_path)],
        {"rows": len(rows), "source_scalar": source_scalar, "x_metric_id": x_metric_id, "y_metric_id": y_metric_id},
    )


def _candidate_metric_bars_table(
    context: WorkspaceContext,
    *,
    source_scalar: str,
    metric_ids: list[str],
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    source_path = _scalar_table_path(context, source_scalar)
    grouped = _candidate_metric_rows_by_candidate(read_table(source_path))
    rows: list[dict[str, object]] = []
    for candidate_id, metrics in sorted(grouped.items()):
        descriptor = _candidate_descriptor(candidate_id)
        for metric_id in metric_ids:
            metric_row = metrics.get(metric_id)
            if metric_row is None:
                continue
            rows.append(
                {
                    "category": metric_id,
                    "label": candidate_id,
                    "panel_id": metric_row["display_name"],
                    "metric_value": float(metric_row["metric_value"]),
                    "display_name": metric_row["display_name"],
                    "direction": metric_row["direction"],
                    "unit": metric_row["unit"],
                    **descriptor,
                }
            )
    return (
        pa.Table.from_pylist(rows),
        [ScalarInputRef(kind="scalar_table", artifact_id=source_scalar, path=source_path)],
        {"rows": len(rows), "source_scalar": source_scalar, "metric_ids": metric_ids},
    )


def _representation_scorecard_table(
    context: WorkspaceContext,
    *,
    candidates: list[dict[str, Any]],
    label_column: str,
    ethanol_values: set[str],
    cipro_values: set[str],
    dual_values: set[str],
    neighbor_label_enrichments: list[dict[str, str]] | None = None,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    config = getattr(context, "config", None)
    validate_metric_registry(config)
    output_rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    seen_inputs: set[tuple[str, str, str]] = set()

    def add_input(kind: str, artifact_id: str, path: Path) -> None:
        key = (kind, artifact_id, path.as_posix())
        if key in seen_inputs:
            return
        seen_inputs.add(key)
        inputs.append(ScalarInputRef(kind=kind, artifact_id=artifact_id, path=path))

    for candidate in candidates:
        candidate_id = str(_require_param(candidate, "candidate_id"))
        candidate_metrics: dict[str, float] = {}

        wildtype_source = _optional_param(candidate, "wildtype_source", default=None)
        if wildtype_source is not None:
            path = _scalar_table_path(context, str(wildtype_source))
            add_input("scalar_table", str(wildtype_source), path)
            rows = read_table(path).to_pylist()
            ethanol_scores = np.asarray(
                [float(row["wildtype_margin_ethanol_vs_control"]) for row in rows], dtype=np.float64
            )
            cipro_scores = np.asarray(
                [float(row["wildtype_margin_cipro_vs_control"]) for row in rows], dtype=np.float64
            )
            candidate_metrics["wildtype_margin_ethanol_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=ethanol_scores,
            )["auroc"]
            candidate_metrics["wildtype_margin_ethanol_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=ethanol_scores,
            )["auprc"]
            candidate_metrics["wildtype_margin_cipro_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=cipro_scores,
            )["auroc"]
            candidate_metrics["wildtype_margin_cipro_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=cipro_scores,
            )["auprc"]
            if dual_values:
                dual_scores = dual_joint_margin(ethanol_scores, cipro_scores)
                candidate_metrics["wildtype_margin_dual_joint_auroc"] = binary_metrics(
                    rows=rows,
                    label_column=label_column,
                    positive_values=dual_values,
                    score_values=dual_scores,
                )["auroc"]
                candidate_metrics["wildtype_margin_dual_joint_auprc"] = binary_metrics(
                    rows=rows,
                    label_column=label_column,
                    positive_values=dual_values,
                    score_values=dual_scores,
                )["auprc"]

        synthetic_source = _optional_param(candidate, "synthetic_source", default=None)
        if synthetic_source is not None:
            path = _scalar_table_path(context, str(synthetic_source))
            add_input("scalar_table", str(synthetic_source), path)
            rows = read_table(path).to_pylist()
            ethanol_scores = np.asarray(
                [float(row["synthetic_margin_ethanol_vs_background"]) for row in rows], dtype=np.float64
            )
            cipro_scores = np.asarray(
                [float(row["synthetic_margin_cipro_vs_background"]) for row in rows], dtype=np.float64
            )
            candidate_metrics["synthetic_margin_ethanol_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=ethanol_scores,
            )["auroc"]
            candidate_metrics["synthetic_margin_ethanol_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=ethanol_scores,
            )["auprc"]
            candidate_metrics["synthetic_margin_cipro_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=cipro_scores,
            )["auroc"]
            candidate_metrics["synthetic_margin_cipro_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=cipro_scores,
            )["auprc"]
            if dual_values:
                dual_scores = dual_joint_margin(ethanol_scores, cipro_scores)
                candidate_metrics["synthetic_margin_dual_joint_auroc"] = binary_metrics(
                    rows=rows,
                    label_column=label_column,
                    positive_values=dual_values,
                    score_values=dual_scores,
                )["auroc"]
                candidate_metrics["synthetic_margin_dual_joint_auprc"] = binary_metrics(
                    rows=rows,
                    label_column=label_column,
                    positive_values=dual_values,
                    score_values=dual_scores,
                )["auprc"]

        scalar_view_id = _optional_param(candidate, "value_view_id", default=None)
        scalar_column = _optional_param(candidate, "value_column", default=None)
        if scalar_view_id is not None and scalar_column is not None:
            _, rows_path = _view_paths(context, str(scalar_view_id))
            add_input("view_rows", str(scalar_view_id), rows_path)
            table = read_table(rows_path)
            if scalar_column not in table.column_names:
                raise ContractViolationError(f"view {scalar_view_id} is missing score column {scalar_column!r}")
            rows = table.to_pylist()
            scores = np.asarray([float(row[scalar_column]) for row in rows], dtype=np.float64)
            candidate_metrics["scalar_ethanol_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=scores,
            )["auroc"]
            candidate_metrics["scalar_ethanol_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=ethanol_values,
                score_values=scores,
            )["auprc"]
            candidate_metrics["scalar_cipro_auroc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=scores,
            )["auroc"]
            candidate_metrics["scalar_cipro_auprc"] = binary_metrics(
                rows=rows,
                label_column=label_column,
                positive_values=cipro_values,
                score_values=scores,
            )["auprc"]

        neighbors_id = _optional_param(candidate, "neighbors_id", default=None)
        if neighbors_id is not None:
            rows_path, indices_path, _ = _neighbor_paths(context, str(neighbors_id))
            add_input("neighbor_rows", str(neighbors_id), rows_path)
            add_input("neighbor_set", str(neighbors_id), indices_path)
            rows_table = read_table(rows_path)
            indices = np.asarray(read_matrix(indices_path, mmap_mode=None), dtype=np.int64)
            if indices.shape[0] != rows_table.num_rows:
                raise ContractViolationError(
                    f"neighbor artifact {neighbors_id!r} row count does not match its neighbor index matrix"
                )
            enrichment_specs = neighbor_label_enrichments or [
                {
                    "label_column": "design_family",
                    "metric_name": "knn_design_family_enrichment_delta",
                }
            ]
            for enrichment_spec in enrichment_specs:
                enrichment_label_column = str(_require_param(enrichment_spec, "label_column"))
                enrichment_metric_name = str(_require_param(enrichment_spec, "metric_name", "metric_id"))
                if enrichment_label_column not in rows_table.column_names:
                    raise ContractViolationError(
                        f"neighbor rows are missing configured label enrichment column {enrichment_label_column!r}"
                    )
                candidate_metrics[enrichment_metric_name] = _neighbor_label_enrichment(
                    rows_table,
                    indices,
                    label_column=enrichment_label_column,
                )
            candidate_metrics.update(
                _reference_neighbor_metrics(
                    rows_table.to_pylist(),
                    indices,
                    label_column=label_column,
                    ethanol_values=ethanol_values,
                    cipro_values=cipro_values,
                )
            )

        context_source = _optional_param(candidate, "context_source", default=None)
        if context_source is not None:
            path = _scalar_table_path(context, str(context_source))
            add_input("scalar_table", str(context_source), path)
            table = read_table(path)
            if table.num_rows == 0:
                raise ContractViolationError(f"context source {context_source!r} is empty")
            context_rows = table.to_pylist()
            candidate_metrics["context_self_cosine_median"] = float(
                np.median(np.asarray(table["context_self_cosine"].to_pylist(), dtype=np.float64))
            )
            candidate_metrics["context_shift_l2_median"] = float(
                np.median(np.asarray(table["context_shift_l2"].to_pylist(), dtype=np.float64))
            )
            if "geometry_distance_correlation" in table.column_names:
                candidate_metrics["geometry_distance_correlation"] = float(
                    context_rows[0]["geometry_distance_correlation"]
                )

        agreement_id = _optional_param(candidate, "agreement_id", default=None)
        if agreement_id is not None:
            path = _agreement_summary_path(context, str(agreement_id))
            add_input("agreement_set", str(agreement_id), path)
            summary = read_json(path)
            knn_summary = summary.get("knn_overlap")
            if isinstance(knn_summary, dict):
                overlap_value = knn_summary.get(
                    "mean_neighbor_overlap_fraction",
                    knn_summary.get("mean_overlap_fraction"),
                )
                if isinstance(overlap_value, int | float):
                    candidate_metrics["neighbor_overlap_fraction"] = float(overlap_value)
            landmark_summary = summary.get("landmark_neighbor_overlap")
            if isinstance(landmark_summary, dict) and "mean_jaccard_overlap" in landmark_summary:
                candidate_metrics["landmark_neighbor_jaccard"] = float(landmark_summary["mean_jaccard_overlap"])

        reducer_id = _optional_param(candidate, "reducer_id", default=None)
        if reducer_id is not None:
            path = _reducer_summary_path(context, str(reducer_id))
            add_input("reducer", str(reducer_id), path)
            summary = read_json(path)
            explained_ratio = [float(value) for value in summary.get("explained_variance_ratio", [])]
            candidate_metrics["selected_rank"] = float(summary.get("output_dims", 0))
            candidate_metrics["explained_variance_captured"] = float(sum(explained_ratio))
            candidate_metrics["effective_rank"] = _effective_rank(explained_ratio)

        for metric_name, metric_value in sorted(candidate_metrics.items()):
            definition = resolve_metric_definition(metric_name, config=config)
            output_rows.append(
                {
                    "candidate_id": candidate_id,
                    "view_id": candidate_id,
                    "metric_name": metric_name,
                    "metric_id": definition.metric_id,
                    "metric_value": metric_value,
                    "value": metric_value,
                    "metric_family": definition.metric_family,
                    "evidence_tier": definition.evidence_tier,
                    "task_id": definition.task_id,
                    "mathematical_definition": definition.mathematical_definition,
                    "unit": definition.unit,
                    "direction": definition.direction,
                    "aggregation_level": definition.aggregation_level,
                    "higher_is_better": definition.higher_is_better,
                    "display_name": definition.display_name,
                    "definition_version": definition.definition_version,
                }
            )

    return pa.Table.from_pylist(output_rows), inputs, {"candidate_count": len(candidates), "metrics": len(output_rows)}


def build_scalar_artifact(
    context: WorkspaceContext,
    *,
    scalar_id: str,
    builder_kind: str,
    params: dict[str, Any],
) -> BuiltScalarArtifact:
    artifact_dir = context.output_root / "scalars" / scalar_id
    extra_outputs: list[tuple[str, str]] = []
    stats: dict[str, object]

    preassay_artifact = build_preassay_scalar_artifact(
        context,
        scalar_id=scalar_id,
        builder_kind=builder_kind,
        params=params,
    )
    if preassay_artifact is not None:
        return preassay_artifact

    if builder_kind == "dataset_overview":
        source_ids = [
            str(value) for value in _optional_param(params, "source_ids", default=list(context.config.sources))
        ]
        if not source_ids:
            raise ContractViolationError("dataset_overview requires at least one source")

        raw_dimensions = params.get("dimensions")
        if not isinstance(raw_dimensions, list) or not raw_dimensions:
            raise ContractViolationError("dataset_overview requires configured dimensions")

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

        dimension_specs: list[dict[str, object]] = []
        for raw_dimension in raw_dimensions:
            if not isinstance(raw_dimension, dict):
                raise ContractViolationError("dataset_overview dimensions must be mapping objects")
            dimension = str(raw_dimension.get("dimension") or raw_dimension.get("id") or "").strip()
            column = str(raw_dimension.get("column") or "").strip()
            if not dimension or not column:
                raise ContractViolationError("dataset_overview dimensions must declare dimension and column")
            dimension_specs.append(
                {
                    "dimension": dimension,
                    "label": str(raw_dimension.get("label") or humanize_label(dimension)).strip(),
                    "column": column,
                    "category_order": _category_order(raw_dimension.get("category_order", [])),
                    "category_labels": {
                        str(key): str(value)
                        for key, value in dict(raw_dimension.get("category_labels", {}) or {}).items()
                    },
                }
            )

        dimension_columns = [str(spec["column"]) for spec in dimension_specs]

        def _dataset_category_label(spec: dict[str, object], category: str) -> str:
            labels = spec.get("category_labels", {})
            if isinstance(labels, dict) and category in labels:
                return str(labels[category])
            return humanize_label(category)

        def _metadata_value(row: dict[str, object], column: str) -> object:
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
                str(spec["dimension"]): Counter(str(_metadata_value(row, str(spec["column"]))) for row in row_dicts)
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
            ordered_categories = _ordered_dataset_overview_categories(counts, category_order)
            dimension_total = sum(int(counts.get(category, 0)) for category, _ in ordered_categories)
            if dimension_total != denominator:
                raise ContractViolationError(
                    f"dataset_overview dimension {dimension!r} sums to {dimension_total}, expected {denominator}"
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
            outputs=extra_outputs,
            stats=stats,
        )

    if builder_kind == "similarity_margin":
        table, inputs, stats = _similarity_margin_table(
            context,
            view_id=str(_require_param(params, "view_id")),
            margin_pairs=[dict(value) for value in _require_param(params, "margin_pairs")],
            alignment_id=_optional_param(params, "alignment_id", default=None),
        )
        write_table(table, artifact_dir / "table.parquet")
        return BuiltScalarArtifact(
            artifact_dir=artifact_dir,
            rows=table.num_rows,
            columns=table.column_names,
            inputs=inputs,
            outputs=extra_outputs,
            stats=stats,
        )

    if builder_kind == "cohort_similarity_margin":
        table, inputs, stats = _cohort_similarity_margin_table(
            context,
            view_id=str(_require_param(params, "view_id")),
            sample_id=_optional_param(params, "sample_id", default=None),
            cohort_column=str(_require_param(params, "cohort_column")),
            margin_pairs=[dict(value) for value in _require_param(params, "margin_pairs")],
            leave_one_out=bool(_optional_param(params, "leave_one_out", default=False)),
            balance_group_column=_optional_param(params, "balance_group_column", default=None),
            balance_columns=[str(value) for value in _optional_param(params, "balance_columns", default=[])] or None,
            balance_reference_only=bool(_optional_param(params, "balance_reference_only", default=False)),
            required_group_values={str(value) for value in _optional_param(params, "required_group_values", default=[])}
            or None,
            exclude_group_values={str(value) for value in _optional_param(params, "exclude_group_values", default=[])}
            or None,
            seed=(
                int(_optional_param(params, "seed", default=context.config.defaults.random_seed))
                if "seed" in params
                else None
            ),
        )
        write_table(table, artifact_dir / "table.parquet")
        return BuiltScalarArtifact(
            artifact_dir=artifact_dir,
            rows=table.num_rows,
            columns=table.column_names,
            inputs=inputs,
            outputs=extra_outputs,
            stats=stats,
        )

    if builder_kind == "alignment_metrics":
        table, inputs, stats, sample_rows = _alignment_metrics_table(
            context,
            alignment_id=str(_require_param(params, "alignment_id")),
            left_view_id=str(_require_param(params, "left_view_id")),
            right_view_id=str(_require_param(params, "right_view_id")),
            metadata_view_id=str(
                _optional_param(params, "metadata_view_id", default=_require_param(params, "left_view_id"))
            ),
            left_margin_source=_optional_param(params, "left_margin_source", default=None),
            right_margin_source=_optional_param(params, "right_margin_source", default=None),
            margin_deltas=[dict(value) for value in _optional_param(params, "margin_deltas", default=[])],
            sample_size=int(_optional_param(params, "sample_size", default=256)),
            sample_group_column=_optional_param(params, "sample_group_column", default=None),
            where=_optional_param(params, "where", default=None),
            table_sample_only=bool(_optional_param(params, "table_sample_only", default=False)),
            seed=int(_optional_param(params, "seed", default=context.config.defaults.random_seed)),
        )
        write_table(table, artifact_dir / "table.parquet")
        write_json(artifact_dir / "sample_ids.json", sample_rows)
        extra_outputs = [("sample_ids.json", "application/json")]
        return BuiltScalarArtifact(
            artifact_dir=artifact_dir,
            rows=table.num_rows,
            columns=table.column_names,
            inputs=inputs,
            outputs=extra_outputs,
            stats=stats,
        )

    if builder_kind == "join_view_columns":
        table, inputs, stats = _join_view_columns_table(
            context,
            sources=[dict(value) for value in _require_param(params, "sources")],
            key_columns=[str(value) for value in _require_param(params, "key_columns")],
        )
        write_table(table, artifact_dir / "table.parquet")
        return BuiltScalarArtifact(
            artifact_dir=artifact_dir,
            rows=table.num_rows,
            columns=table.column_names,
            inputs=inputs,
            outputs=extra_outputs,
            stats=stats,
        )

    if builder_kind in {"candidate_metrics_long", "representation_scorecard"}:
        table, inputs, stats = _representation_scorecard_table(
            context,
            candidates=[dict(value) for value in _require_param(params, "candidates")],
            label_column=str(_optional_param(params, "label_column", default="design_family")),
            ethanol_values={
                str(value)
                for value in _optional_param(
                    params,
                    "ethanol_values",
                    default=["ethanol", "ethanol_ciprofloxacin"],
                )
            },
            cipro_values={
                str(value)
                for value in _optional_param(
                    params,
                    "cipro_values",
                    default=["ciprofloxacin", "ethanol_ciprofloxacin"],
                )
            },
            dual_values={
                str(value) for value in _optional_param(params, "dual_values", default=["ethanol_ciprofloxacin"])
            },
            neighbor_label_enrichments=[
                {str(key): str(value) for key, value in dict(item).items()}
                for item in _optional_param(params, "neighbor_label_enrichments", default=[])
            ],
        )
        write_table(table, artifact_dir / "table.parquet")
        return BuiltScalarArtifact(
            artifact_dir=artifact_dir,
            rows=table.num_rows,
            columns=table.column_names,
            inputs=inputs,
            outputs=extra_outputs,
            stats=stats,
        )

    if builder_kind == "candidate_metric_pairs":
        table, inputs, stats = _candidate_metric_pairs_table(
            context,
            source_scalar=str(_require_param(params, "source_scalar")),
            x_metric_id=str(_require_param(params, "x_metric_id")),
            y_metric_id=str(_require_param(params, "y_metric_id")),
        )
        write_table(table, artifact_dir / "table.parquet")
        return BuiltScalarArtifact(
            artifact_dir=artifact_dir,
            rows=table.num_rows,
            columns=table.column_names,
            inputs=inputs,
            outputs=extra_outputs,
            stats=stats,
        )

    if builder_kind == "candidate_metric_bars":
        table, inputs, stats = _candidate_metric_bars_table(
            context,
            source_scalar=str(_require_param(params, "source_scalar")),
            metric_ids=[str(value) for value in _require_param(params, "metric_ids")],
        )
        write_table(table, artifact_dir / "table.parquet")
        return BuiltScalarArtifact(
            artifact_dir=artifact_dir,
            rows=table.num_rows,
            columns=table.column_names,
            inputs=inputs,
            outputs=extra_outputs,
            stats=stats,
        )

    raise ContractViolationError(f"unsupported scalar.build kind: {builder_kind!r}")
