"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/scalars/builders/tf_axis_orientation.py

TF-axis orientation scalar builders for native promoter overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...geometry.preprocessing import try_l2_normalize_vector
from ...io.matrix_io import read_matrix
from ...io.parquet_io import read_table, write_table
from ...stats.rank import rankdata_average
from ...workspaces.loader import WorkspaceContext
from ..common import BuiltScalarArtifact, ScalarInputRef, _optional_param, _require_param

_TF_AXIS_DEFAULT_BASE_COLUMNS = (
    "id",
    "subject_id",
    "source",
    "promoter_id",
    "promoter_name",
    "core60_sequence",
    "associated_gene_or_TU",
)
_TF_AXIS_DERIVED_COLUMNS = (
    "tf_bin",
    "embedding_view",
    "ethanolness",
    "ciproness",
    "distance_to_ethanol_centroid",
    "distance_to_cipro_centroid",
    "distance_to_background_centroid",
)
_TF_AXIS_MATRIX_CHUNK_ROWS = 4096
_TF_AXIS_EPS = 1e-8


def _view_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path]:
    matrix_path = context.output_root / "views" / view_id / "matrix.npy"
    rows_path = context.output_root / "views" / view_id / "rows.parquet"
    if not matrix_path.is_file() or not rows_path.is_file():
        raise MissingArtifactError(f"view artifact is missing for scalar.build: {view_id}")
    return matrix_path, rows_path


def _scalar_table_path(context: WorkspaceContext, scalar_id: str) -> Path:
    path = context.output_root / "scalars" / scalar_id / "table.parquet"
    if not path.is_file():
        raise MissingArtifactError(f"scalar artifact is missing for scalar.build: {scalar_id}")
    return path


def _truthy_metadata_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "t", "yes", "y"}


def _tf_bin(row: dict[str, object], *, ethanol_columns: list[str], cipro_columns: list[str]) -> str:
    has_ethanol = any(_truthy_metadata_value(row.get(column)) for column in ethanol_columns)
    has_cipro = any(_truthy_metadata_value(row.get(column)) for column in cipro_columns)
    if has_ethanol and has_cipro:
        return "mixed"
    if has_ethanol:
        return "ethanol_TF"
    if has_cipro:
        return "lexA_TF"
    return "neither"


def _dedupe_columns(columns: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        output.append(column)
    return output


def _tf_axis_standardization_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ContractViolationError("tf_axis_orientation_audit expects a 2D matrix")
    row_count, dims = matrix.shape
    if row_count == 0:
        raise ContractViolationError("tf_axis_orientation_audit cannot normalize an empty matrix")
    total = np.zeros(dims, dtype=np.float64)
    for start in range(0, row_count, _TF_AXIS_MATRIX_CHUNK_ROWS):
        chunk = np.asarray(matrix[start : start + _TF_AXIS_MATRIX_CHUNK_ROWS], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ContractViolationError("tf_axis_orientation_audit encountered non-finite matrix values")
        total += np.sum(chunk, axis=0, dtype=np.float64)
    mean = total / float(row_count)
    squared = np.zeros(dims, dtype=np.float64)
    for start in range(0, row_count, _TF_AXIS_MATRIX_CHUNK_ROWS):
        chunk = np.asarray(matrix[start : start + _TF_AXIS_MATRIX_CHUNK_ROWS], dtype=np.float32)
        diff = np.asarray(chunk, dtype=np.float64) - mean
        squared += np.sum(diff * diff, axis=0, dtype=np.float64)
    scales = np.asarray(np.sqrt(squared / float(row_count)), dtype=np.float32)
    zero_mask = scales <= _TF_AXIS_EPS
    scales = np.where(zero_mask, 1.0, scales).astype(np.float32, copy=False)
    return mean.astype(np.float32, copy=False), scales, np.asarray(zero_mask, dtype=bool)


def _tf_axis_normalize_chunk(
    chunk: np.ndarray,
    *,
    mean: np.ndarray,
    scales: np.ndarray,
    zero_mask: np.ndarray,
) -> np.ndarray:
    working = np.asarray(
        (np.asarray(chunk, dtype=np.float32) - mean) / np.maximum(scales, _TF_AXIS_EPS),
        dtype=np.float32,
    )
    if np.any(zero_mask):
        working[:, zero_mask] = 0.0
    norms = np.linalg.norm(working, axis=1, keepdims=True)
    normalized = np.asarray(working / np.maximum(norms, _TF_AXIS_EPS), dtype=np.float32)
    zero_rows = np.asarray(norms[:, 0] <= _TF_AXIS_EPS, dtype=bool)
    if np.any(zero_rows):
        normalized[zero_rows] = 0.0
    return normalized


def _tf_axis_normalized_rows(
    matrix: np.ndarray,
    indices: list[int],
    *,
    stats: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    if not indices:
        return np.empty((0, matrix.shape[1]), dtype=np.float32)
    mean, scales, zero_mask = stats
    return _tf_axis_normalize_chunk(
        np.asarray(matrix[np.asarray(indices, dtype=np.int64)], dtype=np.float32),
        mean=mean,
        scales=scales,
        zero_mask=zero_mask,
    )


def _tf_axis_centroids_from_groups(
    matrix: np.ndarray,
    cohort_values: list[object],
    *,
    groups: dict[str, set[str]],
    stats: tuple[np.ndarray, np.ndarray, np.ndarray],
    cohort_column: str,
) -> dict[str, np.ndarray]:
    if len(cohort_values) != matrix.shape[0]:
        raise ContractViolationError("tf_axis_orientation_audit centroid row count does not match matrix row count")
    mean, scales, zero_mask = stats
    dims = matrix.shape[1]
    sums = {group: np.zeros(dims, dtype=np.float64) for group in groups}
    counts = {group: 0 for group in groups}
    labels = [str(value or "").strip() for value in cohort_values]
    for start in range(0, matrix.shape[0], _TF_AXIS_MATRIX_CHUNK_ROWS):
        end = min(start + _TF_AXIS_MATRIX_CHUNK_ROWS, matrix.shape[0])
        normalized = _tf_axis_normalize_chunk(
            np.asarray(matrix[start:end], dtype=np.float32),
            mean=mean,
            scales=scales,
            zero_mask=zero_mask,
        )
        chunk_labels = labels[start:end]
        for group, values in groups.items():
            mask = np.asarray([label in values for label in chunk_labels], dtype=bool)
            if not np.any(mask):
                continue
            sums[group] += np.sum(normalized[mask], axis=0, dtype=np.float64)
            counts[group] += int(np.count_nonzero(mask))
    centroids: dict[str, np.ndarray] = {}
    for group, count in counts.items():
        if count == 0:
            raise ContractViolationError(
                f"tf_axis_orientation_audit centroid group {group!r} matched no rows on {cohort_column!r}"
            )
        centroid = try_l2_normalize_vector(np.asarray(sums[group] / float(count), dtype=np.float32))
        if centroid is None:
            raise ContractViolationError(f"tf_axis_orientation_audit centroid group {group!r} is degenerate")
        centroids[group] = centroid
    return centroids


def _required_tf_columns(tf_columns: dict[str, object]) -> list[str]:
    return [str(column) for values in tf_columns.values() for column in list(values or [])]


def _tf_axis_audit_output_columns(
    *,
    row_column_names: set[str],
    tf_columns: dict[str, object],
    output_columns: object,
) -> list[str]:
    derived_columns = set(_TF_AXIS_DERIVED_COLUMNS)
    required_tf_columns = _required_tf_columns(tf_columns)
    if output_columns is not None:
        if not isinstance(output_columns, (list, tuple)):
            raise ContractViolationError("tf_axis_orientation_audit output_columns must be a list")
        columns = _dedupe_columns([str(column) for column in output_columns if str(column).strip()])
        if not columns:
            raise ContractViolationError("tf_axis_orientation_audit output_columns cannot be empty")
        missing = sorted(
            column
            for column in columns
            if column not in row_column_names and column not in derived_columns and column not in required_tf_columns
        )
        if missing:
            raise ContractViolationError(f"tf_axis_orientation_audit output_columns are missing: {missing}")
        return columns
    base_columns = [column for column in _TF_AXIS_DEFAULT_BASE_COLUMNS if column in row_column_names]
    return _dedupe_columns([*base_columns, *required_tf_columns, *_TF_AXIS_DERIVED_COLUMNS])


def _resolve_workspace_table_path(
    context: WorkspaceContext,
    raw_path: object,
    *,
    param_name: str,
    contract_name: str = "tf_axis_orientation_audit association_overlay",
) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ContractViolationError(f"{contract_name} requires {param_name}")
    path = Path(text)
    if not path.is_absolute():
        path = context.workspace_dir / path
    if not path.is_file():
        raise MissingArtifactError(f"{contract_name} is missing: {path}")
    return path


def _required_mapping_string(mapping: dict[str, object], key: str, *, contract_name: str) -> str:
    if key not in mapping:
        raise ContractViolationError(f"{contract_name} requires {key}")
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ContractViolationError(f"{contract_name} requires non-empty {key}")
    return value


def _validate_tf_columns_present(column_names: set[str], tf_columns: dict[str, object]) -> None:
    missing = sorted(column for column in _required_tf_columns(tf_columns) if column not in column_names)
    if missing:
        raise ContractViolationError(f"tf_axis_orientation_audit missing required tf columns: {missing}")


def _filter_table_indices(
    table: pa.Table,
    where: object,
    *,
    contract_name: str,
) -> list[int]:
    if where is None:
        return list(range(table.num_rows))
    if not isinstance(where, dict):
        raise ContractViolationError(f"{contract_name} where must be a mapping")
    column = str(where.get("column") or "")
    if not column:
        raise ContractViolationError(f"{contract_name} where requires column")
    if column not in set(table.column_names):
        raise ContractViolationError(f"{contract_name} where column is missing: {column}")
    values = table[column].to_pylist()
    if "equals" in where:
        if "in" in where:
            raise ContractViolationError(f"{contract_name} where supports exactly one of equals or in")
        expected = where["equals"]
        return [index for index, value in enumerate(values) if value == expected]
    if "in" in where:
        expected_values = set(where["in"] or [])
        return [index for index, value in enumerate(values) if value in expected_values]
    raise ContractViolationError(f"{contract_name} where supports equals or in")


def _apply_tf_association_overlay(
    context: WorkspaceContext,
    rows: list[dict[str, object]],
    *,
    rows_column_names: set[str],
    tf_columns: dict[str, object],
    association_overlay: object,
) -> tuple[list[dict[str, object]], list[ScalarInputRef], dict[str, object]]:
    if association_overlay is None:
        _validate_tf_columns_present(rows_column_names, tf_columns)
        return rows, [], {}
    if not isinstance(association_overlay, dict):
        raise ContractViolationError("tf_axis_orientation_audit association_overlay must be a mapping")
    overlay_path = _resolve_workspace_table_path(
        context,
        association_overlay.get("path"),
        param_name="path",
    )
    overlay_table = read_table(overlay_path)
    legacy_keys = sorted(set(association_overlay).intersection({"join_key"}))
    if legacy_keys:
        raise ContractViolationError(f"tf_axis_orientation_audit association_overlay has legacy keys: {legacy_keys}")
    row_key = _required_mapping_string(
        association_overlay,
        "row_key",
        contract_name="tf_axis_orientation_audit association_overlay",
    )
    relation_key = _required_mapping_string(
        association_overlay,
        "relation_key",
        contract_name="tf_axis_orientation_audit association_overlay",
    )
    regulator_column = _required_mapping_string(
        association_overlay,
        "regulator_column",
        contract_name="tf_axis_orientation_audit association_overlay",
    )
    if row_key not in rows_column_names:
        raise ContractViolationError(f"tf_axis_orientation_audit association_overlay row_key missing: {row_key}")
    missing_relation_columns = [
        column for column in (relation_key, regulator_column) if column not in set(overlay_table.column_names)
    ]
    if missing_relation_columns:
        raise ContractViolationError(
            f"tf_axis_orientation_audit association_overlay missing relation columns: {missing_relation_columns}"
        )
    tf_aliases_payload = association_overlay.get("tf_aliases")
    if not isinstance(tf_aliases_payload, dict):
        raise ContractViolationError("tf_axis_orientation_audit association_overlay requires tf_aliases")
    aliases_by_column = {
        str(column): {str(alias).casefold() for alias in list(aliases or []) if str(alias).strip()}
        for column, aliases in tf_aliases_payload.items()
    }
    missing_aliases = sorted(column for column in _required_tf_columns(tf_columns) if column not in aliases_by_column)
    if missing_aliases:
        raise ContractViolationError(
            f"tf_axis_orientation_audit association_overlay missing tf_aliases for columns: {missing_aliases}"
        )

    regulators_by_key: dict[str, set[str]] = defaultdict(set)
    for relation in overlay_table.to_pylist():
        key = str(relation.get(relation_key) or "").strip()
        regulator = str(relation.get(regulator_column) or "").strip().casefold()
        if key and regulator:
            regulators_by_key[key].add(regulator)

    output_rows: list[dict[str, object]] = []
    for row in rows:
        output = dict(row)
        observed = regulators_by_key.get(str(row.get(row_key) or "").strip(), set())
        for column, aliases in aliases_by_column.items():
            output[column] = bool(observed & aliases)
        output_rows.append(output)
    return (
        output_rows,
        [ScalarInputRef(kind="association_overlay", artifact_id=overlay_path.stem, path=overlay_path)],
        {
            "association_overlay_path": str(overlay_path),
            "association_overlay_rows": overlay_table.num_rows,
            "association_overlay_row_key": row_key,
            "association_overlay_relation_key": relation_key,
        },
    )


def _tf_axis_orientation_audit_table(
    context: WorkspaceContext,
    *,
    view_id: str,
    audit_view_id: str | None,
    cohort_column: str,
    centroid_groups: dict[str, object],
    tf_columns: dict[str, object],
    embedding_view: str | None,
    association_overlay: object,
    output_filter: object,
    output_columns: object,
    expected_output_rows: int | None = None,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    centroid_matrix_path, centroid_rows_path = _view_paths(context, view_id)
    centroid_matrix = np.asarray(read_matrix(centroid_matrix_path), dtype=np.float32)
    centroid_rows_table = read_table(centroid_rows_path)
    if cohort_column not in set(centroid_rows_table.column_names):
        raise ContractViolationError(f"tf_axis_orientation_audit cohort column is missing: {cohort_column}")
    centroid_cohort_values = centroid_rows_table[cohort_column].to_pylist()
    resolved_audit_view_id = audit_view_id or view_id
    if resolved_audit_view_id == view_id:
        audit_matrix_path = centroid_matrix_path
        audit_rows_path = centroid_rows_path
        audit_matrix = centroid_matrix
        audit_rows_table = centroid_rows_table
    else:
        audit_matrix_path, audit_rows_path = _view_paths(context, resolved_audit_view_id)
        audit_matrix = np.asarray(read_matrix(audit_matrix_path), dtype=np.float32)
        audit_rows_table = read_table(audit_rows_path)
        if centroid_matrix.ndim != 2 or audit_matrix.ndim != 2 or centroid_matrix.shape[1] != audit_matrix.shape[1]:
            raise ContractViolationError(
                "tf_axis_orientation_audit centroid and audit views must have the same vector dimension"
            )
    required_groups = ("background", "ethanol", "cipro")
    groups = {group: {str(value) for value in centroid_groups.get(group, [])} for group in required_groups}
    missing = [group for group, values in groups.items() if not values]
    if missing:
        raise ContractViolationError(f"tf_axis_orientation_audit missing centroid groups: {missing}")
    centroid_stats = _tf_axis_standardization_stats(centroid_matrix)
    audit_stats = centroid_stats if resolved_audit_view_id == view_id else _tf_axis_standardization_stats(audit_matrix)
    centroids = _tf_axis_centroids_from_groups(
        centroid_matrix,
        centroid_cohort_values,
        groups=groups,
        stats=centroid_stats,
        cohort_column=cohort_column,
    )
    ethanol_columns = [str(value) for value in tf_columns.get("ethanol", [])]
    cipro_columns = [str(value) for value in tf_columns.get("cipro", [])]
    if not ethanol_columns or not cipro_columns:
        raise ContractViolationError("tf_axis_orientation_audit requires ethanol and cipro tf_columns")

    if output_filter is None:
        raise ContractViolationError(
            "tf_axis_orientation_audit requires output_filter so centroid rows and emitted audit rows are explicit"
        )
    audit_column_names = set(audit_rows_table.column_names)
    if association_overlay is None:
        _validate_tf_columns_present(audit_column_names, tf_columns)
    projected_columns = _tf_axis_audit_output_columns(
        row_column_names=audit_column_names,
        tf_columns=tf_columns,
        output_columns=output_columns,
    )
    audit_indices = _filter_table_indices(
        audit_rows_table,
        output_filter,
        contract_name="tf_axis_orientation_audit output_filter",
    )
    if not audit_indices:
        raise ContractViolationError("tf_axis_orientation_audit output_filter matched no rows")
    row_columns_to_select = {
        column
        for column in projected_columns
        if column in audit_column_names and column not in set(_TF_AXIS_DERIVED_COLUMNS)
    }
    if isinstance(output_filter, dict):
        row_columns_to_select.add(str(output_filter.get("column") or ""))
    if association_overlay is not None and isinstance(association_overlay, dict):
        row_columns_to_select.add(str(association_overlay.get("row_key") or "id"))
    if association_overlay is None:
        row_columns_to_select.update(
            column for column in _required_tf_columns(tf_columns) if column in audit_column_names
        )
    row_columns_to_select = {column for column in row_columns_to_select if column in audit_column_names}
    filtered_audit_table = audit_rows_table.take(pa.array(audit_indices, type=pa.int64()))
    audit_row_dicts = filtered_audit_table.select(sorted(row_columns_to_select)).to_pylist()
    audit_row_dicts, association_inputs, association_stats = _apply_tf_association_overlay(
        context,
        audit_row_dicts,
        rows_column_names=set(filtered_audit_table.column_names),
        tf_columns=tf_columns,
        association_overlay=association_overlay,
    )

    filtered_normalized_audit_rows = _tf_axis_normalized_rows(audit_matrix, audit_indices, stats=audit_stats)
    similarities = {
        group: np.asarray(filtered_normalized_audit_rows @ centroid, dtype=np.float32)
        for group, centroid in centroids.items()
    }
    metric_rows: list[dict[str, object]] = []
    for index, row in enumerate(audit_row_dicts):
        output = dict(row)
        tf_bin = _tf_bin(output, ethanol_columns=ethanol_columns, cipro_columns=cipro_columns)
        output["tf_bin"] = tf_bin
        output["embedding_view"] = embedding_view or resolved_audit_view_id
        output["ethanolness"] = float(similarities["ethanol"][index] - similarities["background"][index])
        output["ciproness"] = float(similarities["cipro"][index] - similarities["background"][index])
        output["distance_to_ethanol_centroid"] = float(1.0 - similarities["ethanol"][index])
        output["distance_to_cipro_centroid"] = float(1.0 - similarities["cipro"][index])
        output["distance_to_background_centroid"] = float(1.0 - similarities["background"][index])
        metric_rows.append(output)
    output_rows = [{column: row.get(column) for column in projected_columns} for row in metric_rows]
    table = pa.Table.from_pylist(output_rows)
    if expected_output_rows is not None and table.num_rows != expected_output_rows:
        raise ContractViolationError(
            "tf_axis_orientation_audit expected_output_rows mismatch: "
            f"expected {expected_output_rows}, observed {table.num_rows}"
        )
    input_refs = [
        ScalarInputRef(kind="view_matrix", artifact_id=view_id, path=centroid_matrix_path),
        ScalarInputRef(kind="view_rows", artifact_id=view_id, path=centroid_rows_path),
    ]
    if resolved_audit_view_id != view_id:
        input_refs.extend(
            [
                ScalarInputRef(kind="view_matrix", artifact_id=resolved_audit_view_id, path=audit_matrix_path),
                ScalarInputRef(kind="view_rows", artifact_id=resolved_audit_view_id, path=audit_rows_path),
            ]
        )
    return (
        table,
        [*input_refs, *association_inputs],
        {
            "view_id": view_id,
            "centroid_view_id": view_id,
            "audit_view_id": resolved_audit_view_id,
            "centroid_rows": centroid_rows_table.num_rows,
            "input_rows": audit_rows_table.num_rows,
            "filtered_rows": len(metric_rows),
            "rows": table.num_rows,
            "expected_output_rows": expected_output_rows,
            "embedding_view": embedding_view or resolved_audit_view_id,
            "tf_bin_counts": dict(Counter(row["tf_bin"] for row in output_rows)),
            "output_columns": projected_columns,
            **association_stats,
        },
    )


def _filter_rows(
    rows: list[dict[str, object]],
    where: object,
    *,
    column_names: set[str],
    contract_name: str = "tf_axis_orientation_tests",
) -> list[dict[str, object]]:
    if where is None:
        return rows
    if not isinstance(where, dict):
        raise ContractViolationError(f"{contract_name} where must be a mapping")
    column = str(where.get("column") or "")
    if not column:
        raise ContractViolationError(f"{contract_name} where requires column")
    if column not in column_names:
        raise ContractViolationError(f"{contract_name} where column is missing: {column}")
    if "equals" in where:
        expected = where["equals"]
        return [row for row in rows if row.get(column) == expected]
    if "in" in where:
        expected_values = set(where["in"] or [])
        return [row for row in rows if row.get(column) in expected_values]
    raise ContractViolationError(f"{contract_name} where supports equals or in")


def _mann_whitney_u(target: np.ndarray, background: np.ndarray) -> float:
    combined = np.concatenate([target, background])
    ranks = rankdata_average(combined)
    n_target = target.size
    rank_sum = float(np.sum(ranks[:n_target], dtype=np.float64))
    return rank_sum - (n_target * (n_target + 1) / 2.0)


def _tie_group_count(values: np.ndarray) -> int:
    counts = Counter(float(value) for value in values)
    return sum(1 for count in counts.values() if count > 1)


def _mann_whitney_greater_result(target: np.ndarray, background: np.ndarray, observed_u: float) -> dict[str, object]:
    n_target = target.size
    n_background = background.size
    combined = np.concatenate([target, background])
    tie_group_count = _tie_group_count(combined)
    total_combinations = math.comb(n_target + n_background, n_target)
    if total_combinations <= 100_000:
        exceedances = 0
        for target_indices in combinations(range(combined.size), n_target):
            mask = np.zeros(combined.size, dtype=bool)
            mask[list(target_indices)] = True
            perm_u = _mann_whitney_u(combined[mask], combined[~mask])
            if perm_u >= observed_u - 1e-12:
                exceedances += 1
        return {
            "p_value": exceedances / float(total_combinations),
            "p_value_method": "exact_enumeration",
            "tie_group_count": tie_group_count,
        }
    mean = n_target * n_background / 2.0
    total = n_target + n_background
    tie_term = 0.0
    if tie_group_count:
        counts = Counter(float(value) for value in combined)
        tie_term = sum((count**3) - count for count in counts.values() if count > 1)
    variance = n_target * n_background / 12.0
    if total > 1:
        variance *= (total + 1) - (tie_term / (total * (total - 1)))
    sd = math.sqrt(max(variance, 0.0))
    if sd == 0.0:
        p_value = float("nan")
    else:
        z = (observed_u - mean - 0.5) / sd
        p_value = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {
        "p_value": p_value,
        "p_value_method": "normal_approximation_tie_corrected",
        "tie_group_count": tie_group_count,
    }


def _finite_values(rows: list[dict[str, object]], *, axis: str, tf_bin: str) -> np.ndarray:
    values = []
    for row in rows:
        if row.get("tf_bin") != tf_bin:
            continue
        try:
            value = float(row[axis])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _tf_axis_orientation_tests_table(
    context: WorkspaceContext,
    *,
    source_scalar: str,
    tests: list[dict[str, object]],
    where: object,
) -> tuple[pa.Table, list[ScalarInputRef], dict[str, object]]:
    scalar_path = _scalar_table_path(context, source_scalar)
    scalar_table = read_table(scalar_path)
    if where is None:
        raise ContractViolationError("tf_axis_orientation_tests requires where to declare the tested row population")
    rows = _filter_rows(scalar_table.to_pylist(), where, column_names=set(scalar_table.column_names))
    output_rows: list[dict[str, object]] = []
    for spec in tests:
        axis = str(spec.get("axis") or "")
        target_bin = str(spec.get("target_bin") or "")
        background_bin = str(spec.get("background_bin") or "neither")
        if not axis or not target_bin:
            raise ContractViolationError("tf_axis_orientation_tests entries require axis and target_bin")
        target = _finite_values(rows, axis=axis, tf_bin=target_bin)
        background = _finite_values(rows, axis=axis, tf_bin=background_bin)
        if target.size == 0 or background.size == 0:
            output_rows.append(
                {
                    "axis": axis,
                    "target_bin": target_bin,
                    "background_bin": background_bin,
                    "n_target": int(target.size),
                    "n_background": int(background.size),
                    "target_median": float("nan"),
                    "background_median": float("nan"),
                    "median_difference": float("nan"),
                    "mann_whitney_u": float("nan"),
                    "rank_biserial": float("nan"),
                    "p_value": float("nan"),
                    "p_value_method": None,
                    "tie_group_count": 0,
                    "alternative": "greater",
                    "status": "insufficient_data",
                }
            )
            continue
        observed_u = _mann_whitney_u(target, background)
        pvalue_result = _mann_whitney_greater_result(target, background, observed_u)
        denominator = float(target.size * background.size)
        output_rows.append(
            {
                "axis": axis,
                "target_bin": target_bin,
                "background_bin": background_bin,
                "n_target": int(target.size),
                "n_background": int(background.size),
                "target_median": float(np.median(target)),
                "background_median": float(np.median(background)),
                "median_difference": float(np.median(target) - np.median(background)),
                "mann_whitney_u": float(observed_u),
                "rank_biserial": float((2.0 * observed_u / denominator) - 1.0),
                "p_value": float(pvalue_result["p_value"]),
                "p_value_method": str(pvalue_result["p_value_method"]),
                "tie_group_count": int(pvalue_result["tie_group_count"]),
                "alternative": "greater",
                "status": "ok",
            }
        )
    table = pa.Table.from_pylist(output_rows)
    return (
        table,
        [ScalarInputRef(kind="scalar_table", artifact_id=source_scalar, path=scalar_path)],
        {
            "source_scalar": source_scalar,
            "rows": table.num_rows,
            "tested_axes": [str(spec.get("axis") or "") for spec in tests],
        },
    )


def build_tf_axis_orientation_audit_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, object],
) -> BuiltScalarArtifact:
    audit_view_raw = _optional_param(params, "audit_view_id", default=None)
    table, inputs, stats = _tf_axis_orientation_audit_table(
        context,
        view_id=str(_require_param(params, "view_id")),
        audit_view_id=(str(audit_view_raw).strip() or None) if audit_view_raw is not None else None,
        cohort_column=str(_require_param(params, "cohort_column")),
        centroid_groups=dict(_require_param(params, "centroid_groups")),
        tf_columns=dict(_require_param(params, "tf_columns")),
        embedding_view=_optional_param(params, "embedding_view", default=None),
        association_overlay=_optional_param(params, "association_overlay", default=None),
        output_filter=_optional_param(params, "output_filter", default=None),
        output_columns=_optional_param(params, "output_columns", default=None),
        expected_output_rows=(
            int(_optional_param(params, "expected_output_rows", default=0))
            if "expected_output_rows" in params
            else None
        ),
    )
    write_table(table, artifact_dir / "table.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )


def build_tf_axis_orientation_tests_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, object],
) -> BuiltScalarArtifact:
    table, inputs, stats = _tf_axis_orientation_tests_table(
        context,
        source_scalar=str(_require_param(params, "source_scalar")),
        tests=[
            dict(value)
            for value in _optional_param(
                params,
                "tests",
                default=[
                    {"axis": "ethanolness", "target_bin": "ethanol_TF", "background_bin": "neither"},
                    {"axis": "ciproness", "target_bin": "lexA_TF", "background_bin": "neither"},
                ],
            )
        ],
        where=_optional_param(params, "where", default=None),
    )
    write_table(table, artifact_dir / "table.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[],
        stats=stats,
    )
