"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/scalars/builders/trait_axis_projection.py

Generic fitted trait-axis projection scalar builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...geometry.preprocessing import try_l2_normalize_vector
from ...io.json_io import read_json, write_json
from ...io.matrix_io import read_matrix
from ...io.parquet_io import read_table, write_table
from ...stats.rank import kendall_tau_b, pearson_correlation, spearman_correlation
from ...workspaces.loader import WorkspaceContext
from ..common import BuiltScalarArtifact, ScalarInputRef, _optional_param, _require_param

_ROWS_KIND = "trait_axis_projection_rows"
_SUMMARY_KIND = "trait_axis_projection_summary"
_NORMALIZATION_POLICY = "row_l2"
_ALLOWED_ROLES = frozenset({"fit", "eval", "reference", "sensitivity", "excluded"})
_EPS = 1e-8
_COMMON_IDENTITY_COLUMNS = (
    "subject_id",
    "subject_key",
    "construct_subject__id",
    "candidate_source",
    "source_family",
)
_REQUIRED_ROW_COLUMNS = frozenset(
    {
        "candidate_id",
        "view_id",
        "trait_id",
        "axis_id",
        "endpoint_definition_id",
        "population_id",
        "population_role",
        "endpoint_group",
        "source_value",
        "source_value_available",
        "axis_projection",
        "endpoint_margin",
        "row_status",
        "row_status_reason",
    }
)


def _view_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path]:
    context.require_view(view_id)
    artifact_dir = context.output_root / "views" / view_id
    matrix_path = artifact_dir / "matrix.npy"
    rows_path = artifact_dir / "rows.parquet"
    if not matrix_path.is_file() or not rows_path.is_file():
        raise MissingArtifactError(f"{_ROWS_KIND} view artifact is missing: {view_id}")
    return matrix_path, rows_path


def _as_list(value: object, *, contract_name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ContractViolationError(f"{contract_name} must be a non-empty list")
    return list(value)


def _as_mapping(value: object, *, contract_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractViolationError(f"{contract_name} must be a mapping")
    return dict(value)


def _required_text(mapping: dict[str, object], key: str, *, contract_name: str) -> str:
    value = str(mapping.get(key) or "").strip()
    if not value:
        raise ContractViolationError(f"{contract_name} requires {key}")
    return value


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _selector_items(where: object, *, contract_name: str) -> list[dict[str, object]]:
    if where is None:
        return []
    if isinstance(where, dict):
        return [dict(where)]
    if not isinstance(where, list):
        raise ContractViolationError(f"{contract_name} where must be a mapping or list of mappings")
    selectors: list[dict[str, object]] = []
    for index, selector in enumerate(where):
        if not isinstance(selector, dict):
            raise ContractViolationError(f"{contract_name} where[{index}] must be a mapping")
        selectors.append(dict(selector))
    return selectors


def _selector_operation(selector: dict[str, object], *, contract_name: str) -> str:
    if not str(selector.get("column") or "").strip():
        raise ContractViolationError(f"{contract_name} selector requires column")
    operations = [
        name
        for name in ("equals", "in_values", "in", "regex", "not_regex", "finite", "not_null", "non_null")
        if name in selector
    ]
    if len(operations) != 1:
        raise ContractViolationError(f"{contract_name} selector must declare exactly one predicate: {selector}")
    operation = operations[0]
    if operation in {"in_values", "in"}:
        values = selector.get(operation)
        if isinstance(values, str) or not isinstance(values, (list, tuple, set)) or not values:
            raise ContractViolationError(
                f"{contract_name} selector predicate {operation!r} requires a non-empty sequence"
            )
    elif operation in {"regex", "not_regex"}:
        pattern = selector.get(operation)
        if not isinstance(pattern, str) or not pattern:
            raise ContractViolationError(
                f"{contract_name} selector predicate {operation!r} requires a non-empty regex string"
            )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ContractViolationError(
                f"{contract_name} selector predicate {operation!r} has invalid regex: {pattern!r}"
            ) from exc
    elif operation in {"finite", "not_null", "non_null"} and not isinstance(selector.get(operation), bool):
        raise ContractViolationError(f"{contract_name} selector predicate {operation!r} requires a boolean value")
    return operation


def _validate_selector_columns(
    selectors: Iterable[dict[str, object]],
    *,
    column_names: set[str],
    contract_name: str,
) -> None:
    for selector in selectors:
        operation = _selector_operation(selector, contract_name=contract_name)
        column = str(selector["column"])
        if column not in column_names:
            raise ContractViolationError(
                f"{contract_name} selector column is missing: column={column!r}, predicate={operation!r}"
            )


def _selector_matches(row: dict[str, object], selector: dict[str, object]) -> bool:
    column = str(selector["column"])
    value = row.get(column)
    if "equals" in selector:
        return value == selector["equals"]
    if "in_values" in selector or "in" in selector:
        expected_values = set(selector.get("in_values", selector.get("in")) or [])
        return value in expected_values
    if "regex" in selector:
        return re.search(str(selector["regex"]), "" if _is_missing(value) else str(value)) is not None
    if "not_regex" in selector:
        return re.search(str(selector["not_regex"]), "" if _is_missing(value) else str(value)) is None
    if "finite" in selector:
        return (_finite_float(value) is not None) is bool(selector["finite"])
    if "not_null" in selector or "non_null" in selector:
        expected = bool(selector.get("not_null", selector.get("non_null")))
        return (not _is_missing(value)) is expected
    raise ContractViolationError(f"{_ROWS_KIND} selector reached unsupported predicate: {selector}")


def _mask_rows(
    rows: list[dict[str, object]],
    where: object,
    *,
    column_names: set[str],
    contract_name: str,
) -> np.ndarray:
    selectors = _selector_items(where, contract_name=contract_name)
    _validate_selector_columns(selectors, column_names=column_names, contract_name=contract_name)
    if not selectors:
        return np.ones(len(rows), dtype=bool)
    return np.asarray([all(_selector_matches(row, selector) for selector in selectors) for row in rows], dtype=bool)


def _selector_columns(where: object, *, contract_name: str) -> set[str]:
    columns: set[str] = set()
    for selector in _selector_items(where, contract_name=contract_name):
        _selector_operation(selector, contract_name=contract_name)
        columns.add(str(selector["column"]))
    return columns


def _unique_nonempty_strings(values: Iterable[object], *, contract_name: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            raise ContractViolationError(f"{contract_name} contains an empty identifier")
        if text in seen:
            raise ContractViolationError(f"{contract_name} contains duplicate identifier: {text!r}")
        seen.add(text)
        output.append(text)
    return output


def _axis_id(axis: dict[str, object]) -> str:
    return str(axis.get("axis_id") or axis.get("trait_id") or "").strip()


def _endpoint_quantile_from_id(endpoint_definition_id: str) -> float | None:
    match = re.fullmatch(r"quantile_([0-9]+(?:[_.][0-9]+)?)", endpoint_definition_id)
    if match is None:
        return None
    value = float(match.group(1).replace("_", ".", 1))
    if not 0.0 < value < 0.5 + 1e-12:
        raise ContractViolationError(f"{_ROWS_KIND} quantile endpoint must be in (0, 0.5]: {endpoint_definition_id}")
    return value


def _base_endpoint_definition(axis: dict[str, object]) -> dict[str, object]:
    groups = _as_mapping(axis.get("endpoint_groups"), contract_name=f"{_ROWS_KIND} endpoint_groups")
    method = _required_text(groups, "method", contract_name=f"{_ROWS_KIND} endpoint_groups")
    endpoint_id = str(
        axis.get("primary_endpoint_definition_id")
        or groups.get("endpoint_definition_id")
        or ("min_max" if method == "min_max" else method)
    ).strip()
    if not endpoint_id:
        raise ContractViolationError(f"{_ROWS_KIND} endpoint definition requires endpoint_definition_id")
    definition = dict(groups)
    definition["endpoint_definition_id"] = endpoint_id
    return definition


def _endpoint_from_named_definition(endpoint_definition_id: str, *, base: dict[str, object]) -> dict[str, object]:
    if endpoint_definition_id == "min_max":
        return {
            "endpoint_definition_id": endpoint_definition_id,
            "method": "min_max",
            "value_column": base.get("value_column"),
            "min_low_rows": 1,
            "min_high_rows": 1,
        }
    quantile = _endpoint_quantile_from_id(endpoint_definition_id)
    if quantile is not None:
        return {
            "endpoint_definition_id": endpoint_definition_id,
            "method": "quantile",
            "value_column": base.get("value_column"),
            "low_quantile": quantile,
            "high_quantile": 1.0 - quantile,
            "min_low_rows": int(base.get("min_low_rows", 1) or 1),
            "min_high_rows": int(base.get("min_high_rows", 1) or 1),
        }
    if endpoint_definition_id == str(base.get("endpoint_definition_id")):
        return dict(base)
    raise ContractViolationError(f"{_ROWS_KIND} unknown endpoint definition: {endpoint_definition_id!r}")


def _endpoint_definitions(axis: dict[str, object]) -> list[dict[str, object]]:
    base = _base_endpoint_definition(axis)
    sensitivity = axis.get("endpoint_sensitivity") or {}
    if not isinstance(sensitivity, dict) or not bool(sensitivity.get("enabled", False)):
        return [base]
    raw_definitions = sensitivity.get("endpoint_definitions") or []
    if not isinstance(raw_definitions, list) or not raw_definitions:
        raise ContractViolationError(f"{_ROWS_KIND} endpoint_sensitivity.endpoint_definitions must be non-empty")
    definitions: list[dict[str, object]] = []
    for raw in raw_definitions:
        if isinstance(raw, dict):
            definitions.append(dict(raw))
        else:
            definitions.append(_endpoint_from_named_definition(str(raw), base=base))
    primary_id = str(base["endpoint_definition_id"])
    if primary_id not in {str(definition.get("endpoint_definition_id") or "") for definition in definitions}:
        definitions.insert(0, base)
    _unique_nonempty_strings(
        [definition.get("endpoint_definition_id") for definition in definitions],
        contract_name=f"{_ROWS_KIND} endpoint_definition_id",
    )
    return definitions


def _required_columns_for_axis(axis: dict[str, object], *, candidate_id_column: str) -> set[str]:
    axis_id = _axis_id(axis)
    contract_name = f"{_ROWS_KIND} axis={axis_id!r}"
    columns = {candidate_id_column}
    source_value_column = axis.get("source_value_column")
    if source_value_column is not None:
        columns.add(str(source_value_column))
    fit_population = _as_mapping(axis.get("fit_population"), contract_name=f"{contract_name} fit_population")
    columns.update(_selector_columns(fit_population.get("where"), contract_name=f"{contract_name} fit_population"))
    for population in _as_list(axis.get("score_populations"), contract_name=f"{contract_name} score_populations"):
        population_mapping = _as_mapping(population, contract_name=f"{contract_name} score_population")
        population_id = _required_text(
            population_mapping,
            "population_id",
            contract_name=f"{contract_name} score_population",
        )
        columns.update(
            _selector_columns(
                population_mapping.get("where"),
                contract_name=f"{contract_name} population={population_id!r}",
            )
        )
    for definition in _endpoint_definitions(axis):
        method = str(definition.get("method") or "")
        if method in {"min_max", "quantile"}:
            columns.add(_required_text(definition, "value_column", contract_name=f"{contract_name} endpoint"))
        elif method == "explicit_values":
            columns.add(_required_text(definition, "group_column", contract_name=f"{contract_name} endpoint"))
        elif method == "configured_selector":
            columns.update(
                _selector_columns(
                    definition.get("low_where"),
                    contract_name=f"{contract_name} low endpoint",
                )
            )
            columns.update(
                _selector_columns(definition.get("high_where"), contract_name=f"{contract_name} high endpoint")
            )
        else:
            raise ContractViolationError(f"{contract_name} unsupported endpoint method: {method!r}")
    parent_key = axis.get("parent_key")
    parent_candidate_id_column = axis.get("parent_candidate_id_column")
    if bool(parent_key) != bool(parent_candidate_id_column):
        raise ContractViolationError(
            f"{contract_name} parent_key and parent_candidate_id_column must be configured together"
        )
    if parent_key and parent_candidate_id_column:
        columns.update({str(parent_key), str(parent_candidate_id_column)})
    return columns


def _validate_axes(raw_axes: object) -> list[dict[str, object]]:
    axes = [_as_mapping(axis, contract_name=f"{_ROWS_KIND} axis") for axis in _as_list(raw_axes, contract_name="axes")]
    _unique_nonempty_strings([axis.get("trait_id") for axis in axes], contract_name=f"{_ROWS_KIND} trait_id")
    _unique_nonempty_strings([_axis_id(axis) for axis in axes], contract_name=f"{_ROWS_KIND} axis_id")
    for axis in axes:
        axis_id = _axis_id(axis)
        contract_name = f"{_ROWS_KIND} axis={axis_id!r}"
        fit_population = _as_mapping(axis.get("fit_population"), contract_name=f"{contract_name} fit_population")
        if str(fit_population.get("role") or "") != "fit":
            raise ContractViolationError(f"{contract_name} fit_population role must be exactly 'fit'")
        score_populations = [
            _as_mapping(population, contract_name=f"{contract_name} score_population")
            for population in _as_list(
                axis.get("score_populations"),
                contract_name=f"{contract_name} score_populations",
            )
        ]
        population_ids = [
            fit_population.get("population_id"),
            *[population.get("population_id") for population in score_populations],
        ]
        _unique_nonempty_strings(
            population_ids,
            contract_name=f"{contract_name} population_id",
        )
        for population in score_populations:
            role = str(population.get("role") or "")
            if role not in _ALLOWED_ROLES:
                raise ContractViolationError(
                    f"{contract_name} population={population.get('population_id')!r} has unsupported role: {role!r}"
                )
        _endpoint_definitions(axis)
    return axes


def _row_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ContractViolationError(f"{_ROWS_KIND} expects each view matrix to be 2D")
    if array.shape[0] == 0:
        raise ContractViolationError(f"{_ROWS_KIND} cannot score an empty view matrix")
    if not np.isfinite(array).all():
        raise ContractViolationError(f"{_ROWS_KIND} row_l2 normalization encountered non-finite matrix values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(norms[:, 0] <= _EPS)
    if zero_rows.size:
        preview = [int(index) for index in zero_rows[:5]]
        raise ContractViolationError(f"{_ROWS_KIND} row_l2 normalization encountered zero-norm rows: {preview}")
    return np.ascontiguousarray(array / np.maximum(norms, _EPS), dtype=np.float32)


def _candidate_ids(rows: list[dict[str, object]], *, candidate_id_column: str) -> list[str]:
    ids: list[str] = []
    for index, row in enumerate(rows):
        value = row.get(candidate_id_column)
        if _is_missing(value) or not str(value).strip():
            raise ContractViolationError(f"{_ROWS_KIND} missing candidate_id at row index {index}")
        ids.append(str(value))
    return ids


def _reject_duplicate_fit_ids(candidate_ids: list[str], fit_mask: np.ndarray, *, axis_id: str, view_id: str) -> None:
    counts = Counter(candidate_ids[index] for index in np.flatnonzero(fit_mask))
    duplicates = sorted(candidate_id for candidate_id, count in counts.items() if count > 1)
    if duplicates:
        raise ContractViolationError(
            f"{_ROWS_KIND} axis={axis_id!r} view={view_id!r} has duplicate fit candidate_id values: {duplicates[:5]}"
        )


def _endpoint_masks(
    rows: list[dict[str, object]],
    *,
    fit_mask: np.ndarray,
    definition: dict[str, object],
    column_names: set[str],
    contract_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    method = str(definition.get("method") or "")
    endpoint_id = str(definition.get("endpoint_definition_id") or "")
    if method in {"min_max", "quantile"}:
        value_column = _required_text(definition, "value_column", contract_name=f"{contract_name} endpoint")
        values = np.asarray([_finite_float(row.get(value_column)) for row in rows], dtype=np.float64)
        finite_fit = fit_mask & np.isfinite(values)
        if not np.any(finite_fit):
            raise ContractViolationError(f"{contract_name} endpoint={endpoint_id!r} has no finite endpoint values")
        fit_values = values[finite_fit]
        if method == "min_max":
            low_threshold = float(np.min(fit_values))
            high_threshold = float(np.max(fit_values))
        else:
            low_quantile = float(definition.get("low_quantile", 0.0))
            high_quantile = float(definition.get("high_quantile", 1.0))
            if not 0.0 <= low_quantile <= high_quantile <= 1.0:
                raise ContractViolationError(f"{contract_name} endpoint={endpoint_id!r} has invalid quantiles")
            low_threshold = float(np.quantile(fit_values, low_quantile))
            high_threshold = float(np.quantile(fit_values, high_quantile))
        low_mask = finite_fit & (values <= low_threshold)
        high_mask = finite_fit & (values >= high_threshold)
    elif method == "explicit_values":
        group_column = _required_text(definition, "group_column", contract_name=f"{contract_name} endpoint")
        low_values = set(definition.get("low_values") or [])
        high_values = set(definition.get("high_values") or [])
        if not low_values or not high_values:
            raise ContractViolationError(
                f"{contract_name} endpoint={endpoint_id!r} requires low_values and high_values"
            )
        low_mask = np.asarray([row.get(group_column) in low_values for row in rows], dtype=bool) & fit_mask
        high_mask = np.asarray([row.get(group_column) in high_values for row in rows], dtype=bool) & fit_mask
    elif method == "configured_selector":
        low_mask = (
            _mask_rows(
                rows,
                definition.get("low_where"),
                column_names=column_names,
                contract_name=f"{contract_name} endpoint={endpoint_id!r} low",
            )
            & fit_mask
        )
        high_mask = (
            _mask_rows(
                rows,
                definition.get("high_where"),
                column_names=column_names,
                contract_name=f"{contract_name} endpoint={endpoint_id!r} high",
            )
            & fit_mask
        )
    else:
        raise ContractViolationError(f"{contract_name} endpoint={endpoint_id!r} uses unsupported method: {method!r}")

    overlap = low_mask & high_mask
    if np.any(overlap):
        raise ContractViolationError(f"{contract_name} endpoint={endpoint_id!r} low and high rows overlap")
    min_low = int(definition.get("min_low_rows", 1) or 1)
    min_high = int(definition.get("min_high_rows", 1) or 1)
    low_count = int(np.count_nonzero(low_mask))
    high_count = int(np.count_nonzero(high_mask))
    if low_count < min_low or high_count < min_high:
        raise ContractViolationError(
            f"{contract_name} endpoint={endpoint_id!r} has too few endpoint rows: "
            f"low={low_count}/{min_low}, high={high_count}/{min_high}"
        )
    return low_mask, high_mask


def _metadata_columns_for_output(
    raw_columns: object,
    *,
    candidate_id_column: str,
    column_names: set[str],
    contract_name: str,
) -> list[str]:
    if raw_columns is None:
        requested = []
    elif not isinstance(raw_columns, list):
        raise ContractViolationError(f"{contract_name} metadata_columns must be a list")
    else:
        requested = _unique_nonempty_strings(raw_columns, contract_name=f"{contract_name} metadata_columns")
    missing_requested = sorted(set(requested) - column_names)
    if missing_requested:
        raise ContractViolationError(f"{contract_name} metadata_columns are missing: {missing_requested}")
    columns = [
        candidate_id_column,
        *[column for column in _COMMON_IDENTITY_COLUMNS if column in column_names],
        *requested,
    ]
    output: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column in column_names and column not in seen:
            seen.add(column)
            output.append(column)
    return output


def _axis_projection_rows_for_view(
    *,
    context: WorkspaceContext,
    view_id: str,
    candidate_id_column: str,
    axes: list[dict[str, object]],
    metadata_columns: object,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[ScalarInputRef],
    dict[str, object],
]:
    matrix_path, rows_path = _view_paths(context, view_id)
    rows_table = read_table(rows_path)
    column_names = set(rows_table.column_names)
    rows = rows_table.to_pylist()
    for axis in axes:
        missing_columns = sorted(
            _required_columns_for_axis(axis, candidate_id_column=candidate_id_column) - column_names
        )
        if missing_columns:
            raise ContractViolationError(
                f"{_ROWS_KIND} view={view_id!r} axis={_axis_id(axis)!r} missing required columns: {missing_columns}"
            )
    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    if matrix.shape[0] != len(rows):
        raise ContractViolationError(
            f"{_ROWS_KIND} view={view_id!r} matrix row count does not match metadata rows: "
            f"{matrix.shape[0]} != {len(rows)}"
        )
    normalized = _row_l2_normalize(matrix)
    candidate_ids = _candidate_ids(rows, candidate_id_column=candidate_id_column)
    output_metadata_columns = _metadata_columns_for_output(
        metadata_columns,
        candidate_id_column=candidate_id_column,
        column_names=column_names,
        contract_name=f"{_ROWS_KIND} view={view_id!r}",
    )

    output_rows: list[dict[str, object]] = []
    axis_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    scored_role_counts: Counter[str] = Counter()

    for axis in axes:
        trait_id = _required_text(axis, "trait_id", contract_name=f"{_ROWS_KIND} axis")
        axis_id = _axis_id(axis)
        axis_label = str(axis.get("label") or trait_id)
        source_value_column = (
            str(axis.get("source_value_column")) if axis.get("source_value_column") is not None else None
        )
        primary_endpoint_definition_id = str(
            axis.get("primary_endpoint_definition_id") or _base_endpoint_definition(axis)["endpoint_definition_id"]
        )
        contract_name = f"{_ROWS_KIND} view={view_id!r} axis={axis_id!r}"
        fit_population = _as_mapping(axis.get("fit_population"), contract_name=f"{contract_name} fit_population")
        fit_population_id = _required_text(
            fit_population,
            "population_id",
            contract_name=f"{contract_name} fit_population",
        )
        fit_mask = _mask_rows(
            rows,
            fit_population.get("where"),
            column_names=column_names,
            contract_name=f"{contract_name} fit_population={fit_population_id!r}",
        )
        if not np.any(fit_mask):
            raise ContractViolationError(f"{contract_name} fit_population={fit_population_id!r} matched no rows")
        if not bool(axis.get("allow_duplicate_fit_candidate_ids", False)):
            _reject_duplicate_fit_ids(candidate_ids, fit_mask, axis_id=axis_id, view_id=view_id)

        score_populations = [
            _as_mapping(population, contract_name=f"{contract_name} score_population")
            for population in _as_list(
                axis.get("score_populations"),
                contract_name=f"{contract_name} score_populations",
            )
        ]
        population_masks: list[tuple[str, str, np.ndarray]] = []
        allow_sensitivity_fit_overlap = bool(axis.get("allow_sensitivity_fit_overlap", False))
        for population in score_populations:
            population_id = _required_text(
                population,
                "population_id",
                contract_name=f"{contract_name} score_population",
            )
            role = str(population.get("role") or "")
            mask = _mask_rows(
                rows,
                population.get("where"),
                column_names=column_names,
                contract_name=f"{contract_name} population={population_id!r}",
            )
            if not np.any(mask):
                raise ContractViolationError(f"{contract_name} population={population_id!r} matched no rows")
            if role == "sensitivity" and np.any(mask & fit_mask) and not allow_sensitivity_fit_overlap:
                raise ContractViolationError(
                    f"{contract_name} sensitivity population={population_id!r} overlaps fit rows"
                )
            population_masks.append((population_id, role, mask))

        parent_key_column = str(axis.get("parent_key") or "").strip() or None
        parent_candidate_id_column = str(axis.get("parent_candidate_id_column") or "").strip() or None
        candidate_index_by_id: dict[str, int] = {}
        if parent_key_column and parent_candidate_id_column:
            duplicate_candidates = [candidate_id for candidate_id, count in Counter(candidate_ids).items() if count > 1]
            if duplicate_candidates:
                raise ContractViolationError(
                    f"{contract_name} parent-relative scoring requires unique candidate_id values: "
                    f"{sorted(duplicate_candidates)[:5]}"
                )
            candidate_index_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}

        for endpoint_definition in _endpoint_definitions(axis):
            endpoint_definition_id = _required_text(
                endpoint_definition,
                "endpoint_definition_id",
                contract_name=f"{contract_name} endpoint",
            )
            low_mask, high_mask = _endpoint_masks(
                rows,
                fit_mask=fit_mask,
                definition=endpoint_definition,
                column_names=column_names,
                contract_name=contract_name,
            )
            low_centroid = try_l2_normalize_vector(
                np.asarray(normalized[low_mask].mean(axis=0), dtype=np.float32),
            )
            high_centroid = try_l2_normalize_vector(
                np.asarray(normalized[high_mask].mean(axis=0), dtype=np.float32),
            )
            if low_centroid is None or high_centroid is None:
                raise ContractViolationError(
                    f"{contract_name} endpoint={endpoint_definition_id!r} endpoint centroid is degenerate"
                )
            axis_vector = try_l2_normalize_vector(np.asarray(high_centroid - low_centroid, dtype=np.float32))
            if axis_vector is None:
                raise ContractViolationError(
                    f"{contract_name} endpoint={endpoint_definition_id!r} axis vector is degenerate"
                )
            similarity_to_high = np.asarray(normalized @ high_centroid, dtype=np.float32)
            similarity_to_low = np.asarray(normalized @ low_centroid, dtype=np.float32)
            endpoint_margin = np.asarray(similarity_to_high - similarity_to_low, dtype=np.float32)
            axis_projection = np.asarray(normalized @ axis_vector, dtype=np.float32)

            axis_rows.append(
                {
                    "trait_id": trait_id,
                    "axis_id": axis_id,
                    "axis_label": axis_label,
                    "view_id": view_id,
                    "endpoint_definition_id": endpoint_definition_id,
                    "primary_endpoint_definition_id": primary_endpoint_definition_id,
                    "normalization_policy": _NORMALIZATION_POLICY,
                    "fit_population_id": fit_population_id,
                    "low_endpoint_row_count": int(np.count_nonzero(low_mask)),
                    "high_endpoint_row_count": int(np.count_nonzero(high_mask)),
                    "source_value_column": source_value_column,
                    "axis_vector": [float(value) for value in axis_vector.tolist()],
                }
            )
            provenance_rows.append(
                {
                    "trait_id": trait_id,
                    "axis_id": axis_id,
                    "view_id": view_id,
                    "endpoint_definition_id": endpoint_definition_id,
                    "primary_endpoint_definition_id": primary_endpoint_definition_id,
                    "fit_population_id": fit_population_id,
                    "score_populations": [
                        {"population_id": population_id, "population_role": role}
                        for population_id, role, _ in population_masks
                    ],
                    "fit_row_count": int(np.count_nonzero(fit_mask)),
                    "low_endpoint_row_count": int(np.count_nonzero(low_mask)),
                    "high_endpoint_row_count": int(np.count_nonzero(high_mask)),
                    "source_value_column": source_value_column,
                    "parent_key": parent_key_column,
                    "status": "ok",
                    "status_reason": "",
                }
            )

            for population_id, role, population_mask in population_masks:
                for row_index in np.flatnonzero(population_mask):
                    metadata = {
                        column: rows[row_index].get(column)
                        for column in output_metadata_columns
                        if column != candidate_id_column
                    }
                    row_status = "ok"
                    row_status_reason = ""
                    source_value = (
                        _finite_float(rows[row_index].get(source_value_column)) if source_value_column else None
                    )
                    endpoint_group: str | None = None
                    if fit_mask[row_index]:
                        if low_mask[row_index]:
                            endpoint_group = "low"
                        elif high_mask[row_index]:
                            endpoint_group = "high"
                    parent_payload: dict[str, object] = {}
                    if parent_key_column and parent_candidate_id_column:
                        parent_key_raw = rows[row_index].get(parent_key_column)
                        parent_candidate_id_raw = rows[row_index].get(parent_candidate_id_column)
                        parent_candidate_id = (
                            str(parent_candidate_id_raw).strip()
                            if not _is_missing(parent_candidate_id_raw) and str(parent_candidate_id_raw).strip()
                            else None
                        )
                        parent_key_requested = not _is_missing(parent_key_raw) and bool(str(parent_key_raw).strip())
                        parent_mapping_requested = parent_key_requested or parent_candidate_id is not None
                        parent_payload = {
                            "parent_candidate_id": parent_candidate_id,
                            "parent_key": parent_key_raw,
                            "parent_axis_projection": None,
                            "axis_delta": None,
                            "orthogonal_delta": None,
                        }
                        if parent_mapping_requested and parent_candidate_id is None:
                            row_status = "invalid"
                            row_status_reason = f"missing_parent_candidate_id_column={parent_candidate_id_column}"
                        elif parent_candidate_id is not None and not parent_key_requested:
                            row_status = "invalid"
                            row_status_reason = f"missing_parent_key_column={parent_key_column}"
                        elif parent_candidate_id is not None:
                            parent_index = candidate_index_by_id.get(parent_candidate_id)
                            if parent_index is None:
                                raise ContractViolationError(
                                    f"{contract_name} parent_candidate_id is missing from view rows: "
                                    f"{parent_candidate_id!r}"
                                )
                            parent_projection = float(axis_projection[parent_index])
                            parent_residual = normalized[parent_index] - (parent_projection * axis_vector)
                            row_residual = normalized[row_index] - (float(axis_projection[row_index]) * axis_vector)
                            parent_payload.update(
                                {
                                    "parent_axis_projection": parent_projection,
                                    "axis_delta": float(axis_projection[row_index]) - parent_projection,
                                    "orthogonal_delta": float(np.linalg.norm(row_residual - parent_residual)),
                                }
                            )
                    output_rows.append(
                        {
                            **metadata,
                            "candidate_id": candidate_ids[row_index],
                            "candidate_id_column": candidate_id_column,
                            "view_id": view_id,
                            "trait_id": trait_id,
                            "axis_id": axis_id,
                            "axis_label": axis_label,
                            "endpoint_definition_id": endpoint_definition_id,
                            "primary_endpoint_definition_id": primary_endpoint_definition_id,
                            "population_id": population_id,
                            "population_role": role,
                            "similarity_to_high": float(similarity_to_high[row_index]),
                            "similarity_to_low": float(similarity_to_low[row_index]),
                            "endpoint_margin": float(endpoint_margin[row_index]),
                            "axis_projection": float(axis_projection[row_index]),
                            "endpoint_group": endpoint_group,
                            "source_value": source_value,
                            "source_value_column": source_value_column,
                            "source_value_available": source_value is not None,
                            "is_low_endpoint": bool(low_mask[row_index]),
                            "is_high_endpoint": bool(high_mask[row_index]),
                            "row_status": row_status,
                            "row_status_reason": row_status_reason,
                            **parent_payload,
                        }
                    )
                    scored_role_counts[role] += 1

    inputs = [
        ScalarInputRef(kind="view_matrix", artifact_id=view_id, path=matrix_path),
        ScalarInputRef(kind="view_rows", artifact_id=view_id, path=rows_path),
    ]
    return (
        output_rows,
        axis_rows,
        provenance_rows,
        inputs,
        {
            "view_id": view_id,
            "view_rows": len(rows),
            "scored_role_counts": dict(scored_role_counts),
        },
    )


def build_trait_axis_projection_rows_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, object],
) -> BuiltScalarArtifact:
    candidate_id_column = str(_optional_param(params, "candidate_id_column", default="id"))
    candidate_views = _unique_nonempty_strings(
        _as_list(_require_param(params, "candidate_views"), contract_name="candidate_views"),
        contract_name=f"{_ROWS_KIND} candidate_views",
    )
    axes = _validate_axes(_require_param(params, "axes"))

    output_rows: list[dict[str, object]] = []
    axis_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    inputs: list[ScalarInputRef] = []
    view_stats: list[dict[str, object]] = []
    for view_id in candidate_views:
        view_rows, view_axis_rows, view_provenance_rows, view_inputs, stats = _axis_projection_rows_for_view(
            context=context,
            view_id=view_id,
            candidate_id_column=candidate_id_column,
            axes=axes,
            metadata_columns=_optional_param(params, "metadata_columns", default=[]),
        )
        output_rows.extend(view_rows)
        axis_rows.extend(view_axis_rows)
        provenance_rows.extend(view_provenance_rows)
        inputs.extend(view_inputs)
        view_stats.append(stats)

    table = pa.Table.from_pylist(output_rows)
    axis_table = pa.Table.from_pylist(axis_rows)
    write_table(table, artifact_dir / "table.parquet")
    write_table(axis_table, artifact_dir / "fitted_axes.parquet")
    write_json(
        artifact_dir / "provenance.json",
        {
            "schema_version": "latentdna.trait_axis_projection.provenance.v1",
            "builder_kind": _ROWS_KIND,
            "normalization_policy": _NORMALIZATION_POLICY,
            "candidate_id_column": candidate_id_column,
            "candidate_views": candidate_views,
            "axes": provenance_rows,
        },
    )
    role_counts = Counter(str(row["population_role"]) for row in output_rows)
    invalid_row_count = sum(1 for row in output_rows if row.get("row_status") != "ok")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=table.num_rows,
        columns=table.column_names,
        inputs=inputs,
        outputs=[
            ("provenance.json", "application/json"),
            ("fitted_axes.parquet", "application/x-parquet"),
        ],
        stats={
            "configured_trait_count": len(axes),
            "configured_view_count": len(candidate_views),
            "scored_row_count": table.num_rows,
            "invalid_skipped_row_count": invalid_row_count,
            "failed_axis_endpoint_count": 0,
            "axis_endpoint_count": len(axis_rows),
            "normalization_policy": _NORMALIZATION_POLICY,
            "population_role_counts": dict(role_counts),
            "view_stats": view_stats,
        },
    )


def _source_scalar_paths(context: WorkspaceContext, source_scalar: str) -> tuple[Path, Path]:
    scalar_dir = context.output_root / "scalars" / source_scalar
    table_path = scalar_dir / "table.parquet"
    manifest_path = scalar_dir / "manifest.json"
    if not table_path.is_file():
        raise MissingArtifactError(f"{_SUMMARY_KIND} is missing source scalar table: {table_path}")
    if not manifest_path.is_file():
        raise MissingArtifactError(f"{_SUMMARY_KIND} is missing source scalar manifest: {manifest_path}")
    return table_path, manifest_path


def _validate_rows_scalar_manifest(
    manifest: dict[str, object],
    *,
    source_scalar: str,
    require_axes: bool,
) -> None:
    if manifest.get("artifact_kind") != "scalar_table":
        raise ContractViolationError(f"{_SUMMARY_KIND} source_scalar {source_scalar!r} is not a scalar_table")
    if manifest.get("artifact_id") != source_scalar:
        raise ContractViolationError(
            f"{_SUMMARY_KIND} source_scalar manifest id mismatch: "
            f"expected {source_scalar!r}, found {manifest.get('artifact_id')!r}"
        )
    if manifest.get("status", "ok") != "ok":
        raise ContractViolationError(
            f"{_SUMMARY_KIND} source_scalar {source_scalar!r} manifest is not ok: {manifest.get('status')!r}"
        )
    params = manifest.get("params") if isinstance(manifest.get("params"), dict) else {}
    if params.get("builder_kind") != _ROWS_KIND:
        raise ContractViolationError(
            f"{_SUMMARY_KIND} source_scalar {source_scalar!r} must be produced by {_ROWS_KIND}"
        )
    output_paths = {
        str(output.get("path") or "").strip()
        for output in list(manifest.get("outputs") or [])
        if isinstance(output, dict)
    }
    required = {"table.parquet"} | ({"fitted_axes.parquet"} if require_axes else set())
    missing = sorted(required - output_paths)
    if missing:
        raise ContractViolationError(
            f"{_SUMMARY_KIND} source_scalar {source_scalar!r} manifest does not declare outputs: {missing}"
        )


def _manifest_declares_output(manifest: dict[str, object], relative_path: str) -> bool:
    return any(
        str(output.get("path") or "").strip() == relative_path
        for output in list(manifest.get("outputs") or [])
        if isinstance(output, dict)
    )


def _finite_pair_count(rows: list[dict[str, object]], left: str, right: str) -> int:
    return sum(
        1 for row in rows if _finite_float(row.get(left)) is not None and _finite_float(row.get(right)) is not None
    )


def _values(rows: list[dict[str, object]], column: str) -> np.ndarray:
    return np.asarray(
        [(_finite_float(row.get(column)) if _finite_float(row.get(column)) is not None else np.nan) for row in rows],
        dtype=np.float64,
    )


def _endpoint_effect(rows: list[dict[str, object]], score_column: str) -> tuple[float, float]:
    low_values = _values([row for row in rows if row.get("endpoint_group") == "low"], score_column)
    high_values = _values([row for row in rows if row.get("endpoint_group") == "high"], score_column)
    low_values = low_values[np.isfinite(low_values)]
    high_values = high_values[np.isfinite(high_values)]
    if low_values.size == 0 or high_values.size == 0:
        return float("nan"), float("nan")
    mean_difference = float(np.mean(high_values) - np.mean(low_values))
    if low_values.size < 2 or high_values.size < 2:
        return mean_difference, float("nan")
    pooled_var = (
        ((low_values.size - 1) * float(np.var(low_values, ddof=1)))
        + ((high_values.size - 1) * float(np.var(high_values, ddof=1)))
    ) / float(low_values.size + high_values.size - 2)
    if pooled_var <= 1e-12:
        return mean_difference, float("nan")
    return mean_difference, float(mean_difference / math.sqrt(pooled_var))


def _sign(value: float) -> int | None:
    if not math.isfinite(value):
        return None
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _matched_score_pairs(
    rows: list[dict[str, object]],
    primary_rows: list[dict[str, object]],
    *,
    score_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    primary_by_key = {
        (row.get("candidate_id"), row.get("population_id")): row
        for row in primary_rows
        if _finite_float(row.get(score_column)) is not None
    }
    left: list[float] = []
    right: list[float] = []
    for row in rows:
        value = _finite_float(row.get(score_column))
        primary = primary_by_key.get((row.get("candidate_id"), row.get("population_id")))
        primary_value = _finite_float(primary.get(score_column)) if primary is not None else None
        if value is not None and primary_value is not None:
            left.append(primary_value)
            right.append(value)
    return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)


def _summary_row(
    *,
    key: tuple[str, str, str, str],
    rows: list[dict[str, object]],
    primary_rows: list[dict[str, object]],
    score_columns: list[str],
    min_correlation_pairs: int,
    axis_vectors: dict[tuple[str, str, str, str], np.ndarray],
) -> dict[str, object]:
    trait_id, axis_id, view_id, endpoint_definition_id = key
    role_counts = Counter(str(row.get("population_role") or "") for row in rows)
    low_count = sum(1 for row in rows if row.get("endpoint_group") == "low")
    high_count = sum(1 for row in rows if row.get("endpoint_group") == "high")
    row_status_reasons = Counter(
        str(row.get("row_status_reason") or "unspecified") for row in rows if str(row.get("row_status") or "ok") != "ok"
    )
    invalid_row_count = sum(row_status_reasons.values())
    status = "ok" if rows and low_count > 0 and high_count > 0 else "insufficient"
    status_reason = "" if status == "ok" else "missing endpoint rows"
    primary_endpoint_definition_id = str(rows[0].get("primary_endpoint_definition_id") or "")
    output: dict[str, object] = {
        "trait_id": trait_id,
        "axis_id": axis_id,
        "view_id": view_id,
        "endpoint_definition_id": endpoint_definition_id,
        "primary_endpoint_definition_id": primary_endpoint_definition_id,
        "total_scored_rows": len(rows),
        "fit_population_row_count": role_counts.get("fit", 0),
        "low_endpoint_row_count": low_count,
        "high_endpoint_row_count": high_count,
        "eval_population_row_count": role_counts.get("eval", 0),
        "reference_population_row_count": role_counts.get("reference", 0),
        "sensitivity_population_row_count": role_counts.get("sensitivity", 0),
        "excluded_population_row_count": role_counts.get("excluded", 0),
        "invalid_row_count": invalid_row_count,
        "row_status_reasons": ";".join(f"{reason}:{count}" for reason, count in sorted(row_status_reasons.items())),
        "status": status,
        "status_reason": status_reason,
    }
    source_rows = [
        row
        for row in rows
        if row.get("population_role") in {"fit", "eval"}
        and bool(row.get("source_value_available"))
        and str(row.get("row_status") or "ok") == "ok"
    ]
    for score_column in score_columns:
        pair_count = _finite_pair_count(source_rows, "source_value", score_column)
        source_values = _values(source_rows, "source_value")
        score_values = _values(source_rows, score_column)
        output[f"{score_column}_source_value_pair_count"] = pair_count
        output[f"{score_column}_pearson"] = pearson_correlation(
            source_values,
            score_values,
            min_pairs=min_correlation_pairs,
        )
        output[f"{score_column}_spearman"] = spearman_correlation(
            source_values,
            score_values,
            min_pairs=min_correlation_pairs,
        )
        output[f"{score_column}_kendall"] = kendall_tau_b(
            source_values,
            score_values,
            min_pairs=min_correlation_pairs,
        )
        effect, cohen_d = _endpoint_effect(rows, score_column)
        output[f"{score_column}_endpoint_effect"] = effect
        output[f"{score_column}_endpoint_cohen_d"] = cohen_d
        primary_effect, _ = _endpoint_effect(primary_rows, score_column)
        primary_values, endpoint_values = _matched_score_pairs(
            rows,
            primary_rows,
            score_column=score_column,
        )
        output[f"{score_column}_primary_overlap_count"] = int(primary_values.size)
        output[f"{score_column}_primary_spearman"] = spearman_correlation(
            primary_values,
            endpoint_values,
            min_pairs=2,
        )
        primary_sign = _sign(primary_effect)
        endpoint_sign = _sign(effect)
        output[f"{score_column}_effect_sign_matches_primary"] = (
            primary_sign == endpoint_sign if primary_sign is not None and endpoint_sign is not None else None
        )
    primary_axis_key = (trait_id, axis_id, view_id, primary_endpoint_definition_id)
    endpoint_axis_key = (trait_id, axis_id, view_id, endpoint_definition_id)
    primary_axis_vector = axis_vectors.get(primary_axis_key)
    endpoint_axis_vector = axis_vectors.get(endpoint_axis_key)
    if primary_axis_vector is not None and endpoint_axis_vector is not None:
        concordance = float(np.clip(np.dot(primary_axis_vector, endpoint_axis_vector), -1.0, 1.0))
        output["axis_vector_primary_concordance"] = concordance
        output["axis_vector_primary_angle"] = float(math.acos(concordance))
    else:
        output["axis_vector_primary_concordance"] = None
        output["axis_vector_primary_angle"] = None
    return output


def _axis_vector(value: object) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ContractViolationError(f"{_SUMMARY_KIND} fitted axis_vector must be a non-empty list")
    vector = np.asarray([float(item) for item in value], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ContractViolationError(f"{_SUMMARY_KIND} fitted axis_vector contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        raise ContractViolationError(f"{_SUMMARY_KIND} fitted axis_vector is degenerate")
    return vector / norm


def _concordance_rows(
    *,
    axes_table: pa.Table,
    compare_trait_ids: list[object],
) -> list[dict[str, object]]:
    axes = axes_table.to_pylist()
    output: list[dict[str, object]] = []
    for pair in compare_trait_ids:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ContractViolationError(f"{_SUMMARY_KIND} concordance.compare_trait_ids entries must be pairs")
        left_trait, right_trait = str(pair[0]), str(pair[1])
        for left in axes:
            if str(left.get("trait_id")) != left_trait:
                continue
            for right in axes:
                if str(right.get("trait_id")) != right_trait:
                    continue
                if left.get("view_id") != right.get("view_id"):
                    continue
                if left.get("endpoint_definition_id") != right.get("endpoint_definition_id"):
                    continue
                if left.get("normalization_policy") != right.get("normalization_policy"):
                    continue
                left_vector = _axis_vector(left.get("axis_vector"))
                right_vector = _axis_vector(right.get("axis_vector"))
                if left_vector.shape != right_vector.shape:
                    raise ContractViolationError(
                        f"{_SUMMARY_KIND} cannot compare axis vectors with different dimensions: "
                        f"{left_trait!r}, {right_trait!r}"
                    )
                concordance = float(np.clip(np.dot(left_vector, right_vector), -1.0, 1.0))
                output.append(
                    {
                        "view_id": str(left.get("view_id")),
                        "endpoint_definition_id": str(left.get("endpoint_definition_id")),
                        "left_trait_id": left_trait,
                        "right_trait_id": right_trait,
                        "left_axis_id": str(left.get("axis_id")),
                        "right_axis_id": str(right.get("axis_id")),
                        "axis_concordance": concordance,
                        "axis_angle": float(math.acos(concordance)),
                        "normalization_policy": str(left.get("normalization_policy")),
                    }
                )
    return output


def _concordance_schema() -> pa.Schema:
    return pa.schema(
        [
            ("view_id", pa.string()),
            ("endpoint_definition_id", pa.string()),
            ("left_trait_id", pa.string()),
            ("right_trait_id", pa.string()),
            ("left_axis_id", pa.string()),
            ("right_axis_id", pa.string()),
            ("axis_concordance", pa.float64()),
            ("axis_angle", pa.float64()),
            ("normalization_policy", pa.string()),
        ]
    )


def _axis_vectors_by_key(axes_table: pa.Table) -> dict[tuple[str, str, str, str], np.ndarray]:
    required_columns = {"trait_id", "axis_id", "view_id", "endpoint_definition_id", "axis_vector"}
    missing_columns = sorted(required_columns - set(axes_table.column_names))
    if missing_columns:
        raise ContractViolationError(f"{_SUMMARY_KIND} fitted axes table is missing columns: {missing_columns}")
    output: dict[tuple[str, str, str, str], np.ndarray] = {}
    for row in axes_table.to_pylist():
        key = (
            str(row.get("trait_id") or ""),
            str(row.get("axis_id") or ""),
            str(row.get("view_id") or ""),
            str(row.get("endpoint_definition_id") or ""),
        )
        if any(not part for part in key):
            raise ContractViolationError(f"{_SUMMARY_KIND} fitted axes table contains an empty axis key")
        if key in output:
            raise ContractViolationError(f"{_SUMMARY_KIND} fitted axes table contains a duplicate axis key: {key}")
        output[key] = _axis_vector(row.get("axis_vector"))
    return output


def build_trait_axis_projection_summary_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, object],
) -> BuiltScalarArtifact:
    source_scalar = str(_require_param(params, "source_scalar"))
    concordance_config = _optional_param(params, "concordance", default={})
    concordance_enabled = isinstance(concordance_config, dict) and bool(concordance_config.get("enabled", False))
    table_path, manifest_path = _source_scalar_paths(context, source_scalar)
    manifest = read_json(manifest_path)
    _validate_rows_scalar_manifest(manifest, source_scalar=source_scalar, require_axes=concordance_enabled)
    source_scalar_dir = context.output_root / "scalars" / source_scalar
    axes_path = source_scalar_dir / "fitted_axes.parquet"
    axes_table: pa.Table | None = None
    axis_vectors: dict[tuple[str, str, str, str], np.ndarray] = {}
    axes_declared = _manifest_declares_output(manifest, "fitted_axes.parquet")
    if axes_declared:
        if not axes_path.is_file():
            raise MissingArtifactError(f"{_SUMMARY_KIND} is missing declared fitted axes sidecar: {axes_path}")
        axes_table = read_table(axes_path)
        axis_vectors = _axis_vectors_by_key(axes_table)
    table = read_table(table_path)
    missing_columns = sorted(_REQUIRED_ROW_COLUMNS - set(table.column_names))
    if missing_columns:
        raise ContractViolationError(f"{_SUMMARY_KIND} source scalar is missing row columns: {missing_columns}")
    rows = table.to_pylist()
    score_columns = [str(column) for column in _optional_param(params, "score_columns", default=["axis_projection"])]
    if not score_columns:
        raise ContractViolationError(f"{_SUMMARY_KIND} score_columns cannot be empty")
    missing_score_columns = sorted(set(score_columns) - set(table.column_names))
    if missing_score_columns:
        raise ContractViolationError(f"{_SUMMARY_KIND} source scalar is missing score columns: {missing_score_columns}")
    min_correlation_pairs = int(_optional_param(params, "min_correlation_pairs", default=3))
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("trait_id") or ""),
            str(row.get("axis_id") or ""),
            str(row.get("view_id") or ""),
            str(row.get("endpoint_definition_id") or ""),
        )
        grouped[key].append(row)
    primary_groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for key, group_rows in grouped.items():
        trait_id, axis_id, view_id, endpoint_id = key
        primary_id = str(group_rows[0].get("primary_endpoint_definition_id") or endpoint_id)
        if endpoint_id == primary_id:
            primary_groups[(trait_id, axis_id, view_id)] = group_rows
    summary_rows = [
        _summary_row(
            key=key,
            rows=group_rows,
            primary_rows=primary_groups.get((key[0], key[1], key[2]), group_rows),
            score_columns=score_columns,
            min_correlation_pairs=min_correlation_pairs,
            axis_vectors=axis_vectors,
        )
        for key, group_rows in sorted(grouped.items())
    ]
    summary_table = pa.Table.from_pylist(summary_rows)
    write_table(summary_table, artifact_dir / "table.parquet")

    outputs: list[tuple[str, str]] = []
    extra_stats: dict[str, object] = {}
    if concordance_enabled:
        if axes_table is None:
            raise MissingArtifactError(f"{_SUMMARY_KIND} is missing fitted axes sidecar: {axes_path}")
        compare_trait_ids = list(concordance_config.get("compare_trait_ids") or [])
        if not compare_trait_ids:
            raise ContractViolationError(f"{_SUMMARY_KIND} concordance.compare_trait_ids cannot be empty")
        concordance_rows = _concordance_rows(axes_table=axes_table, compare_trait_ids=compare_trait_ids)
        concordance_table = (
            pa.Table.from_pylist(concordance_rows)
            if concordance_rows
            else pa.Table.from_pylist([], schema=_concordance_schema())
        )
        write_table(concordance_table, artifact_dir / "axis_concordance.parquet")
        outputs.append(("axis_concordance.parquet", "application/x-parquet"))
        extra_stats["axis_concordance_rows"] = concordance_table.num_rows

    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=summary_table.num_rows,
        columns=summary_table.column_names,
        inputs=[
            ScalarInputRef(kind="scalar_table", artifact_id=source_scalar, path=table_path),
            ScalarInputRef(kind="scalar_manifest", artifact_id=source_scalar, path=manifest_path),
            *(
                [ScalarInputRef(kind="scalar_sidecar", artifact_id=source_scalar, path=axes_path)]
                if axes_declared
                else []
            ),
        ],
        outputs=outputs,
        stats={
            "source_scalar": source_scalar,
            "summary_group_count": summary_table.num_rows,
            "score_columns": score_columns,
            **extra_stats,
        },
    )
