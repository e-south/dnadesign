"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/references/sets.py

Reference-set selection helpers shared by sampling, plots, and notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isnan
from typing import Any


@dataclass(frozen=True, slots=True)
class ReferenceSetResolution:
    expected_ids: list[str]
    matched_ids: list[str]
    selected_rows: list[dict[str, object]]
    missing_columns: list[str]
    complete: bool


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return isnan(value)
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False
    return False


def _as_comparable(value: object) -> str:
    return "" if _is_missing(value) else str(value)


def _selector_value_matches(value: object, selector: object) -> bool:
    if getattr(selector, "non_null", False) and _is_missing(value):
        return False
    equals = getattr(selector, "equals", None)
    if equals is not None and _as_comparable(value) != _as_comparable(equals):
        return False
    in_values = list(getattr(selector, "in_values", []) or [])
    if in_values and _as_comparable(value) not in {_as_comparable(item) for item in in_values}:
        return False
    regex = getattr(selector, "regex", None)
    if regex is not None and re.search(str(regex), _as_comparable(value)) is None:
        return False
    not_regex = getattr(selector, "not_regex", None)
    if not_regex is not None and re.search(str(not_regex), _as_comparable(value)) is not None:
        return False
    return True


def _selector_matches(row: Mapping[str, object], selector: object) -> bool:
    column = str(getattr(selector, "column"))
    return _selector_value_matches(row.get(column), selector)


def _column_value(values: Sequence[Any], row_index: int) -> Any:
    iloc = getattr(values, "iloc", None)
    if iloc is not None:
        return iloc[row_index]
    return values[row_index]


def reference_set_required_columns(reference_set: object) -> list[str]:
    columns = [str(getattr(reference_set, "match_column", ""))]
    label_column = getattr(reference_set, "label_column", None)
    if label_column:
        columns.append(str(label_column))
    for selector in getattr(reference_set, "where", []) or []:
        columns.append(str(getattr(selector, "column")))
    for selector in getattr(reference_set, "where_all", []) or []:
        columns.append(str(getattr(selector, "column")))
    return list(dict.fromkeys(column for column in columns if column))


def _missing_required_columns_in_rows(
    rows: Sequence[Mapping[str, object]],
    required_columns: Sequence[str],
) -> list[str]:
    if not rows:
        return []
    missing: list[str] = []
    for column in required_columns:
        if any(column not in row for row in rows):
            missing.append(column)
    return missing


def resolve_reference_set_rows(
    reference_set: object,
    rows: Sequence[Mapping[str, object]],
) -> ReferenceSetResolution:
    required_columns = reference_set_required_columns(reference_set)
    missing_columns = _missing_required_columns_in_rows(rows, required_columns)
    if missing_columns:
        return ReferenceSetResolution(
            expected_ids=[str(value) for value in getattr(reference_set, "ids", [])],
            matched_ids=[],
            selected_rows=[],
            missing_columns=missing_columns,
            complete=False,
        )

    match_column = str(getattr(reference_set, "match_column"))
    explicit_ids = [str(value) for value in getattr(reference_set, "ids", [])]
    selector_ids: list[str] = []
    selectors = list(getattr(reference_set, "where", []) or [])
    all_selectors = list(getattr(reference_set, "where_all", []) or [])
    if selectors:
        for row in rows:
            match_value = row.get(match_column)
            if _is_missing(match_value):
                continue
            if any(_selector_matches(row, selector) for selector in selectors):
                selector_ids.append(str(match_value))
    if all_selectors:
        for row in rows:
            match_value = row.get(match_column)
            if _is_missing(match_value):
                continue
            if all(_selector_matches(row, selector) for selector in all_selectors):
                selector_ids.append(str(match_value))

    expected_ids = list(dict.fromkeys([*explicit_ids, *selector_ids]))
    selected_by_id = {
        str(row.get(match_column)): dict(row) for row in rows if str(row.get(match_column)) in expected_ids
    }
    matched_ids = [value for value in expected_ids if value in selected_by_id]
    require_non_empty = bool(getattr(reference_set, "require_non_empty", True))
    complete = len(matched_ids) == len(expected_ids) and (bool(expected_ids) or not require_non_empty)
    return ReferenceSetResolution(
        expected_ids=expected_ids,
        matched_ids=matched_ids,
        selected_rows=[selected_by_id[value] for value in matched_ids],
        missing_columns=[],
        complete=complete,
    )


def resolve_reference_set_ids_from_columns(
    reference_set: object,
    columns: Mapping[str, Sequence[Any]],
) -> ReferenceSetResolution:
    required_columns = reference_set_required_columns(reference_set)
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        return ReferenceSetResolution(
            expected_ids=[str(value) for value in getattr(reference_set, "ids", [])],
            matched_ids=[],
            selected_rows=[],
            missing_columns=missing_columns,
            complete=False,
        )
    expected_ids = [str(value) for value in getattr(reference_set, "ids", [])]
    if not columns:
        return ReferenceSetResolution(
            expected_ids=expected_ids,
            matched_ids=[],
            selected_rows=[],
            missing_columns=[],
            complete=not bool(getattr(reference_set, "require_non_empty", True)),
        )

    lengths = {len(columns[column]) for column in required_columns}
    if len(lengths) != 1:
        raise ValueError("reference set column inputs must share one row axis")
    row_count = next(iter(lengths))
    match_column = str(getattr(reference_set, "match_column"))
    explicit_ids = list(expected_ids)
    explicit_id_set = set(explicit_ids)
    selectors = list(getattr(reference_set, "where", []) or [])
    all_selectors = list(getattr(reference_set, "where_all", []) or [])
    selector_ids: list[str] = []
    selected_by_id: dict[str, dict[str, object]] = {}

    for row_index in range(row_count):
        match_value = _column_value(columns[match_column], row_index)
        if _is_missing(match_value):
            continue
        match_text = str(match_value)
        selector_matched = False
        if selectors and any(
            _selector_value_matches(_column_value(columns[str(getattr(selector, "column"))], row_index), selector)
            for selector in selectors
        ):
            selector_ids.append(match_text)
            selector_matched = True
        if all_selectors and all(
            _selector_value_matches(_column_value(columns[str(getattr(selector, "column"))], row_index), selector)
            for selector in all_selectors
        ):
            selector_ids.append(match_text)
            selector_matched = True
        if match_text in explicit_id_set or selector_matched:
            selected_by_id[match_text] = {
                column: _column_value(columns[column], row_index) for column in required_columns
            }

    expected_ids = list(dict.fromkeys([*explicit_ids, *selector_ids]))
    matched_ids = [value for value in expected_ids if value in selected_by_id]
    require_non_empty = bool(getattr(reference_set, "require_non_empty", True))
    complete = len(matched_ids) == len(expected_ids) and (bool(expected_ids) or not require_non_empty)
    return ReferenceSetResolution(
        expected_ids=expected_ids,
        matched_ids=matched_ids,
        selected_rows=[selected_by_id[value] for value in matched_ids],
        missing_columns=[],
        complete=complete,
    )
