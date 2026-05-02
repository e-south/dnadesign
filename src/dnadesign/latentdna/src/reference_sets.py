"""
Reference-set selection helpers shared by sampling, plots, and notebooks.
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
    return False


def _as_comparable(value: object) -> str:
    return "" if _is_missing(value) else str(value)


def _selector_matches(row: Mapping[str, object], selector: object) -> bool:
    column = str(getattr(selector, "column"))
    value = row.get(column)
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


def resolve_reference_set_rows(
    reference_set: object,
    rows: Sequence[Mapping[str, object]],
) -> ReferenceSetResolution:
    required_columns = reference_set_required_columns(reference_set)
    missing_columns = [column for column in required_columns if rows and column not in rows[0]]
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
    if not columns:
        rows: list[dict[str, object]] = []
    else:
        row_count = len(next(iter(columns.values())))
        rows = [{column: values[index] for column, values in columns.items()} for index in range(row_count)]
    return resolve_reference_set_rows(reference_set, rows)
