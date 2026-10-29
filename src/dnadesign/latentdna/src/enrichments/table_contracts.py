"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/enrichments/table_contracts.py

Generic table and config validation helpers for enrichment builders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pyarrow as pa

from ..contracts.errors import ContractViolationError


def require_columns(table: pa.Table, columns: Iterable[str], *, contract_name: str) -> None:
    missing = sorted(column for column in columns if column not in table.column_names)
    if missing:
        raise ContractViolationError(f"{contract_name} missing required columns: {missing}")


def require_sequence(value: object, *, field_name: str, contract_name: str) -> list[object]:
    if isinstance(value, str) or isinstance(value, Mapping) or not isinstance(value, Iterable):
        raise ContractViolationError(f"{contract_name} {field_name} must be a sequence, not a scalar or mapping")
    return list(value)


def string_values(value: object, *, field_name: str, contract_name: str) -> list[str]:
    values = [str(item).strip() for item in require_sequence(value, field_name=field_name, contract_name=contract_name)]
    return [item for item in values if item]


def filter_indices(table: pa.Table, where: dict[str, object], *, contract_name: str) -> list[int]:
    if not isinstance(where, dict):
        raise ContractViolationError(f"{contract_name} requires a where mapping")
    column = str(where.get("column") or "").strip()
    if not column:
        raise ContractViolationError(f"{contract_name} where requires column")
    operators = [operator for operator in ("equals", "in") if operator in where]
    if len(operators) != 1:
        raise ContractViolationError(f"{contract_name} where supports exactly one of equals or in")
    require_columns(table, [column], contract_name=contract_name)
    values = table[column].to_pylist()
    if operators[0] == "equals":
        expected = where["equals"]
        return [index for index, value in enumerate(values) if value == expected]
    expected_values = set(require_sequence(where["in"], field_name=f"where.{column}.in", contract_name=contract_name))
    return [index for index, value in enumerate(values) if value in expected_values]
