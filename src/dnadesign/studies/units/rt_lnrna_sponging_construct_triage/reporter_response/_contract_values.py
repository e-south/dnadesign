"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/_contract_values.py

Shared scalar and canonical-value validation for reporter-response contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass


class ReporterResponseContractError(ValueError):
    """Raised when a reporter-response profile violates its study contract."""


def ordered_dose_grid(values: Iterable[float]) -> tuple[float, ...]:
    grid = tuple(finite_number(value, field_name="dose_grid_uM") for value in values)
    if not grid:
        raise ReporterResponseContractError("dose_grid_uM must not be empty")
    if any(value <= 0.0 for value in grid):
        raise ReporterResponseContractError("dose_grid_uM values must be positive")
    if any(left >= right for left, right in zip(grid, grid[1:], strict=False)):
        raise ReporterResponseContractError("dose_grid_uM must be strictly increasing without duplicates")
    return grid


def explicit_id_set(values: tuple[str, ...], *, field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ReporterResponseContractError(f"{field_name} must be a non-empty tuple")
    for value in values:
        required_text(value, field_name=field_name)
    if len(values) != len(set(values)):
        raise ReporterResponseContractError(f"{field_name} must not contain duplicates")


def required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReporterResponseContractError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ReporterResponseContractError(f"{field_name} must not contain surrounding whitespace")
    return value


def finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ReporterResponseContractError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReporterResponseContractError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ReporterResponseContractError(f"{field_name} must be a finite number")
    return result


def nonnegative_number(value: object, *, field_name: str) -> float:
    result = finite_number(value, field_name=field_name)
    if result < 0.0:
        raise ReporterResponseContractError(f"{field_name} must be non-negative")
    return result


def positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ReporterResponseContractError(f"{field_name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ReporterResponseContractError(f"{field_name} must be a positive integer") from exc
    if result <= 0:
        raise ReporterResponseContractError(f"{field_name} must be a positive integer")
    return result


def sha256_digest(value: object, *, field_name: str) -> str:
    token = required_text(value, field_name=field_name)
    if len(token) != 71 or not token.startswith("sha256:"):
        raise ReporterResponseContractError(f"{field_name} must be a lowercase sha256 digest")
    if any(character not in "0123456789abcdef" for character in token[7:]):
        raise ReporterResponseContractError(f"{field_name} must be a lowercase sha256 digest")
    return token


def json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [json_value(child) for child in value]
    return value


__all__ = [
    "ReporterResponseContractError",
    "explicit_id_set",
    "finite_number",
    "json_value",
    "nonnegative_number",
    "ordered_dose_grid",
    "positive_integer",
    "required_text",
    "sha256_digest",
]
