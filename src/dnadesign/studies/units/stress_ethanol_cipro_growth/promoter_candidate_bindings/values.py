"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/values.py

Shared scalar and table validation for promoter binding contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import PromoterCandidateBindingsError


def required_text(value: Any, *, field: str, row_id: str | None = None) -> str:
    if value is None or scalar_missing(value) or not str(value).strip():
        suffix = f" for {row_id!r}" if row_id is not None else ""
        raise PromoterCandidateBindingsError(f"Missing required {field}{suffix}.")
    return str(value).strip()


def required_sha256(value: Any, *, field: str) -> str:
    text = required_text(value, field=field).lower().removeprefix("sha256:")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise PromoterCandidateBindingsError(f"{field} must be a 64-character hexadecimal SHA-256 digest.")
    return text


def optional_value(value: Any) -> Any:
    return None if value is None or scalar_missing(value) else value


def scalar_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def has_value(value: Any) -> bool:
    return value is not None and not scalar_missing(value) and bool(str(value).strip())


def nonempty_collection(value: Any) -> bool:
    if value is None or scalar_missing(value):
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PromoterCandidateBindingsError(f"{label} missing required columns: {missing}")
