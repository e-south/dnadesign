"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/validation.py

Strict scalar and schema validation for RT-lnRNA subject bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from .contracts import SubjectBindingContractError


def require_exact_fields(payload: Mapping[str, object], expected: set[str], *, label: str) -> None:
    observed = set(payload)
    unknown = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unknown:
        raise SubjectBindingContractError(f"{label} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise SubjectBindingContractError(f"{label} is missing required field(s): {', '.join(missing)}")


def digest(value: object, *, label: str) -> str:
    value_text = text(value, label=label)
    normalized = value_text if value_text.startswith("sha256:") else f"sha256:{value_text}"
    if len(normalized) != 71 or any(char not in "0123456789abcdef" for char in normalized[7:]):
        raise SubjectBindingContractError(f"{label} must be a lowercase sha256 digest")
    return normalized


def sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def span(value: object, *, label: str) -> tuple[int, int]:
    rows = object_list(value, label=label)
    if len(rows) != 2 or not all(isinstance(item, int) and not isinstance(item, bool) for item in rows):
        raise SubjectBindingContractError(f"{label} must be [start, end] integers")
    start, end = int(rows[0]), int(rows[1])
    if start < 0 or end <= start:
        raise SubjectBindingContractError(f"{label} must be a non-empty zero-based half-open span")
    return start, end


def text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubjectBindingContractError(f"{label} must be a non-empty string")
    return value.strip()


def mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SubjectBindingContractError(f"{label} must be a mapping")
    return value


def object_list(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise SubjectBindingContractError(f"{label} must be a list")
    return value


__all__ = ["digest", "mapping", "object_list", "require_exact_fields", "sha256", "span", "text"]
