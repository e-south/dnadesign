"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/parsing.py

Primitive parsing and validation helpers for OPS status inputs and metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def required_text(value: str | None, *, flag_name: str, status_kind: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"status kind '{status_kind}' requires {flag_name}")
    return str(value).strip()


def required_metadata_text(value: object, *, label: str, source: Path) -> str:
    text = string_or_none(value)
    if text is None:
        raise ValueError(f"{label} is required in {source}")
    return text


def optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError as exc:
        raise ValueError(f"expected integer value, received: {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"expected non-negative integer value, received: {value!r}")
    return parsed


def string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def string_list_or_empty(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = string_or_none(item)
        if text is not None:
            result.append(text)
    return result


__all__ = [
    "optional_positive_int",
    "required_metadata_text",
    "required_text",
    "string_list_or_empty",
    "string_or_none",
]
