"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/reader_records/validation.py

Primitive validation for Reader record handoffs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


class ReaderDataframeRecordError(ValueError):
    """Raised when Reader's public record handoff fails its contract."""


ReaderRecordError = ReaderDataframeRecordError


def require_contained(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReaderDataframeRecordError(f"{label} escapes {root}: {path}") from exc


def mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReaderDataframeRecordError(f"{label} must be an object")
    return value


def list_value(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ReaderDataframeRecordError(f"{label} must be an array")
    return value


def text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderDataframeRecordError(f"{label} must be a non-empty string")
    return value.strip()


def optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return text(value, label=label)


def sha256_digest(value: object, *, label: str) -> str:
    token = text(value, label=label)
    if not token.startswith("sha256:") or len(token) != 71:
        raise ReaderDataframeRecordError(f"{label} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in token[7:]):
        raise ReaderDataframeRecordError(f"{label} must be a lowercase sha256 digest")
    return token


def positive_revision(value: object, *, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ReaderDataframeRecordError(f"{label} must be a positive integer")
    return value


def nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReaderDataframeRecordError(f"{label} must be a nonnegative integer")
    return value
