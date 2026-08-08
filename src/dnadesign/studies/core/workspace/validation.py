"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/workspace/validation.py

Primitive validation shared by study workspace loaders.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARTIFACT_URI_SCHEMES = frozenset({"https", "s3", "gs"})


def mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def sequence(value: object, *, label: str, allow_empty: bool = False) -> list[object]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{label} must be {qualifier}")
    return list(value)


def text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return text(value, label=label)


def identifier(value: object, *, label: str) -> str:
    token = text(value, label=label)
    if not _IDENTIFIER.fullmatch(token):
        raise ValueError(f"{label} must be a lowercase identifier")
    return token


def sha256_digest(value: object, *, label: str) -> str:
    token = text(value, label=label)
    if not _SHA256.fullmatch(token):
        raise ValueError(f"{label} must be a lowercase sha256 digest")
    return token


def iso_date(value: object, *, label: str) -> str:
    if type(value) is date:
        return value.isoformat()
    token = text(value, label=label)
    try:
        date.fromisoformat(token)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO 8601 calendar date") from exc
    return token


def require_keys(payload: dict[str, object], *, required: frozenset[str], label: str) -> None:
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise ValueError(f"{label} is missing required key(s): {', '.join(missing)}")


def reject_unknown_keys(payload: dict[str, object], *, allowed: frozenset[str], label: str) -> None:
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(f"{label} has unknown key(s): {', '.join(unknown)}")


def relative_file(*, base: Path, value: object, boundary: Path, label: str) -> Path:
    raw = text(value, label=label)
    relative = Path(raw)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError(f"{label} must be a repository-relative path without '..'")
    boundary_resolved = boundary.expanduser().resolve()
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(boundary_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} escapes workspace root {boundary_resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def artifact_uri(value: object, *, label: str) -> str:
    uri = text(value, label=label)
    parsed = urlsplit(uri)
    if parsed.scheme not in _ARTIFACT_URI_SCHEMES or not parsed.netloc:
        schemes = ", ".join(sorted(_ARTIFACT_URI_SCHEMES))
        raise ValueError(f"{label} must use one of these external URI schemes: {schemes}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not embed credentials")
    return uri


def string_mapping(value: object, *, label: str) -> dict[str, str]:
    payload = mapping(value, label=label)
    if not payload:
        raise ValueError(f"{label} is required and must not be empty")
    return {identifier(key, label=f"{label} key"): text(item, label=f"{label}.{key}") for key, item in payload.items()}


__all__ = [
    "artifact_uri",
    "identifier",
    "iso_date",
    "mapping",
    "optional_text",
    "relative_file",
    "reject_unknown_keys",
    "require_keys",
    "sequence",
    "sha256_digest",
    "string_mapping",
    "text",
]
