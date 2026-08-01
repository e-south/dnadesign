"""Primitive validation helpers for sensitivity-coverage contracts."""

from __future__ import annotations

from ..contracts._values import MetastudyContractError


def require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise MetastudyContractError(f"{label} must be a lowercase sha256 digest")
    return value


__all__ = ["require_digest"]
