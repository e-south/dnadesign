"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/contracts/request/validation.py

Primitive validation rules for the TriJunction request boundary.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from ...errors import TriJunctionConfigError
from ...sequence import DNA_ALPHABET
from .limits import MAX_REQUEST_IDENTIFIER_BYTES, MAX_REQUEST_PLAIN_TEXT_BYTES

COMPLEMENT_END_PREPARATIONS = frozenset(
    {
        "vendor_5_prime_phosphate",
        "downstream_phosphorylation",
    }
)
RECOVERY_PRIMER_MODES = frozenset({"target_specific", "universal"})

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def require_exact_fields(
    raw: Mapping[str, object],
    *,
    required: frozenset[str],
    context: str,
) -> None:
    """Reject unknown and missing keys at one mapping boundary."""

    keys = set(raw)
    if unknown := sorted(keys - required):
        raise TriJunctionConfigError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing := sorted(required - keys):
        raise TriJunctionConfigError(f"{context} is missing field(s): {', '.join(missing)}")


def require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    """Return a string-keyed mapping or reject the boundary value."""

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TriJunctionConfigError(f"{context} must be an object with string keys")
    return value


def require_nonempty_string(value: object, *, context: str) -> str:
    """Return a non-empty, surrounding-whitespace-free string."""

    if not isinstance(value, str) or not value.strip():
        raise TriJunctionConfigError(f"{context} must be a non-empty string")
    if value != value.strip():
        raise TriJunctionConfigError(f"{context} must not contain leading or trailing whitespace")
    return value


def require_identifier(value: object, *, context: str) -> str:
    """Return a filesystem- and table-safe request identifier."""

    identifier = require_nonempty_string(value, context=context)
    if _IDENTIFIER.fullmatch(identifier) is None:
        raise TriJunctionConfigError(
            f"{context} must start with an alphanumeric character and contain only alphanumerics, '.', '_', or '-'"
        )
    if len(identifier.encode("ascii")) > MAX_REQUEST_IDENTIFIER_BYTES:
        raise TriJunctionConfigError(f"{context} must not exceed {MAX_REQUEST_IDENTIFIER_BYTES} ASCII bytes")
    return identifier


def require_plain_text(value: object, *, context: str) -> str:
    """Return plain text safe for tabular order exports."""

    text = require_nonempty_string(value, context=context)
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise TriJunctionConfigError(f"{context} must not contain control characters")
    if text[0] in "=+-@":
        raise TriJunctionConfigError(f"{context} must not begin with a spreadsheet formula marker")
    if len(text.encode("utf-8")) > MAX_REQUEST_PLAIN_TEXT_BYTES:
        raise TriJunctionConfigError(f"{context} must not exceed {MAX_REQUEST_PLAIN_TEXT_BYTES} UTF-8 bytes")
    return text


def require_int(value: object, *, context: str, minimum: int) -> int:
    """Return a bounded integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if minimum == 0 else "positive"
        raise TriJunctionConfigError(f"{context} must be a {qualifier} integer")
    return value


def require_fraction(value: object, *, context: str) -> float:
    """Return a numeric fraction in the closed unit interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TriJunctionConfigError(f"{context} must be a number between 0 and 1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise TriJunctionConfigError(f"{context} must be between 0 and 1")
    return result


def require_dna(value: object, *, context: str) -> str:
    """Return an exact uppercase ACGT sequence."""

    if not isinstance(value, str) or not value or not set(value) <= DNA_ALPHABET:
        raise TriJunctionConfigError(f"{context} must be a non-empty uppercase ACGT sequence")
    return value


def require_optional_dna(value: object, *, context: str) -> str:
    """Return an exact uppercase ACGT sequence or an explicit empty value."""

    if not isinstance(value, str) or not set(value) <= DNA_ALPHABET:
        raise TriJunctionConfigError(f"{context} must be an uppercase ACGT sequence or an empty string")
    return value


def parse_recovery_primer_mode(value: object, *, context: str) -> str:
    """Parse the closed recovery-primer mode vocabulary."""

    if not isinstance(value, str) or value not in RECOVERY_PRIMER_MODES:
        allowed = ", ".join(sorted(RECOVERY_PRIMER_MODES))
        raise TriJunctionConfigError(f"{context} must be one of: {allowed}")
    return value
