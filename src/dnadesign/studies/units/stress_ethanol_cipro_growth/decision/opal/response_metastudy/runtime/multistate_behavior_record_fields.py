"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_record_fields.py

Strict scalar and mapping fields shared by behavior bundle verifiers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

ACTIVATION = {"campaign": "prohibited", "synthesis": "prohibited"}
SOURCE_DIGEST_FIELDS = {
    "reader_bundle_manifest_sha256",
    "reader_request_sha256",
    "candidate_bindings_manifest_sha256",
    "observation_policy_sha256",
}


def mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping.")
    return value


def require_fields(value: dict[str, object], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} fields are incomplete or unexpected.")


def require_literals(
    value: dict[str, object],
    expected: dict[str, object],
    *,
    context: str,
    exact: bool = False,
) -> None:
    if exact:
        require_fields(value, set(expected), context=context)
    for field, literal in expected.items():
        if value.get(field) != literal:
            raise ValueError(f"{context}.{field} must be {literal!r}.")


def positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be positive finite numeric evidence.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be positive finite numeric evidence.")
    return result


def positive_int(value: object, *, field: str) -> int:
    result = nonnegative_int(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive.")
    return result


def nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer.")
    return value


def prefixed_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError(f"{field} must be a canonical SHA-256 digest.")
    unprefixed_digest(value.removeprefix("sha256:"), field=field)
    return value


def unprefixed_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return value


__all__ = [
    "ACTIVATION",
    "SOURCE_DIGEST_FIELDS",
    "mapping",
    "nonnegative_int",
    "positive_float",
    "positive_int",
    "prefixed_digest",
    "require_fields",
    "require_literals",
    "unprefixed_digest",
]
