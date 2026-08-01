"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/_values.py

Shared strict values, validators, and canonical digest functions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math


class MetastudyContractError(ValueError):
    """Raised when meta-study evidence or a decision violates the protocol."""


def canonical_digest(value: object) -> str:
    """Digest JSON-compatible evidence deterministically."""

    return _canonical_digest(value)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise MetastudyContractError(f"{label} must be a lowercase sha256 digest")
    if any(character not in "0123456789abcdef" for character in value[7:]):
        raise MetastudyContractError(f"{label} must be a lowercase sha256 digest")
    return value


def _nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MetastudyContractError(f"{label} must be a finite non-negative number")
    result = float(value)
    if result < 0.0:
        raise MetastudyContractError(f"{label} must be a finite non-negative number")
    return result


def _unique_text(values: tuple[str, ...], *, label: str, allow_empty: bool) -> None:
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise MetastudyContractError(f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} tuple")
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in values):
        raise MetastudyContractError(f"{label} must contain non-empty trimmed strings")
    if len(values) != len(set(values)):
        raise MetastudyContractError(f"{label} must not contain duplicates")


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MetastudyContractError(f"{label} must be non-empty trimmed text")
    return value


__all__ = ["MetastudyContractError", "canonical_digest"]
