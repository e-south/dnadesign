"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/acquisition_projection/_values.py

Scalar validation shared by acquisition-projection contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ..contracts._values import MetastudyContractError


def exact_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise MetastudyContractError(f"{label} fields do not match the exact contract")
    return dict(value)


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MetastudyContractError(f"{label} must be trimmed non-empty text")
    return value


def digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise MetastudyContractError(f"{label} must be a canonical SHA-256 digest")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise MetastudyContractError(f"{label} must be a canonical SHA-256 digest") from exc
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MetastudyContractError(f"{label} must be finite")
    return float(value)


def window(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise MetastudyContractError("acquisition selected_reduction must contain two values")
    start_h = finite(value[0], "selected_reduction start")
    end_h = finite(value[1], "selected_reduction end")
    if start_h < 0.0 or end_h <= start_h:
        raise MetastudyContractError("selected_reduction must be an ordered non-negative window")
    return (start_h, end_h)


__all__ = ["digest", "exact_object", "finite", "text", "window"]
