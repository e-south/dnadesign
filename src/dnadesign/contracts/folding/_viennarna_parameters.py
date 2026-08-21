"""Fail-closed validation for the ViennaRNA parameters DNA Design implements."""

from __future__ import annotations

import math
from typing import Any


def validate_viennarna_parameters(value: object) -> dict[str, Any]:
    """Accept only parameters that both supported ViennaRNA interfaces apply."""
    if not isinstance(value, dict):
        raise ValueError("ViennaRNA parameters must be a mapping.")
    unknown = sorted(set(value) - {"temperature_c"})
    if unknown:
        raise ValueError(f"Unsupported ViennaRNA parameters: {', '.join(unknown)}.")
    if "temperature_c" not in value:
        return {}
    temperature = value["temperature_c"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("ViennaRNA temperature_c must be a finite positive number.")
    normalized = float(temperature)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("ViennaRNA temperature_c must be a finite positive number.")
    return {"temperature_c": normalized}


__all__ = ["validate_viennarna_parameters"]
