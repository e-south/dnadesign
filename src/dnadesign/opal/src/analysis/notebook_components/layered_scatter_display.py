"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/layered_scatter_display.py

Defines invariant and shared display geometry for round-overlay scatters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def invariant_round_display(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Return display semantics that must agree across model rounds."""

    contract = {key: value for key, value in runtime.items() if key not in {"x_limits", "y_limits"}}
    raw_color_scale = contract.get("color_scale")
    color_scale = dict(raw_color_scale) if isinstance(raw_color_scale, Mapping) else {}
    for field in ("context", "extend", "extent"):
        color_scale.pop(field, None)
    contract["color_scale"] = color_scale
    return contract


def runtime_limits(runtime: Mapping[str, Any], field: str) -> tuple[float, float]:
    """Load one finite increasing axis interval."""

    value = runtime[field]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Layered-scatter {field} must contain two values.")
    lower, upper = (float(item) for item in value)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError(f"Layered-scatter {field} must be finite and increasing.")
    return lower, upper


def shared_colorbar_extend(*, minimum: float, maximum: float, center: float, extent: float) -> str:
    """Describe which endpoints clip values across the loaded rounds."""

    below = minimum < center - extent
    above = maximum > center + extent
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


__all__ = ["invariant_round_display", "runtime_limits", "shared_colorbar_extend"]
