"""Public Response-Magnitude Feasibility mathematics API."""

from __future__ import annotations

from ..src.objectives.response_magnitude_feasibility_math import (
    ResponseMagnitudeFeasibilityComponents,
    ResponseMagnitudeFeasibilityScore,
    binary_target_mask,
    calibrate_response_magnitude_feasibility,
    response_magnitude_feasibility_components,
    score_response_magnitude_feasibility,
    validated_response_magnitude,
)

RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION = "1"

__all__ = [
    "RESPONSE_MAGNITUDE_FEASIBILITY_API_VERSION",
    "ResponseMagnitudeFeasibilityComponents",
    "ResponseMagnitudeFeasibilityScore",
    "binary_target_mask",
    "calibrate_response_magnitude_feasibility",
    "response_magnitude_feasibility_components",
    "score_response_magnitude_feasibility",
    "validated_response_magnitude",
]
