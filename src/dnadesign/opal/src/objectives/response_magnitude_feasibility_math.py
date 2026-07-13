"""Pure mathematics for Response-Magnitude Feasibility (RMF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

OBJECTIVE_NAME = "response_magnitude_feasibility_v1"
CALIBRATION_FIELDS = (
    "response_separation_min",
    "on_magnitude_min",
    "off_magnitude_max",
    "response_separation_scale",
    "on_magnitude_scale",
    "off_magnitude_scale",
)
SCALE_FIELDS = (
    "response_separation_scale",
    "on_magnitude_scale",
    "off_magnitude_scale",
)


@dataclass(frozen=True)
class ResponseMagnitudeFeasibilityComponents:
    """Worst-state response and reference-relative magnitude components."""

    response_separation: np.ndarray
    on_magnitude_floor: np.ndarray
    off_magnitude_ceiling: np.ndarray


@dataclass(frozen=True)
class ResponseMagnitudeFeasibilityScore:
    """Calibrated constraint margins and their maximin score."""

    components: ResponseMagnitudeFeasibilityComponents
    response_constraint_margin: np.ndarray
    on_magnitude_constraint_margin: np.ndarray
    off_magnitude_constraint_margin: np.ndarray
    feasibility_margin: np.ndarray
    calibration: dict[str, float]


def response_magnitude_feasibility_components(
    response_magnitude: np.ndarray,
    *,
    target_mask: Sequence[int | float],
) -> ResponseMagnitudeFeasibilityComponents:
    """Calculate RMF components for any finite binary state partition."""

    target_on = binary_target_mask(target_mask)
    values = validated_response_magnitude(response_magnitude, state_count=target_on.size)
    target_off = ~target_on
    response = values[:, : target_on.size]
    magnitude = values[:, target_on.size :]
    return ResponseMagnitudeFeasibilityComponents(
        response_separation=np.min(response[:, target_on], axis=1) - np.max(response[:, target_off], axis=1),
        on_magnitude_floor=np.min(magnitude[:, target_on], axis=1),
        off_magnitude_ceiling=np.max(magnitude[:, target_off], axis=1),
    )


def score_response_magnitude_feasibility(
    response_magnitude: np.ndarray,
    *,
    target_mask: Sequence[int | float],
    calibration: Mapping[str, object],
) -> ResponseMagnitudeFeasibilityScore:
    """Score response/magnitude summaries with explicit calibrated margins."""

    components = response_magnitude_feasibility_components(response_magnitude, target_mask=target_mask)
    return calibrate_response_magnitude_feasibility(components, calibration=calibration)


def calibrate_response_magnitude_feasibility(
    components: ResponseMagnitudeFeasibilityComponents,
    *,
    calibration: Mapping[str, object],
) -> ResponseMagnitudeFeasibilityScore:
    """Apply explicit thresholds and scales to precomputed RMF components."""

    if not isinstance(components, ResponseMagnitudeFeasibilityComponents):
        raise TypeError("components must be ResponseMagnitudeFeasibilityComponents.")
    parsed = parse_calibration(calibration)
    response_values = _component_array(components.response_separation, name="response_separation")
    on_values = _component_array(components.on_magnitude_floor, name="on_magnitude_floor")
    off_values = _component_array(components.off_magnitude_ceiling, name="off_magnitude_ceiling")
    lengths = {len(response_values), len(on_values), len(off_values)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("RMF component arrays must be aligned and non-empty.")
    if not all(np.all(np.isfinite(values)) for values in (response_values, on_values, off_values)):
        raise ValueError("RMF component arrays must be finite.")

    response_constraint = (response_values - parsed["response_separation_min"]) / parsed["response_separation_scale"]
    on_constraint = (on_values - parsed["on_magnitude_min"]) / parsed["on_magnitude_scale"]
    off_constraint = (parsed["off_magnitude_max"] - off_values) / parsed["off_magnitude_scale"]
    feasibility = np.minimum.reduce((response_constraint, on_constraint, off_constraint))
    normalized = ResponseMagnitudeFeasibilityComponents(
        response_separation=response_values,
        on_magnitude_floor=on_values,
        off_magnitude_ceiling=off_values,
    )
    return ResponseMagnitudeFeasibilityScore(
        components=normalized,
        response_constraint_margin=np.asarray(response_constraint, dtype=float),
        on_magnitude_constraint_margin=np.asarray(on_constraint, dtype=float),
        off_magnitude_constraint_margin=np.asarray(off_constraint, dtype=float),
        feasibility_margin=np.asarray(feasibility, dtype=float),
        calibration=parsed,
    )


def validated_response_magnitude(values: np.ndarray, *, state_count: int) -> np.ndarray:
    """Validate ordered response values followed by aligned magnitude values."""

    if not isinstance(state_count, int) or isinstance(state_count, bool) or state_count < 2:
        raise ValueError(f"{OBJECTIVE_NAME}: state_count must be an integer >= 2.")
    matrix = np.asarray(values, dtype=float)
    expected_columns = 2 * state_count
    if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
        raise ValueError(
            f"{OBJECTIVE_NAME}: input must have exact shape (n, {expected_columns}) with "
            f"{state_count} response columns followed by {state_count} aligned magnitude columns; "
            f"got {getattr(matrix, 'shape', None)}."
        )
    if matrix.shape[0] == 0:
        raise ValueError(f"{OBJECTIVE_NAME}: input must contain at least one candidate row.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{OBJECTIVE_NAME}: input must be finite.")
    return matrix


def _component_array(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"RMF component {name!r} must be one-dimensional; got {array.shape}.")
    return array


def binary_target_mask(target_mask: Sequence[int | float]) -> np.ndarray:
    """Validate a variable-length binary target with ON and OFF support."""

    values = np.asarray(target_mask, dtype=float).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{OBJECTIVE_NAME}: target_mask must contain at least two finite entries.")
    if not np.all(np.isin(values, (0.0, 1.0))):
        raise ValueError(f"{OBJECTIVE_NAME}: target_mask must be binary; got {values.tolist()}.")
    on_count = int(np.sum(values))
    if on_count <= 0 or on_count >= values.size:
        raise ValueError(
            f"{OBJECTIVE_NAME}: target_mask must contain at least one ON and one OFF state; "
            f"got {values.astype(int).tolist()}."
        )
    return values.astype(bool)


def parse_calibration(raw: Mapping[str, object]) -> dict[str, float]:
    """Validate exact thresholds and positive scales."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{OBJECTIVE_NAME}: calibration must be an explicit mapping.")
    missing = sorted(set(CALIBRATION_FIELDS) - set(raw))
    extra = sorted(set(raw) - set(CALIBRATION_FIELDS))
    if missing or extra:
        raise ValueError(
            f"{OBJECTIVE_NAME}: calibration keys do not match the contract; missing={missing}, extra={extra}."
        )
    calibration = {name: float(raw[name]) for name in CALIBRATION_FIELDS}
    nonfinite = [name for name, value in calibration.items() if not np.isfinite(value)]
    if nonfinite:
        raise ValueError(f"{OBJECTIVE_NAME}: calibration values must be finite; invalid={nonfinite}.")
    nonpositive = [name for name in SCALE_FIELDS if calibration[name] <= 0.0]
    if nonpositive:
        raise ValueError(f"{OBJECTIVE_NAME}: calibration scales must be positive; invalid={nonpositive}.")
    return calibration


__all__ = [
    "CALIBRATION_FIELDS",
    "OBJECTIVE_NAME",
    "ResponseMagnitudeFeasibilityComponents",
    "ResponseMagnitudeFeasibilityScore",
    "SCALE_FIELDS",
    "binary_target_mask",
    "calibrate_response_magnitude_feasibility",
    "parse_calibration",
    "response_magnitude_feasibility_components",
    "score_response_magnitude_feasibility",
    "validated_response_magnitude",
]
