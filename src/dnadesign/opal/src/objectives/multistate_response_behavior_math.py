"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/objectives/multistate_response_behavior_math.py

Pure mathematics for the threshold-free Multistate Response Behavior objective.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

OBJECTIVE_NAME = "multistate_response_behavior_v1"
NORMALIZATION_FIELDS = ("response_scale", "signal_scale")
NORMALIZED_TEMPERATURE = 1.0
_CLEARANCE_LIMIT = np.finfo(float).max / 64.0


@dataclass(frozen=True)
class MultistateResponseBehaviorClearances:
    """State-level clearances grouped by the three behavior families."""

    response: np.ndarray
    response_labels: tuple[str, ...]
    on_signal: np.ndarray
    on_signal_labels: tuple[str, ...]
    off_signal_suppression: np.ndarray
    off_signal_suppression_labels: tuple[str, ...]
    state_ids: tuple[str, ...]
    target_mask: tuple[int, ...]
    normalization: dict[str, float]

    @property
    def coordinate_clearances(self) -> np.ndarray:
        """Return all clearances in response, ON, then OFF family order."""

        return np.concatenate((self.response, self.on_signal, self.off_signal_suppression), axis=1)

    @property
    def coordinate_labels(self) -> tuple[str, ...]:
        """Return stable labels aligned with ``coordinate_clearances``."""

        return self.response_labels + self.on_signal_labels + self.off_signal_suppression_labels


@dataclass(frozen=True)
class MultistateResponseBehaviorScore:
    """Family-balanced behavior score and complete state-level diagnostics."""

    clearances: MultistateResponseBehaviorClearances
    behavior_score: np.ndarray
    hard_bottleneck_clearance: np.ndarray
    response_family_score: np.ndarray
    on_signal_family_score: np.ndarray
    off_signal_suppression_family_score: np.ndarray
    coordinate_prior_weights: np.ndarray
    coordinate_weights: np.ndarray
    limiting_coordinate_index: np.ndarray
    limiting_coordinate_label: tuple[str, ...]
    compensation_gap: np.ndarray
    maximum_compensation_gap: np.ndarray
    all_reference_directions_met: np.ndarray
    normalization: dict[str, float]

    @property
    def coordinate_clearances(self) -> np.ndarray:
        """Return all normalized state-level clearances."""

        return self.clearances.coordinate_clearances

    @property
    def coordinate_labels(self) -> tuple[str, ...]:
        """Return stable labels aligned with clearances and weights."""

        return self.clearances.coordinate_labels


def multistate_response_behavior_clearances(
    response_signal: np.ndarray,
    *,
    state_ids: Sequence[str],
    target_mask: Sequence[int | float],
    normalization: Mapping[str, object],
) -> MultistateResponseBehaviorClearances:
    """Build every normalized response, ON-signal, and OFF-signal-suppression clearance."""

    states = validated_state_ids(state_ids)
    target_on = binary_target_mask(target_mask)
    if target_on.size != len(states):
        raise ValueError(
            f"{OBJECTIVE_NAME}: state_ids and target_mask must have equal length; "
            f"got {len(states)} and {target_on.size}."
        )
    values = validated_response_signal(response_signal, state_count=len(states))
    scales = parse_normalization(normalization)
    target_off = ~target_on
    on_indices = np.flatnonzero(target_on)
    off_indices = np.flatnonzero(target_off)
    response = values[:, : len(states)]
    signal = values[:, len(states) :]

    response_pairs = _safe_scaled_difference(
        response[:, on_indices, None],
        response[:, None, off_indices],
        scale=scales["response_scale"],
    ).reshape(len(values), -1)
    on_signal = _safe_scaled(
        signal[:, on_indices],
        scale=scales["signal_scale"],
    )
    off_signal_suppression = _safe_scaled(
        -signal[:, off_indices],
        scale=scales["signal_scale"],
    )

    response_labels = tuple(
        f"response:{states[on_index]}>{states[off_index]}" for on_index in on_indices for off_index in off_indices
    )
    on_labels = tuple(f"on_signal:{states[index]}" for index in on_indices)
    off_labels = tuple(f"off_signal_suppression:{states[index]}" for index in off_indices)
    return MultistateResponseBehaviorClearances(
        response=response_pairs,
        response_labels=response_labels,
        on_signal=on_signal,
        on_signal_labels=on_labels,
        off_signal_suppression=off_signal_suppression,
        off_signal_suppression_labels=off_labels,
        state_ids=states,
        target_mask=tuple(int(value) for value in target_on),
        normalization=scales,
    )


def score_multistate_response_behavior(
    response_signal: np.ndarray,
    *,
    state_ids: Sequence[str],
    target_mask: Sequence[int | float],
    normalization: Mapping[str, object],
) -> MultistateResponseBehaviorScore:
    """Score a finite ``[r(state...), b(state...)]`` matrix without thresholds."""

    clearances = multistate_response_behavior_clearances(
        response_signal,
        state_ids=state_ids,
        target_mask=target_mask,
        normalization=normalization,
    )
    response_score = _smooth_bottleneck(clearances.response)
    on_score = _smooth_bottleneck(clearances.on_signal)
    off_score = _smooth_bottleneck(clearances.off_signal_suppression)
    coordinate_clearances = clearances.coordinate_clearances
    coordinate_prior = np.concatenate(
        (
            np.full(clearances.response.shape[1], 1.0 / (3.0 * clearances.response.shape[1])),
            np.full(clearances.on_signal.shape[1], 1.0 / (3.0 * clearances.on_signal.shape[1])),
            np.full(
                clearances.off_signal_suppression.shape[1],
                1.0 / (3.0 * clearances.off_signal_suppression.shape[1]),
            ),
        )
    )
    behavior_score, coordinate_weights = _weighted_smooth_bottleneck(
        coordinate_clearances,
        prior_weights=coordinate_prior,
    )
    limiting_index = np.argmin(coordinate_clearances, axis=1).astype(int)
    hard_bottleneck = np.min(coordinate_clearances, axis=1)
    compensation_gap = behavior_score - hard_bottleneck
    maximum_compensation_gap = -np.log(coordinate_prior[limiting_index])
    labels = clearances.coordinate_labels
    return MultistateResponseBehaviorScore(
        clearances=clearances,
        behavior_score=behavior_score,
        hard_bottleneck_clearance=hard_bottleneck,
        response_family_score=response_score,
        on_signal_family_score=on_score,
        off_signal_suppression_family_score=off_score,
        coordinate_prior_weights=coordinate_prior,
        coordinate_weights=coordinate_weights,
        limiting_coordinate_index=limiting_index,
        limiting_coordinate_label=tuple(labels[index] for index in limiting_index),
        compensation_gap=compensation_gap,
        maximum_compensation_gap=maximum_compensation_gap,
        all_reference_directions_met=np.all(coordinate_clearances >= 0.0, axis=1),
        normalization=dict(clearances.normalization),
    )


def validated_state_ids(raw: Sequence[str]) -> tuple[str, ...]:
    """Validate an ordered state identity contract."""

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) < 2:
        raise ValueError(f"{OBJECTIVE_NAME}: state_ids must contain at least two ordered strings.")
    if any(not isinstance(value, str) for value in raw):
        raise ValueError(f"{OBJECTIVE_NAME}: state_ids must contain strings, not coerced identifiers.")
    values = tuple(raw)
    if any(value != value.strip() for value in values):
        raise ValueError(f"{OBJECTIVE_NAME}: state_ids must not contain leading or trailing whitespace.")
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{OBJECTIVE_NAME}: state_ids must be non-empty and unique; got {list(values)}.")
    return values


def binary_target_mask(target_mask: Sequence[int | float]) -> np.ndarray:
    """Validate a variable-length binary target with ON and OFF support."""

    try:
        raw_values = tuple(target_mask)
    except TypeError as exc:
        raise ValueError(f"{OBJECTIVE_NAME}: target_mask must be one-dimensional.") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise ValueError(f"{OBJECTIVE_NAME}: target_mask must use numeric zero or one, not boolean aliases.")
    values = np.asarray(raw_values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{OBJECTIVE_NAME}: target_mask must be one-dimensional.")
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


def validated_response_signal(values: np.ndarray, *, state_count: int) -> np.ndarray:
    """Validate ordered response values followed by aligned signal values."""

    if not isinstance(state_count, int) or isinstance(state_count, bool) or state_count < 2:
        raise ValueError(f"{OBJECTIVE_NAME}: state_count must be an integer >= 2.")
    matrix = np.asarray(values, dtype=float)
    expected_columns = 2 * state_count
    if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
        raise ValueError(
            f"{OBJECTIVE_NAME}: input must have exact shape (n, {expected_columns}) with "
            f"{state_count} response columns followed by {state_count} aligned signal columns; "
            f"got {getattr(matrix, 'shape', None)}."
        )
    if matrix.shape[0] == 0:
        raise ValueError(f"{OBJECTIVE_NAME}: input must contain at least one candidate row.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{OBJECTIVE_NAME}: input must be finite.")
    return matrix


def parse_normalization(raw: Mapping[str, object]) -> dict[str, float]:
    """Validate the exact two-scale assay-resolution normalization contract."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{OBJECTIVE_NAME}: normalization must be an explicit mapping.")
    missing = sorted(set(NORMALIZATION_FIELDS) - set(raw))
    extra = sorted(set(raw) - set(NORMALIZATION_FIELDS))
    if missing or extra:
        raise ValueError(
            f"{OBJECTIVE_NAME}: normalization keys do not match the contract; missing={missing}, extra={extra}."
        )
    if any(isinstance(raw[name], (bool, np.bool_)) for name in NORMALIZATION_FIELDS):
        raise ValueError(f"{OBJECTIVE_NAME}: normalization values must be numeric, not boolean.")
    try:
        normalization = {name: float(raw[name]) for name in NORMALIZATION_FIELDS}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{OBJECTIVE_NAME}: normalization values must be numeric.") from exc
    invalid = [name for name, value in normalization.items() if not np.isfinite(value) or value <= 0.0]
    if invalid:
        raise ValueError(f"{OBJECTIVE_NAME}: normalization scales must be positive and finite; invalid={invalid}.")
    return normalization


def _safe_scaled(values: np.ndarray, *, scale: float) -> np.ndarray:
    """Scale finite values and saturate only arithmetic overflow at a finite bound."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = np.asarray(values, dtype=float) / float(scale)
    return np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=_CLEARANCE_LIMIT,
        neginf=-_CLEARANCE_LIMIT,
    )


def _safe_scaled_difference(left: np.ndarray, right: np.ndarray, *, scale: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return _safe_scaled(difference, scale=scale)


def _smooth_bottleneck(clearances: np.ndarray) -> np.ndarray:
    count = clearances.shape[1]
    score, _weights = _weighted_smooth_bottleneck(
        clearances,
        prior_weights=np.full(count, 1.0 / count),
    )
    return score


def _weighted_smooth_bottleneck(
    clearances: np.ndarray,
    *,
    prior_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(clearances, dtype=float)
    prior = np.asarray(prior_weights, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != prior.size or prior.size == 0:
        raise ValueError("smooth bottleneck values and prior weights must be non-empty and aligned.")
    if np.any(prior <= 0.0) or not np.isclose(float(np.sum(prior)), 1.0):
        raise ValueError("smooth bottleneck prior weights must be positive and sum to one.")
    log_terms = -(values / NORMALIZED_TEMPERATURE) + np.log(prior)[None, :]
    row_max = np.max(log_terms, axis=1, keepdims=True)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shifted = np.exp(log_terms - row_max)
    normalizer = np.sum(shifted, axis=1, keepdims=True)
    weights = shifted / normalizer
    score = -NORMALIZED_TEMPERATURE * (row_max[:, 0] + np.log(normalizer[:, 0]))
    return np.asarray(score, dtype=float), np.asarray(weights, dtype=float)


__all__ = [
    "NORMALIZATION_FIELDS",
    "NORMALIZED_TEMPERATURE",
    "OBJECTIVE_NAME",
    "MultistateResponseBehaviorClearances",
    "MultistateResponseBehaviorScore",
    "binary_target_mask",
    "multistate_response_behavior_clearances",
    "parse_normalization",
    "score_multistate_response_behavior",
    "validated_response_signal",
    "validated_state_ids",
]
