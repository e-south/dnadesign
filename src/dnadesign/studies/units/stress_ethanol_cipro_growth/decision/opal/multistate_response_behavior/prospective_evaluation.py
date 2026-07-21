"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/prospective_evaluation.py

Fixed calculations for prospective evaluation of the first MSRB learning probe.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Y8_COORDINATE_ORDER = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")


@dataclass(frozen=True)
class RawY8ErrorSummary:
    """Equal-weight prediction errors for the fixed response-window phenotype."""

    coordinate_order: tuple[str, ...]
    coordinate_mae: tuple[float, ...]
    coordinate_rmse: tuple[float, ...]
    pooled_mae: float
    pooled_rmse: float


def raw_y8_error_summary(
    predicted: Sequence[Sequence[float]],
    observed: Sequence[Sequence[float]],
) -> RawY8ErrorSummary:
    """Compare paired Y8 rows with equal weight per candidate and coordinate."""

    predicted_array = _y8_matrix(predicted, label="predicted")
    observed_array = _y8_matrix(observed, label="observed")
    if predicted_array.shape != observed_array.shape:
        raise ValueError(
            "predicted and observed Y8 matrices must have the same shape; "
            f"observed {predicted_array.shape} and {observed_array.shape}."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        residual = predicted_array - observed_array
        absolute_error = np.abs(residual)
        squared_error = residual * residual
    if not np.isfinite(absolute_error).all() or not np.isfinite(squared_error).all():
        raise ValueError("Y8 errors must remain finite.")

    coordinate_mae = np.mean(absolute_error, axis=0)
    coordinate_rmse = np.sqrt(np.mean(squared_error, axis=0))
    return RawY8ErrorSummary(
        coordinate_order=Y8_COORDINATE_ORDER,
        coordinate_mae=tuple(float(value) for value in coordinate_mae),
        coordinate_rmse=tuple(float(value) for value in coordinate_rmse),
        pooled_mae=float(np.mean(absolute_error)),
        pooled_rmse=float(np.sqrt(np.mean(squared_error))),
    )


def spearman_average_rank(
    predicted: Sequence[float],
    observed: Sequence[float],
) -> float | None:
    """Return tie-aware Spearman correlation, or ``None`` for a constant input."""

    predicted_array = _finite_vector(predicted, label="predicted ranks", minimum_size=2)
    observed_array = _finite_vector(observed, label="observed ranks", minimum_size=2)
    if predicted_array.shape != observed_array.shape:
        raise ValueError(
            "predicted and observed rank vectors must have the same shape; "
            f"observed {predicted_array.shape} and {observed_array.shape}."
        )
    predicted_ranks = _average_ranks(predicted_array)
    observed_ranks = _average_ranks(observed_array)
    predicted_centered = predicted_ranks - np.mean(predicted_ranks)
    observed_centered = observed_ranks - np.mean(observed_ranks)
    denominator = float(
        np.sqrt(np.dot(predicted_centered, predicted_centered) * np.dot(observed_centered, observed_centered))
    )
    if denominator == 0.0:
        return None
    return float(np.dot(predicted_centered, observed_centered) / denominator)


def midpoint_median(values: Sequence[float]) -> float:
    """Return the middle value, or the midpoint of the two middle values."""

    array = np.sort(_finite_vector(values, label="median values", minimum_size=1))
    middle = array.size // 2
    if array.size % 2:
        return float(array[middle])
    return float(array[middle - 1] / 2.0 + array[middle] / 2.0)


def midrank_percentile(observed: float, reference: Sequence[float]) -> float:
    """Locate an observed value in an exhaustive reference set using midranks."""

    observed_array = _finite_vector([observed], label="observed value", minimum_size=1)
    reference_array = _finite_vector(reference, label="reference values", minimum_size=1)
    observed_value = observed_array[0]
    below = int(np.count_nonzero(reference_array < observed_value))
    tied = int(np.count_nonzero(reference_array == observed_value))
    return float(100.0 * (below + 0.5 * tied) / reference_array.size)


def _y8_matrix(values: Sequence[Sequence[float]], *, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} Y8 values must form a numeric matrix.") from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != len(Y8_COORDINATE_ORDER):
        raise ValueError(
            f"{label} Y8 values must have shape (candidate, {len(Y8_COORDINATE_ORDER)}); observed {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} Y8 values must be finite.")
    return array


def _finite_vector(values: Sequence[float], *, label: str, minimum_size: int) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if array.ndim != 1 or array.size < minimum_size:
        raise ValueError(f"{label} must be a one-dimensional vector with at least {minimum_size} value(s).")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values.")
    return array


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


__all__ = [
    "RawY8ErrorSummary",
    "Y8_COORDINATE_ORDER",
    "midpoint_median",
    "midrank_percentile",
    "raw_y8_error_summary",
    "spearman_average_rank",
]
