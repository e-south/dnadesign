"""Fixed-point edit distances for reproducible TriJunction design."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from functools import lru_cache

import numpy as np

from .alphabet import validate_dna

POSITION_WEIGHT_SCALE = 1_000_000_000
_MAX_SIGNED_64 = (1 << 63) - 1


def levenshtein_distance(left: str, right: str) -> int:
    """Return conventional Levenshtein distance with unit edit costs."""

    left = validate_dna(left, name="left")
    right = validate_dna(right, name="right")
    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for left_index, left_base in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_base in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_base != right_base),
                )
            )
        previous = current
    return previous[-1]


def _levenshtein_distance_encoded_many(encoded_left: np.ndarray, encoded_right: np.ndarray) -> np.ndarray:
    """Evaluate unit-cost Levenshtein over validated equal-shape encoded pairs."""

    if encoded_left.shape != encoded_right.shape or encoded_left.ndim != 2:
        raise ValueError("encoded pair matrices must have one equal two-dimensional shape")
    pair_count, sequence_length = encoded_left.shape
    previous = np.broadcast_to(
        np.arange(sequence_length + 1, dtype=np.int32),
        (pair_count, sequence_length + 1),
    ).copy()
    for left_index in range(1, sequence_length + 1):
        current = np.empty_like(previous)
        current[:, 0] = left_index
        for right_index in range(1, sequence_length + 1):
            deletion = previous[:, right_index] + 1
            insertion = current[:, right_index - 1] + 1
            substitution = previous[:, right_index - 1] + (
                encoded_left[:, left_index - 1] != encoded_right[:, right_index - 1]
            )
            current[:, right_index] = np.minimum(np.minimum(deletion, insertion), substitution)
        previous = current
    return previous[:, -1]


@lru_cache(maxsize=128)
def position_weight_units(length: int) -> tuple[int, ...]:
    """Return ``1 + exp(-u)`` weights quantized to stable nanounits."""

    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if 2 * length * POSITION_WEIGHT_SCALE > _MAX_SIGNED_64:
        raise ValueError("sequence length exceeds the fixed-point distance envelope")
    if length == 0:
        return ()
    with localcontext() as context:
        context.prec = 50
        scale = Decimal(POSITION_WEIGHT_SCALE)
        denominator = Decimal(max(length - 1, 1))
        weights = []
        for index in range(length):
            normalized = Decimal(0) if length <= 1 else Decimal(index) / denominator
            weight = (Decimal(1) + (-normalized).exp()) * scale
            weights.append(int(weight.to_integral_value(rounding=ROUND_HALF_EVEN)))
    return tuple(weights)


def _directional_position_weighted_levenshtein_units(source: str, target: str) -> int:
    source_weights = position_weight_units(len(source))
    target_weights = position_weight_units(len(target))

    previous = [0]
    for target_weight in target_weights:
        previous.append(previous[-1] + target_weight)

    for source_index, source_base in enumerate(source, start=1):
        source_weight = source_weights[source_index - 1]
        current = [previous[0] + source_weight]
        for target_index, target_base in enumerate(target, start=1):
            substitution_cost = 0 if source_base == target_base else source_weight
            current.append(
                min(
                    previous[target_index] + source_weight,
                    current[target_index - 1] + target_weights[target_index - 1],
                    previous[target_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def directional_position_weighted_levenshtein(source: str, target: str) -> float:
    """Return TriJunction v1's directional position-weighted edit distance.

    The source paper specifies ``w(u) = 1 + exp(-u)`` with normalized position
    ``u`` but leaves the insertion/deletion recurrence ambiguous. TriJunction's
    explicit v1 policy weights substitutions and deletions at source position
    ``i - 1`` and insertions at target position ``j - 1``. Exact matches cost
    zero. This named primitive makes that pragmatic policy inspectable.
    """

    source = validate_dna(source, name="source")
    target = validate_dna(target, name="target")

    return _directional_position_weighted_levenshtein_units(source, target) / POSITION_WEIGHT_SCALE


def position_weighted_levenshtein_units(left: str, right: str) -> int:
    """Return the symmetric v1 score as stable fixed-point nanounits."""

    left = validate_dna(left, name="left")
    right = validate_dna(right, name="right")
    return min(
        _directional_position_weighted_levenshtein_units(left, right),
        _directional_position_weighted_levenshtein_units(right, left),
    )


def _directional_units_many(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pair_count, sequence_length = source.shape
    cumulative = np.empty(sequence_length + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(weights, out=cumulative[1:])
    previous = np.broadcast_to(cumulative, (pair_count, sequence_length + 1)).copy()

    source_prefix = 0
    for source_index in range(sequence_length):
        source_weight = weights[source_index]
        source_prefix += int(source_weight)
        current = np.empty_like(previous)
        current[:, 0] = source_prefix
        for target_index in range(1, sequence_length + 1):
            deletion = previous[:, target_index] + source_weight
            insertion = current[:, target_index - 1] + weights[target_index - 1]
            substitution = previous[:, target_index - 1] + np.where(
                source[:, source_index] == target[:, target_index - 1],
                0,
                source_weight,
            )
            current[:, target_index] = np.minimum(np.minimum(deletion, insertion), substitution)
        previous = current
    return previous[:, -1]


def position_weighted_levenshtein_units_many(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> np.ndarray:
    """Vectorize the exact fixed-point v1 score over equally sized DNA pairs."""

    if len(left) != len(right):
        raise ValueError("left and right pair collections must have equal size")
    if not left:
        return np.empty(0, dtype=np.uint64)
    sequence_length = len(left[0])
    if any(len(value) != sequence_length for value in (*left, *right)):
        raise ValueError("batched weighted distances require one common sequence length")
    validated_left = tuple(validate_dna(value, name="left") for value in left)
    validated_right = tuple(validate_dna(value, name="right") for value in right)
    encoded_left = np.frombuffer("".join(validated_left).encode("ascii"), dtype=np.uint8).reshape(
        len(left), sequence_length
    )
    encoded_right = np.frombuffer("".join(validated_right).encode("ascii"), dtype=np.uint8).reshape(
        len(right), sequence_length
    )
    return _position_weighted_levenshtein_units_encoded_many(encoded_left, encoded_right)


def _position_weighted_levenshtein_units_encoded_many(
    encoded_left: np.ndarray,
    encoded_right: np.ndarray,
) -> np.ndarray:
    """Evaluate already-validated, equal-shape encoded DNA pairs."""

    if encoded_left.shape != encoded_right.shape or encoded_left.ndim != 2:
        raise ValueError("encoded pair matrices must have one equal two-dimensional shape")
    sequence_length = encoded_left.shape[1]
    weights = np.asarray(position_weight_units(sequence_length), dtype=np.int64)
    forward = _directional_units_many(encoded_left, encoded_right, weights)
    reverse = _directional_units_many(encoded_right, encoded_left, weights)
    return np.minimum(forward, reverse).astype(np.uint64, copy=False)


def position_weighted_levenshtein(left: str, right: str) -> float:
    """Return the paper-inspired score symmetrized by minimum direction."""

    return position_weighted_levenshtein_units(left, right) / POSITION_WEIGHT_SCALE


__all__ = [
    "directional_position_weighted_levenshtein",
    "levenshtein_distance",
    "POSITION_WEIGHT_SCALE",
    "position_weighted_levenshtein",
    "position_weighted_levenshtein_units",
    "position_weighted_levenshtein_units_many",
    "position_weight_units",
]
