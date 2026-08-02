"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/sequence/distances.py

Fixed-point edit distances for reproducible junction design.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from functools import lru_cache
from itertools import chain

import numpy as np

from .alphabet import validate_dna

POSITION_WEIGHT_SCALE = 1_000_000_000
_MAX_SIGNED_64 = (1 << 63) - 1
_MAX_LEVENSHTEIN_SCRATCH_BYTES = 64 * 1024 * 1024
_MAX_POSITION_WEIGHTED_SCRATCH_BYTES = 64 * 1024 * 1024
_MAX_POSITION_WEIGHT_CACHE_CONSTRUCTION_BYTES = 4 * 1024 * 1024
_POSITION_WEIGHT_CACHE_MAXSIZE = 16
# Covers the retained compact forward scores plus per-cell NumPy temporaries.
_POSITION_WEIGHTED_TEMPORARY_VECTORS = 10
# Conservatively covers each Python integer, an overallocated list slot, and a
# tuple slot while both containers coexist during conversion on a 64-bit runtime.
_POSITION_WEIGHT_CACHE_BYTES_PER_UNIT = 32 + 16 + 8
_POSITION_WEIGHT_CACHE_CONTAINER_BYTES = 2 * 64
_MAX_CACHED_POSITION_WEIGHT_COUNT = (
    _MAX_POSITION_WEIGHT_CACHE_CONSTRUCTION_BYTES - _POSITION_WEIGHT_CACHE_CONTAINER_BYTES
) // _POSITION_WEIGHT_CACHE_BYTES_PER_UNIT


def _bounded_pair_chunk_size(
    *,
    fixed_bytes: int,
    per_pair_bytes: int,
    budget_bytes: int,
    error_message: str,
) -> int:
    """Return a deterministic pair batch size inside an explicit byte budget."""

    if fixed_bytes < 0 or per_pair_bytes < 1 or budget_bytes < 1:
        raise ValueError("pair-chunk sizing inputs are invalid")
    if fixed_bytes + per_pair_bytes > budget_bytes:
        raise ValueError(error_message)
    return (budget_bytes - fixed_bytes) // per_pair_bytes


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


def _levenshtein_scratch_bytes(pair_count: int, sequence_length: int) -> int:
    """Conservatively estimate inputs and transient unit-edit DP storage."""

    if pair_count < 0 or sequence_length < 0:
        raise ValueError("Levenshtein scratch inputs must be nonnegative")
    int32_bytes = np.dtype(np.int32).itemsize
    uint8_bytes = np.dtype(np.uint8).itemsize
    bool_bytes = np.dtype(np.bool_).itemsize
    fixed_bytes = (sequence_length + 1) * int32_bytes
    encoded_input_bytes = 2 * sequence_length * uint8_bytes
    row_matrix_bytes = 2 * (sequence_length + 1) * int32_bytes
    # Deletion, insertion, substitution, two nested-minimum temporaries, and
    # the compact returned score coexist at the inner-loop peak.
    temporary_bytes = 6 * int32_bytes + bool_bytes
    return fixed_bytes + pair_count * (encoded_input_bytes + row_matrix_bytes + temporary_bytes)


def _levenshtein_chunk_size(
    sequence_length: int,
    *,
    additional_per_pair_bytes: int = 0,
    budget_bytes: int = _MAX_LEVENSHTEIN_SCRATCH_BYTES,
) -> int:
    """Return the largest unit-edit pair batch inside the scratch budget."""

    fixed_bytes = (sequence_length + 1) * np.dtype(np.int32).itemsize
    one_pair_bytes = _levenshtein_scratch_bytes(1, sequence_length)
    return _bounded_pair_chunk_size(
        fixed_bytes=fixed_bytes,
        per_pair_bytes=one_pair_bytes - fixed_bytes + additional_per_pair_bytes,
        budget_bytes=budget_bytes,
        error_message="sequence length exceeds the Levenshtein scratch envelope",
    )


@lru_cache(maxsize=_POSITION_WEIGHT_CACHE_MAXSIZE)
def position_weight_units(length: int) -> tuple[int, ...]:
    """Return ``1 + exp(-u)`` weights quantized to stable nanounits."""

    _guard_position_weight_length(length)
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


def _guard_position_weight_length(length: int) -> None:
    """Reject unsupported lengths before sequence scans or array allocation."""

    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if 2 * length * POSITION_WEIGHT_SCALE > _MAX_SIGNED_64:
        raise ValueError("sequence length exceeds the fixed-point distance envelope")
    if length > _MAX_CACHED_POSITION_WEIGHT_COUNT:
        raise ValueError("sequence length exceeds the cached-weight construction envelope")


def _guard_position_weighted_strings(*collections: tuple[object, ...]) -> None:
    """Apply the cheap length envelope to strings before alphabet validation."""

    for collection in collections:
        for value in collection:
            if isinstance(value, str):
                _guard_position_weight_length(len(value))


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
    """Return junction v1's directional position-weighted edit distance.

    The pooled preprint specifies ``u = i / (t - 1)`` and
    ``w(u) = 1 + exp(-u)`` but leaves the insertion/deletion recurrence
    ambiguous. junction's explicit v1 policy weights substitutions and
    deletions at source position ``i - 1`` and insertions at target position
    ``j - 1``. Exact matches cost zero. This named primitive makes that policy
    inspectable.
    """

    _guard_position_weighted_strings((source, target))
    source = validate_dna(source, name="source")
    target = validate_dna(target, name="target")

    return _directional_position_weighted_levenshtein_units(source, target) / POSITION_WEIGHT_SCALE


def position_weighted_levenshtein_units(left: str, right: str) -> int:
    """Return the symmetric v1 score as stable fixed-point nanounits."""

    _guard_position_weighted_strings((left, right))
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
    return previous[:, -1].copy()


def _position_weighted_scratch_bytes(pair_count: int, sequence_length: int) -> int:
    """Estimate simultaneous cached-weight, NumPy, and directional scratch."""

    int64_bytes = np.dtype(np.int64).itemsize
    row_matrix_values = 2 * (sequence_length + 1)
    per_pair_values = row_matrix_values + _POSITION_WEIGHTED_TEMPORARY_VECTORS
    cumulative_values = sequence_length + 1
    weight_values = sequence_length
    encoded_input_bytes = pair_count * 2 * sequence_length * np.dtype(np.uint8).itemsize
    return (
        _position_weight_cache_construction_bytes(sequence_length)
        + int64_bytes * (pair_count * per_pair_values + cumulative_values + weight_values)
        + encoded_input_bytes
    )


def _position_weight_cache_construction_bytes(sequence_length: int) -> int:
    """Conservatively estimate peak Python bytes while caching one weight tuple."""

    return _POSITION_WEIGHT_CACHE_CONTAINER_BYTES + sequence_length * _POSITION_WEIGHT_CACHE_BYTES_PER_UNIT


def _position_weighted_chunk_size(
    sequence_length: int,
    *,
    additional_per_pair_bytes: int = 0,
    budget_bytes: int = _MAX_POSITION_WEIGHTED_SCRATCH_BYTES,
) -> int:
    """Return the largest deterministic pair chunk inside the scratch budget."""

    _guard_position_weight_length(sequence_length)
    one_pair_bytes = _position_weighted_scratch_bytes(1, sequence_length)
    fixed_bytes = _position_weighted_scratch_bytes(0, sequence_length)
    per_pair_bytes = one_pair_bytes - fixed_bytes + additional_per_pair_bytes
    return _bounded_pair_chunk_size(
        fixed_bytes=fixed_bytes,
        per_pair_bytes=per_pair_bytes,
        budget_bytes=budget_bytes,
        error_message="sequence length exceeds the weighted-distance scratch envelope",
    )


def position_weighted_levenshtein_units_many(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> np.ndarray:
    """Vectorize the exact fixed-point v1 score over equally sized DNA pairs."""

    if len(left) != len(right):
        raise ValueError("left and right pair collections must have equal size")
    if not left:
        return np.empty(0, dtype=np.uint64)
    _guard_position_weighted_strings(left, right)
    sequence_length = len(left[0])
    if any(len(value) != sequence_length for value in chain(left, right)):
        raise ValueError("batched weighted distances require one common sequence length")
    validated_left = tuple(validate_dna(value, name="left") for value in left)
    validated_right = tuple(validate_dna(value, name="right") for value in right)
    chunk_size = _position_weighted_chunk_size(sequence_length)
    result = np.empty(len(left), dtype=np.uint64)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        encoded_left = np.frombuffer(
            "".join(validated_left[start:stop]).encode("ascii"),
            dtype=np.uint8,
        ).reshape(stop - start, sequence_length)
        encoded_right = np.frombuffer(
            "".join(validated_right[start:stop]).encode("ascii"),
            dtype=np.uint8,
        ).reshape(stop - start, sequence_length)
        result[start:stop] = _position_weighted_levenshtein_units_encoded_many(encoded_left, encoded_right)
    return result


def _position_weighted_levenshtein_units_encoded_many(
    encoded_left: np.ndarray,
    encoded_right: np.ndarray,
) -> np.ndarray:
    """Evaluate already-validated, equal-shape encoded DNA pairs."""

    if encoded_left.shape != encoded_right.shape or encoded_left.ndim != 2:
        raise ValueError("encoded pair matrices must have one equal two-dimensional shape")
    pair_count, sequence_length = encoded_left.shape
    if pair_count == 0:
        return np.empty(0, dtype=np.uint64)
    chunk_size = _position_weighted_chunk_size(sequence_length)
    weights = np.asarray(position_weight_units(sequence_length), dtype=np.int64)
    result = np.empty(pair_count, dtype=np.uint64)
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        forward = _directional_units_many(encoded_left[start:stop], encoded_right[start:stop], weights)
        reverse = _directional_units_many(encoded_right[start:stop], encoded_left[start:stop], weights)
        np.minimum(forward, reverse, out=forward)
        result[start:stop] = forward
    return result


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
