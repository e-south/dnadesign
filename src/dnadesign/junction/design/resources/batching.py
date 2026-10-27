"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/design/resources/batching.py

Bounded pair-batch primitives shared by junction search stages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

# Covers the two generated pair-index buffers plus the masks, positions,
# triangular-cache indices, unique/sort work, and cache-value vectors retained
# while one distance batch is evaluated. Values use uint64 so the estimate is
# conservative for both the uint16 barcode cache and uint64 toehold cache.
PAIR_LOOKUP_SCRATCH_BYTES_PER_PAIR = (
    18 * np.dtype(np.int64).itemsize + 4 * np.dtype(np.uint64).itemsize + 3 * np.dtype(np.bool_).itemsize
)
_NDARRAY_HEADER_BYTES = 128

# Covers the second-pass caller peak explicitly: both pair-index arrays, the
# returned distances, signed deltas, NumPy unique's flattened/sort/permutation/
# mask/inverse work, unique lookup values, and expanded contributions. One
# byte-per-pair boolean mask is added to sixteen conservative integer buffers.
PAIR_SELECTION_REDUCTION_BYTES_PER_PAIR = 16 * np.dtype(np.int64).itemsize + np.dtype(np.bool_).itemsize
_PAIR_SELECTION_REDUCTION_ARRAY_HEADERS = 16 * _NDARRAY_HEADER_BYTES


def pair_lookup_scratch_bytes(pair_count: int) -> int:
    """Return the conservative transient index/cache bytes for pair lookup."""

    if pair_count < 0:
        raise ValueError("pair count must be nonnegative")
    return pair_count * PAIR_LOOKUP_SCRATCH_BYTES_PER_PAIR


def pair_selection_fixed_scratch_bytes(
    *,
    trial_count: int,
    option_count: int,
    prior_count: int,
) -> int:
    """Model live caller state around one streamed pair-distance block.

    The estimate covers the selected option and prior-choice matrices, caller
    row indices, per-trial maxima in unsigned and signed form, accumulated
    option weights, the baseline vector, and worst-case block reductions. Pair
    inputs, cache lookup state, and distance output remain in
    :func:`pair_lookup_scratch_bytes` so there is one per-pair resource model.
    """

    if trial_count < 0 or option_count < 0 or prior_count < 0:
        raise ValueError("pair-selection dimensions must be nonnegative")
    integer_bytes = np.dtype(np.int64).itemsize
    option_matrix_bytes = trial_count * option_count * integer_bytes
    prior_matrix_bytes = trial_count * prior_count * integer_bytes
    per_trial_bytes = trial_count * integer_bytes
    return (
        # options, accumulated weights, and the largest option-block sum
        3 * option_matrix_bytes
        + prior_matrix_bytes
        # rows, maxima, signed maxima, block maxima, and baselines
        + 5 * per_trial_bytes
        + 8 * _NDARRAY_HEADER_BYTES
    )


def pair_selection_reduction_scratch_bytes(pair_count: int) -> int:
    """Model pair-proportional arrays in streamed selection reduction."""

    if pair_count < 0:
        raise ValueError("pair count must be nonnegative")
    return _PAIR_SELECTION_REDUCTION_ARRAY_HEADERS + pair_count * PAIR_SELECTION_REDUCTION_BYTES_PER_PAIR


def pair_selection_reduction_chunk_size(*, budget_bytes: int) -> int:
    """Return the largest selection-reduction batch inside ``budget_bytes``."""

    if budget_bytes <= _PAIR_SELECTION_REDUCTION_ARRAY_HEADERS:
        raise ValueError("selection reduction cannot fit one pair in the scratch envelope")
    pair_count = (budget_bytes - _PAIR_SELECTION_REDUCTION_ARRAY_HEADERS) // PAIR_SELECTION_REDUCTION_BYTES_PER_PAIR
    if pair_count < 1:
        raise ValueError("selection reduction cannot fit one pair in the scratch envelope")
    return pair_count


def upper_triangle_index_batches(
    indices: tuple[int, ...],
    *,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield upper-triangle value pairs without materializing the full triangle.

    Each yielded buffer is reused when iteration resumes. Consumers must finish
    processing one batch before requesting the next, as junction's scorers do.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("pair batch size must be a positive integer")
    if len(indices) < 2:
        return

    values = np.asarray(indices, dtype=np.int64)
    pair_count = len(indices) * (len(indices) - 1) // 2
    capacity = min(pair_count, batch_size)
    left_batch = np.empty(capacity, dtype=np.int64)
    right_batch = np.empty(capacity, dtype=np.int64)
    filled = 0

    for left_position in range(len(values) - 1):
        right_position = left_position + 1
        while right_position < len(values):
            take = min(capacity - filled, len(values) - right_position)
            stop = filled + take
            left_batch[filled:stop] = values[left_position]
            right_batch[filled:stop] = values[right_position : right_position + take]
            filled = stop
            right_position += take
            if filled == capacity:
                yield left_batch, right_batch
                filled = 0

    if filled:
        yield left_batch[:filled], right_batch[:filled]


__all__ = [
    "PAIR_LOOKUP_SCRATCH_BYTES_PER_PAIR",
    "PAIR_SELECTION_REDUCTION_BYTES_PER_PAIR",
    "pair_lookup_scratch_bytes",
    "pair_selection_fixed_scratch_bytes",
    "pair_selection_reduction_chunk_size",
    "pair_selection_reduction_scratch_bytes",
    "upper_triangle_index_batches",
]
