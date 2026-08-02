"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/performance/test_toehold_selection_batching.py

Peak-memory contracts for streamed toehold-choice reduction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.junction.design import toeholds as toehold_module
from dnadesign.junction.design.randomness import StablePrng
from dnadesign.junction.design.resources import (
    guard_uniform_toehold_search,
    pair_selection_fixed_scratch_bytes,
    pair_selection_reduction_chunk_size,
    pair_selection_reduction_scratch_bytes,
)


def _eager_selection_weights(distances: np.ndarray) -> np.ndarray:
    maxima = distances.max(axis=(1, 2))
    deltas = distances.astype(np.int64) - maxima[:, None, None].astype(np.int64)
    distinct, inverse = np.unique(deltas, return_inverse=True)
    lookup = np.asarray(
        [toehold_module._stable_exp_weight(int(delta)) for delta in distinct],
        dtype=np.uint64,
    )
    contributions = lookup[inverse].reshape(distances.shape)
    baselines = np.asarray(
        [toehold_module._stable_exp_weight(-int(maximum)) for maximum in maxima],
        dtype=np.uint64,
    )
    return contributions.sum(axis=2, dtype=np.uint64) + baselines[:, None]


def test_toehold_choice_reduction_streams_legal_wide_search_with_exact_weights() -> None:
    trial_count = 64
    option_count = 128
    prior_count = 63
    sequence_length = 7
    options = np.arange(trial_count * option_count, dtype=np.int64).reshape(trial_count, option_count)
    previous = np.arange(trial_count * prior_count, dtype=np.int64).reshape(trial_count, prior_count) + 17
    pair_counts: list[int] = []

    class DeterministicDistances:
        pair_chunk_size = toehold_module._toehold_pair_chunk_size(sequence_length)

        def distances(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
            assert left.shape == right.shape
            pair_counts.append(left.size)
            return ((left.astype(np.uint64) * 17 + right.astype(np.uint64) * 31) % 97) * 10_000_000

    full_distances = (
        DeterministicDistances()
        .distances(
            np.broadcast_to(options[:, :, None], (trial_count, option_count, prior_count)).reshape(-1),
            np.broadcast_to(previous[:, None, :], (trial_count, option_count, prior_count)).reshape(-1),
        )
        .reshape(trial_count, option_count, prior_count)
    )
    expected = _eager_selection_weights(full_distances)
    del full_distances
    pair_counts.clear()

    pair_chunk_size = toehold_module._toehold_selection_pair_chunk_size(
        sequence_length,
        trial_count=trial_count,
        option_count=option_count,
        prior_count=prior_count,
    )
    observed = toehold_module._selection_weights_streamed(
        options,
        previous,
        DeterministicDistances(),
        pair_chunk_size=pair_chunk_size,
    )

    full_pair_count = trial_count * option_count * prior_count
    assert np.array_equal(observed, expected)
    assert [
        StablePrng(1_000 + trial).weighted_choice(range(option_count), tuple(int(value) for value in row))
        for trial, row in enumerate(observed)
    ] == [
        StablePrng(1_000 + trial).weighted_choice(range(option_count), tuple(int(value) for value in row))
        for trial, row in enumerate(expected)
    ]
    guard_uniform_toehold_search(
        locus_count=64,
        candidates_per_locus=128,
        sequence_length=sequence_length,
        iterations=64,
    )
    assert full_pair_count == 516_096
    assert full_pair_count * 3 * np.dtype(np.int64).itemsize == 12_386_304
    assert len(pair_counts) > 2
    assert sum(pair_counts) == 2 * full_pair_count
    assert max(pair_counts) < full_pair_count
    fixed_bytes = pair_selection_fixed_scratch_bytes(
        trial_count=trial_count,
        option_count=option_count,
        prior_count=prior_count,
    )
    reduction_chunk_size = pair_selection_reduction_chunk_size(
        budget_bytes=toehold_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES - fixed_bytes
    )
    assert (
        fixed_bytes + pair_selection_reduction_scratch_bytes(reduction_chunk_size)
        <= toehold_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
    )
    assert (
        fixed_bytes + pair_selection_reduction_scratch_bytes(reduction_chunk_size + 1)
        > toehold_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
    )
    assert all(
        toehold_module._toehold_selection_scratch_bytes(
            pair_count,
            sequence_length=sequence_length,
            trial_count=trial_count,
            option_count=option_count,
            prior_count=prior_count,
        )
        <= toehold_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
        for pair_count in pair_counts
    )
    assert (
        toehold_module._toehold_selection_scratch_bytes(
            pair_chunk_size,
            sequence_length=sequence_length,
            trial_count=trial_count,
            option_count=option_count,
            prior_count=prior_count,
        )
        <= toehold_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
    )
    assert (
        toehold_module._toehold_selection_scratch_bytes(
            pair_chunk_size + 1,
            sequence_length=sequence_length,
            trial_count=trial_count,
            option_count=option_count,
            prior_count=prior_count,
        )
        > toehold_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
    )
