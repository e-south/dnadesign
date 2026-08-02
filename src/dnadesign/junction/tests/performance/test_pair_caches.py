"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/performance/test_pair_caches.py

Bounded pair-cache and pair-index streaming contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from dnadesign.junction.design import barcodes as barcode_module
from dnadesign.junction.design import toeholds as toehold_module
from dnadesign.junction.sequence import levenshtein_distance, position_weighted_levenshtein_units
from dnadesign.junction.tests.performance._factories import candidate as _candidate


def test_toehold_pair_cache_chunks_missing_pairs_before_encoded_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_batch_size = 64
    search_range = 4_096
    toehold_length = 61
    request_pair_count = search_batch_size * search_range
    candidate_count = 1_024
    left, right = np.triu_indices(candidate_count, 1)
    left = left[:request_pair_count]
    right = right[:request_pair_count]
    pair_chunk_size = toehold_module._toehold_pair_chunk_size(toehold_length)
    indexed_pair_counts: list[int] = []
    dispatched_pair_counts: list[int] = []

    class TrackedEncodedCandidates:
        shape = (candidate_count, toehold_length)

        def __getitem__(self, indices: np.ndarray) -> np.ndarray:
            indexed_pair_counts.append(len(indices))
            encoded = np.zeros((len(indices), toehold_length), dtype=np.uint8)
            encoded[:, 0] = indices % 251
            return encoded

    def record_distance_dispatch(encoded_left: np.ndarray, encoded_right: np.ndarray) -> np.ndarray:
        assert encoded_left.shape == encoded_right.shape
        dispatched_pair_counts.append(encoded_left.shape[0])
        return encoded_left[:, 0].astype(np.uint64) * 256 + encoded_right[:, 0]

    cache = toehold_module._PairDistanceCache(
        tuple(_candidate(index, "A" * toehold_length) for index in range(candidate_count))
    )
    cache._encoded = TrackedEncodedCandidates()
    monkeypatch.setattr(
        toehold_module,
        "_position_weighted_levenshtein_units_encoded_many",
        record_distance_dispatch,
    )

    observed = cache.distances(left, right)
    expected = (left % 251).astype(np.uint64) * 256 + right % 251

    assert observed.shape == (request_pair_count,)
    assert np.array_equal(observed, expected)
    assert cache.computed_pairs == request_pair_count
    assert len(dispatched_pair_counts) > 1
    assert indexed_pair_counts == [count for count in dispatched_pair_counts for _ in range(2)]
    assert max((*indexed_pair_counts, *dispatched_pair_counts)) <= pair_chunk_size
    assert (
        toehold_module._toehold_pair_scratch_bytes(pair_chunk_size, toehold_length)
        <= toehold_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
    )
    assert (
        toehold_module._toehold_pair_scratch_bytes(pair_chunk_size + 1, toehold_length)
        > toehold_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
    )

    dispatched_count = len(dispatched_pair_counts)
    assert np.array_equal(cache.distances(right[::-1], left[::-1]), expected[::-1])
    assert cache.computed_pairs == request_pair_count
    assert len(dispatched_pair_counts) == dispatched_count


def test_toehold_pair_cache_computes_cross_chunk_duplicates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequences = ("AAAAAAAA", "AAAAAAAC", "AAAAAACC", "AAAACCCC")
    candidates = tuple(_candidate(index, sequence) for index, sequence in enumerate(sequences))
    left = np.asarray([0, 0, 1, 0, 1, 2, 0], dtype=np.int64)
    right = np.asarray([1, 2, 2, 1, 2, 3, 3], dtype=np.int64)
    dispatched_pair_counts: list[int] = []
    original_distance = toehold_module._position_weighted_levenshtein_units_encoded_many

    def record_distance_dispatch(encoded_left: np.ndarray, encoded_right: np.ndarray) -> np.ndarray:
        dispatched_pair_counts.append(encoded_left.shape[0])
        return original_distance(encoded_left, encoded_right)

    monkeypatch.setattr(toehold_module, "_toehold_pair_chunk_size", lambda _length: 2)
    monkeypatch.setattr(
        toehold_module,
        "_position_weighted_levenshtein_units_encoded_many",
        record_distance_dispatch,
    )
    cache = toehold_module._PairDistanceCache(candidates)

    observed = cache.distances(left, right)
    expected = np.asarray(
        [position_weighted_levenshtein_units(sequences[a], sequences[b]) for a, b in zip(left, right, strict=True)],
        dtype=np.uint64,
    )

    assert np.array_equal(observed, expected)
    assert sum(dispatched_pair_counts) == 5
    assert cache.computed_pairs == 5

    dispatched_count = len(dispatched_pair_counts)
    assert np.array_equal(cache.distances(right[::-1], left[::-1]), expected[::-1])
    assert cache.computed_pairs == 5
    assert len(dispatched_pair_counts) == dispatched_count


@pytest.mark.parametrize(
    "score_subset",
    [barcode_module._subset_score_indices, toehold_module._path_score],
)
def test_pair_scores_stream_upper_triangle_indices_in_bounded_batches(
    score_subset: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (0, 1, 2, 3, 4)
    batch_sizes: list[int] = []

    class PairScoreCache:
        pair_chunk_size = 3

        def distances(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
            batch_sizes.append(len(left))
            return (10 * left + right + 1).astype(np.uint64)

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pair scoring must not materialize the complete upper triangle")

    monkeypatch.setattr(np, "triu_indices", fail_if_materialized)
    values = [10 * left + right + 1 for left, right in combinations(path, 2)]

    observed = score_subset(path, PairScoreCache())  # type: ignore[operator]

    assert observed == (min(values), Fraction(sum(values), len(values)))
    assert batch_sizes == [3, 3, 3, 1]


def test_barcode_pair_cache_chunks_a_legal_wide_subset_before_encoded_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_count = 2_317
    candidate_count = 11_585
    barcode_length = 8
    request_pair_count = selected_count * (selected_count - 1) // 2
    indexed_pair_counts: list[int] = []
    dispatched_pair_counts: list[int] = []

    class SyntheticDistanceValues:
        """Model cache publication without allocating its 128 MiB matrix."""

        def __init__(self) -> None:
            self.assigned_count = 0
            self.last_assignment: np.ndarray | None = None

        def __getitem__(self, indices: np.ndarray) -> np.ndarray:
            fill = (
                0
                if self.last_assignment is not None and np.array_equal(indices, self.last_assignment)
                else barcode_module._UNSET_BARCODE_DISTANCE
            )
            return np.full(indices.shape, fill, dtype=np.uint16)

        def __setitem__(self, indices: np.ndarray, values: np.ndarray) -> None:
            assert np.all(values == 0)
            self.assigned_count += len(indices)
            self.last_assignment = indices.copy()

    class TrackedEncodedCandidates:
        shape = (candidate_count, barcode_length)

        def __getitem__(self, indices: np.ndarray) -> np.ndarray:
            indexed_pair_counts.append(len(indices))
            return np.zeros((len(indices), barcode_length), dtype=np.uint8)

    def record_distance_dispatch(encoded_left: np.ndarray, encoded_right: np.ndarray) -> np.ndarray:
        assert encoded_left.shape == encoded_right.shape
        dispatched_pair_counts.append(encoded_left.shape[0])
        return np.zeros(encoded_left.shape[0], dtype=np.int32)

    cache = object.__new__(barcode_module._BarcodeDistanceCache)
    cache._size = candidate_count
    cache._encoded = TrackedEncodedCandidates()
    cache._values = SyntheticDistanceValues()
    monkeypatch.setattr(barcode_module, "_levenshtein_distance_encoded_many", record_distance_dispatch)

    def fail_if_materialized(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("wide subset scoring must stream pair indices")

    monkeypatch.setattr(np, "triu_indices", fail_if_materialized)
    observed = barcode_module._subset_score_indices(tuple(range(selected_count)), cache)

    assert request_pair_count == 2_683_086
    assert observed == (0, Fraction(0))
    assert len(dispatched_pair_counts) > 1
    assert sum(dispatched_pair_counts) == request_pair_count
    assert cache._values.assigned_count == request_pair_count
    assert indexed_pair_counts == [count for count in dispatched_pair_counts for _ in range(2)]
    pair_chunk_size = cache.pair_chunk_size
    assert max(dispatched_pair_counts) <= pair_chunk_size
    assert (
        barcode_module._barcode_pair_scratch_bytes(pair_chunk_size, barcode_length)
        <= barcode_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
    )
    assert (
        barcode_module._barcode_pair_scratch_bytes(pair_chunk_size + 1, barcode_length)
        > barcode_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
    )
    assert all(
        barcode_module._barcode_pair_scratch_bytes(call_pair_count, barcode_length)
        <= barcode_module.MAX_PAIR_DISTANCE_SCRATCH_BYTES
        for call_pair_count in dispatched_pair_counts
    )


def test_barcode_pair_cache_preserves_order_and_computes_duplicate_pairs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = ("AAAAAAAA", "AAAAAAAC", "AAAAAACC", "AAAACCCC")
    left = np.asarray([0, 0, 1, 0, 1, 2, 0], dtype=np.int64)
    right = np.asarray([1, 2, 2, 1, 2, 3, 3], dtype=np.int64)
    dispatched_pair_counts: list[int] = []
    original_distance = barcode_module._levenshtein_distance_encoded_many

    def record_distance_dispatch(encoded_left: np.ndarray, encoded_right: np.ndarray) -> np.ndarray:
        dispatched_pair_counts.append(encoded_left.shape[0])
        return original_distance(encoded_left, encoded_right)

    monkeypatch.setattr(barcode_module, "_barcode_pair_chunk_size", lambda _length: 2)
    monkeypatch.setattr(barcode_module, "_levenshtein_distance_encoded_many", record_distance_dispatch)
    cache = barcode_module._BarcodeDistanceCache(candidates)

    observed = cache.distances(left, right)
    expected = np.asarray(
        [levenshtein_distance(candidates[a], candidates[b]) for a, b in zip(left, right, strict=True)],
        dtype=np.uint16,
    )

    assert np.array_equal(observed, expected)
    assert sum(dispatched_pair_counts) == 5

    dispatched_count = len(dispatched_pair_counts)
    assert np.array_equal(cache.distances(right[::-1], left[::-1]), expected[::-1])
    assert len(dispatched_pair_counts) == dispatched_count
