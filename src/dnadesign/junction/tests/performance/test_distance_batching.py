"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/performance/test_distance_batching.py

Batching and scratch-memory contracts for weighted sequence distance.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.junction.sequence import distances as distance_module
from dnadesign.junction.sequence import (
    position_weighted_levenshtein_units,
    position_weighted_levenshtein_units_many,
)


def test_vectorized_weighted_distances_equal_the_scalar_fixed_point_contract() -> None:
    left = ("ACGATTCGGT", "GATTACAGAT", "ACGTACGTAC", "TTTTTTTTTT")
    right = ("CGCTTAGACT", "TACTAGATTA", "TACGTACGTA", "TTTTTATTTT")

    observed = tuple(int(value) for value in position_weighted_levenshtein_units_many(left, right))
    expected = tuple(position_weighted_levenshtein_units(a, b) for a, b in zip(left, right, strict=True))

    assert observed == expected


def test_public_weighted_distance_batch_bounds_encoding_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = ("ACGA", "GATT", "ACGT", "TTTT", "CGAT")
    right = ("CGCT", "TACT", "TACG", "TTTA", "GCTA")
    encoded_byte_counts: list[int] = []
    dispatched_pair_counts: list[int] = []
    original_frombuffer = np.frombuffer
    original_dispatch = distance_module._position_weighted_levenshtein_units_encoded_many

    def record_frombuffer(buffer: bytes, *args: object, **kwargs: object) -> np.ndarray:
        encoded_byte_counts.append(len(buffer))
        return original_frombuffer(buffer, *args, **kwargs)

    def record_dispatch(encoded_left: np.ndarray, encoded_right: np.ndarray) -> np.ndarray:
        dispatched_pair_counts.append(encoded_left.shape[0])
        return original_dispatch(encoded_left, encoded_right)

    monkeypatch.setattr(distance_module, "_position_weighted_chunk_size", lambda _length, **_kwargs: 2)
    monkeypatch.setattr(distance_module.np, "frombuffer", record_frombuffer)
    monkeypatch.setattr(distance_module, "_position_weighted_levenshtein_units_encoded_many", record_dispatch)

    observed = tuple(int(value) for value in position_weighted_levenshtein_units_many(left, right))
    expected = tuple(position_weighted_levenshtein_units(a, b) for a, b in zip(left, right, strict=True))

    assert observed == expected
    assert dispatched_pair_counts == [2, 2, 1]
    assert encoded_byte_counts == [8, 8, 8, 8, 4, 4]


def test_public_weighted_distance_batch_validates_all_sequences_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_encoded(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("batch encoding must not begin before validation completes")

    monkeypatch.setattr(distance_module.np, "frombuffer", fail_if_encoded)

    with pytest.raises(ValueError, match=r"left.*position 3.*'N'"):
        position_weighted_levenshtein_units_many(("ACGT", "ACGN"), ("TGCA", "TGCA"))


def test_wide_weighted_distance_batches_bound_every_directional_scratch_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_batch_size = 64
    search_range = 4_096
    toehold_length = 61
    pair_count = search_batch_size * search_range
    encoded = np.broadcast_to(
        np.zeros((1, toehold_length), dtype=np.uint8),
        (pair_count, toehold_length),
    )
    directional_shapes: list[tuple[int, int]] = []

    def record_directional_call(
        source: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        assert source.shape == target.shape
        assert weights.shape == (toehold_length,)
        directional_shapes.append(source.shape)
        return np.zeros(source.shape[0], dtype=np.int64)

    monkeypatch.setattr(distance_module, "_directional_units_many", record_directional_call)

    observed = distance_module._position_weighted_levenshtein_units_encoded_many(encoded, encoded)

    assert observed.shape == (pair_count,)
    assert len(directional_shapes) > 2
    assert all(
        distance_module._position_weighted_scratch_bytes(call_pair_count, sequence_length)
        <= distance_module._MAX_POSITION_WEIGHTED_SCRATCH_BYTES
        for call_pair_count, sequence_length in directional_shapes
    )


def test_weighted_chunk_boundary_includes_the_simultaneous_python_weight_tuple() -> None:
    sequence_length = 61
    chunk_size = distance_module._position_weighted_chunk_size(sequence_length)
    int64_bytes = np.dtype(np.int64).itemsize
    numpy_fixed_bytes = int64_bytes * (2 * sequence_length + 1)
    per_pair_bytes = (
        int64_bytes * (2 * (sequence_length + 1) + distance_module._POSITION_WEIGHTED_TEMPORARY_VECTORS)
        + 2 * sequence_length * np.dtype(np.uint8).itemsize
    )
    expected_simultaneous_peak = (
        numpy_fixed_bytes
        + distance_module._position_weight_cache_construction_bytes(sequence_length)
        + chunk_size * per_pair_bytes
    )

    assert distance_module._position_weighted_scratch_bytes(chunk_size, sequence_length) == (expected_simultaneous_peak)
    assert expected_simultaneous_peak <= distance_module._MAX_POSITION_WEIGHTED_SCRATCH_BYTES
    assert (
        distance_module._position_weighted_scratch_bytes(chunk_size + 1, sequence_length)
        > distance_module._MAX_POSITION_WEIGHTED_SCRATCH_BYTES
    )


def test_directional_weighted_distance_returns_a_compact_owned_score_vector() -> None:
    encoded = np.asarray([[ord(base) for base in sequence] for sequence in ("ACGT", "TGCA")], dtype=np.uint8)
    weights = np.asarray(distance_module.position_weight_units(4), dtype=np.int64)

    observed = distance_module._directional_units_many(encoded, encoded, weights)

    assert observed.base is None
    assert observed.nbytes == encoded.shape[0] * np.dtype(np.int64).itemsize


def test_chunked_weighted_distances_preserve_exact_pair_order(monkeypatch: pytest.MonkeyPatch) -> None:
    left = ("ACGATTCGGT", "GATTACAGAT", "ACGTACGTAC", "TTTTTTTTTT", "CGATCGATCG")
    right = ("CGCTTAGACT", "TACTAGATTA", "TACGTACGTA", "TTTTTATTTT", "GCTAGCTAGC")
    monkeypatch.setattr(
        distance_module,
        "_MAX_POSITION_WEIGHTED_SCRATCH_BYTES",
        distance_module._position_weighted_scratch_bytes(2, len(left[0])),
    )

    observed = tuple(int(value) for value in position_weighted_levenshtein_units_many(left, right))
    expected = tuple(position_weighted_levenshtein_units(a, b) for a, b in zip(left, right, strict=True))

    assert observed == expected
