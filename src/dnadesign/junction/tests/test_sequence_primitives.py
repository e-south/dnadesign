"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_sequence_primitives.py

Behavior tests for junction's dependency-free DNA sequence primitives.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dnadesign.junction.sequence import (
    directional_position_weighted_levenshtein,
    kmer_set,
    kmer_set_with_reverse_complements,
    levenshtein_distance,
    longest_common_substring_length,
    position_weighted_levenshtein,
    reverse_complement,
    validate_dna,
)
from dnadesign.junction.sequence import distances as distance_module


def test_validate_dna_accepts_only_strict_uppercase_dna() -> None:
    assert validate_dna("ACGT") == "ACGT"
    assert validate_dna("") == ""

    with pytest.raises(ValueError, match=r"sequence.*uppercase.*position 2.*'n'"):
        validate_dna("ACnT")

    with pytest.raises(TypeError, match=r"sequence must be a string.*int"):
        validate_dna(42)  # type: ignore[arg-type]


def test_reverse_complement_preserves_five_prime_to_three_prime_convention() -> None:
    assert reverse_complement("AAGTC") == "GACTT"
    assert reverse_complement("") == ""


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("", "", 0),
        ("ACGT", "ACGT", 0),
        ("ACGT", "AGT", 1),
        ("ACGT", "TCGA", 2),
        ("A", "AAA", 2),
    ],
)
def test_levenshtein_distance_uses_conventional_unit_edit_costs(
    left: str,
    right: str,
    expected: int,
) -> None:
    assert levenshtein_distance(left, right) == expected
    assert levenshtein_distance(right, left) == expected


def test_directional_position_weighting_penalizes_ligation_proximal_edits_more() -> None:
    proximal = directional_position_weighted_levenshtein("AC", "TC")
    distal = directional_position_weighted_levenshtein("AC", "AT")

    assert proximal == pytest.approx(2.0)
    assert distal == pytest.approx(1.0 + math.exp(-1.0))
    assert proximal > distal


def test_directional_position_weighting_applies_explicit_v1_indel_coordinates() -> None:
    expected = 2.0 + (1.0 + math.exp(-1.0))

    assert directional_position_weighted_levenshtein("AC", "") == pytest.approx(expected)
    assert directional_position_weighted_levenshtein("", "AC") == pytest.approx(expected)
    assert directional_position_weighted_levenshtein("AC", "AC") == 0.0


def test_position_weighted_distance_uses_minimum_directional_score() -> None:
    forward = directional_position_weighted_levenshtein("ACG", "CA")
    reverse = directional_position_weighted_levenshtein("CA", "ACG")

    assert position_weighted_levenshtein("ACG", "CA") == pytest.approx(min(forward, reverse))
    assert position_weighted_levenshtein("CA", "ACG") == pytest.approx(min(forward, reverse))


def test_empty_encoded_weighted_distance_batch_returns_before_resource_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence_length = 10_000_000
    encoded = np.empty((0, sequence_length), dtype=np.uint8)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("empty batches must not construct weights or size chunks")

    monkeypatch.setattr(distance_module, "position_weight_units", fail_if_called)
    monkeypatch.setattr(distance_module, "_position_weighted_chunk_size", fail_if_called)

    observed = distance_module._position_weighted_levenshtein_units_encoded_many(encoded, encoded)

    assert observed.shape == (0,)
    assert observed.dtype == np.uint64


def test_weighted_distance_scratch_estimate_includes_weights_and_encoded_pairs() -> None:
    pair_count = 7
    sequence_length = 31
    int64_bytes = np.dtype(np.int64).itemsize
    expected_values = (
        pair_count * (2 * (sequence_length + 1) + distance_module._POSITION_WEIGHTED_TEMPORARY_VECTORS)
        + (sequence_length + 1)
        + sequence_length
    )

    encoded_pair_bytes = pair_count * 2 * sequence_length * np.dtype(np.uint8).itemsize
    assert distance_module._position_weighted_scratch_bytes(pair_count, sequence_length) == (
        int64_bytes * expected_values
        + encoded_pair_bytes
        + distance_module._position_weight_cache_construction_bytes(sequence_length)
    )

    chunk_size = distance_module._position_weighted_chunk_size(sequence_length)
    assert (
        distance_module._position_weighted_scratch_bytes(chunk_size, sequence_length)
        <= distance_module._MAX_POSITION_WEIGHTED_SCRATCH_BYTES
    )
    assert (
        distance_module._position_weighted_scratch_bytes(chunk_size + 1, sequence_length)
        > distance_module._MAX_POSITION_WEIGHTED_SCRATCH_BYTES
    )


def test_oversized_weighted_distance_rejects_before_constructing_cached_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary_length = distance_module._MAX_CACHED_POSITION_WEIGHT_COUNT
    assert (
        distance_module._position_weight_cache_construction_bytes(boundary_length)
        <= distance_module._MAX_POSITION_WEIGHT_CACHE_CONSTRUCTION_BYTES
    )
    assert (
        distance_module._position_weight_cache_construction_bytes(boundary_length + 1)
        > distance_module._MAX_POSITION_WEIGHT_CACHE_CONSTRUCTION_BYTES
    )
    assert distance_module._position_weighted_chunk_size(boundary_length) >= 1

    encoded = np.empty((1, boundary_length + 1), dtype=np.uint8)

    def fail_if_called(_length: int) -> tuple[int, ...]:
        raise AssertionError("oversized batches must reject before constructing cached weights")

    monkeypatch.setattr(distance_module, "position_weight_units", fail_if_called)

    with pytest.raises(ValueError, match="cached-weight construction envelope"):
        distance_module._position_weighted_levenshtein_units_encoded_many(encoded, encoded)


def test_position_weight_cache_retention_stays_inside_the_resource_envelope() -> None:
    boundary_length = distance_module._MAX_CACHED_POSITION_WEIGHT_COUNT
    cache_maxsize = distance_module.position_weight_units.cache_info().maxsize

    assert cache_maxsize == distance_module._POSITION_WEIGHT_CACHE_MAXSIZE
    assert cache_maxsize is not None
    assert (
        cache_maxsize * distance_module._position_weight_cache_construction_bytes(boundary_length)
        <= distance_module._MAX_POSITION_WEIGHTED_SCRATCH_BYTES
    )


def test_position_weight_units_rejects_oversized_direct_calls_before_decimal_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized_length = distance_module._MAX_CACHED_POSITION_WEIGHT_COUNT + 1

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized weight tuples must reject before Decimal construction")

    monkeypatch.setattr(distance_module, "Decimal", fail_if_called)

    with pytest.raises(ValueError, match="cached-weight construction envelope"):
        distance_module.position_weight_units(oversized_length)


@pytest.mark.parametrize(
    "distance",
    [
        distance_module.directional_position_weighted_levenshtein,
        distance_module.position_weighted_levenshtein_units,
        distance_module.position_weighted_levenshtein,
    ],
)
def test_oversized_scalar_weighted_distances_reject_before_dna_validation(
    distance: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = "A" * (distance_module._MAX_CACHED_POSITION_WEIGHT_COUNT + 1)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized strings must reject before DNA validation")

    monkeypatch.setattr(distance_module, "validate_dna", fail_if_called)

    with pytest.raises(ValueError, match="cached-weight construction envelope"):
        distance(oversized, "A")  # type: ignore[operator]


def test_oversized_batched_weighted_distances_reject_before_validation_or_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = "A" * (distance_module._MAX_CACHED_POSITION_WEIGHT_COUNT + 1)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized batches must reject before validation or encoding")

    monkeypatch.setattr(distance_module, "validate_dna", fail_if_called)
    monkeypatch.setattr(distance_module.np, "frombuffer", fail_if_called)

    with pytest.raises(ValueError, match="cached-weight construction envelope"):
        distance_module.position_weighted_levenshtein_units_many((oversized,), (oversized,))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("", "ACGT", 0),
        ("AAAA", "TTTT", 0),
        ("GATTACA", "TACTAG", 3),
        ("ACGT", "TACGT", 4),
    ],
)
def test_longest_common_substring_length_is_contiguous(
    left: str,
    right: str,
    expected: int,
) -> None:
    assert longest_common_substring_length(left, right) == expected


def test_kmer_sets_are_unique_and_can_include_reverse_complements() -> None:
    assert kmer_set("AACAA", 2) == {"AA", "AC", "CA"}
    assert kmer_set_with_reverse_complements("AAGT", 2) == {
        "AA",
        "AC",
        "AG",
        "CT",
        "GT",
        "TT",
    }


@pytest.mark.parametrize("k", [0, -1, 5])
def test_kmer_set_rejects_infeasible_k_with_context(k: int) -> None:
    with pytest.raises(ValueError, match=rf"k must be.*{k}.*sequence length 4"):
        kmer_set("ACGT", k)


def test_all_sequence_operations_fail_fast_on_invalid_dna() -> None:
    with pytest.raises(ValueError, match="right.*position 1.*'N'"):
        levenshtein_distance("AC", "AN")

    with pytest.raises(ValueError, match="sequence.*position 1.*'u'"):
        kmer_set("Au", 1)
