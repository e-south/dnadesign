"""Behavior tests for TriJunction's dependency-free DNA sequence primitives."""

from __future__ import annotations

import math

import pytest

from dnadesign.trijunction.sequence import (
    directional_position_weighted_levenshtein,
    kmer_set,
    kmer_set_with_reverse_complements,
    levenshtein_distance,
    longest_common_substring_length,
    position_weighted_levenshtein,
    reverse_complement,
    validate_dna,
)


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
