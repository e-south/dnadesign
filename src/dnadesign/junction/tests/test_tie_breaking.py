"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/tests/test_tie_breaking.py

Deterministic tie contracts for method-v1 search results.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from fractions import Fraction

from dnadesign.junction.design.loci import ToeholdCandidate
from dnadesign.junction.design.matching import match_barcodes
from dnadesign.junction.design.scoring import rank_aggregate_maximin


def _candidate(*, target_id: str, sequence: str) -> ToeholdCandidate:
    return ToeholdCandidate(
        target_id=target_id,
        assembly_group_id="assembly-a",
        locus_index=0,
        candidate_offset=0,
        start=0,
        sequence=sequence,
    )


def test_dense_rank_aggregation_gives_equal_scores_equal_ranks() -> None:
    ranks = rank_aggregate_maximin(
        {
            "lexically-last": (5, Fraction(7, 2)),
            "lexically-first": (5, Fraction(7, 2)),
            "lower": (3, Fraction(2)),
        }
    )

    assert ranks["lexically-first"] == ranks["lexically-last"]
    assert ranks["lexically-first"].weighted_score_fraction == Fraction(3, 2)
    assert ranks["lower"].weighted_score_fraction == Fraction(0)


def test_equal_lcs_matchings_choose_the_lexically_smallest_assignment() -> None:
    result = match_barcodes(
        (
            _candidate(target_id="target-b", sequence="CCCC"),
            _candidate(target_id="target-a", sequence="AAAA"),
        ),
        ("TTTT", "GGGG"),
        iterations=2,
        seed=17,
    )

    assert result.matchings_evaluated == 2
    assert result.max_pairwise_lcs == 0
    assert tuple(assignment.candidate.target_id for assignment in result.assignments) == (
        "target-a",
        "target-b",
    )
    assert tuple(assignment.barcode for assignment in result.assignments) == ("GGGG", "TTTT")
