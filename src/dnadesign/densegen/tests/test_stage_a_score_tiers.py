"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_score_tiers.py

Score-based tiering and retention ordering for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources import stage_a_summary
from dnadesign.densegen.src.core import score_tiers


def test_score_tier_counts_partition() -> None:
    assert score_tiers.score_tier_counts(0) == (0, 0, 0, 0)
    assert score_tiers.score_tier_counts(1) == (1, 0, 0, 0)
    assert score_tiers.score_tier_counts(99) == (1, 1, 9, 88)
    assert score_tiers.score_tier_counts(100) == (1, 1, 9, 89)
    for total in (1, 5, 42, 99, 100, 101):
        n0, n1, n2, n3 = score_tiers.score_tier_counts(total)
        assert n0 >= 1
        assert n0 + n1 + n2 + n3 == total


def test_assign_score_tiers_partitions_ranked_list() -> None:
    ranked = [(f"seq{i:03d}", float(100 - i)) for i in range(100)]
    tiers = stage_a_summary._assign_score_tiers(ranked)
    assert len(tiers) == len(ranked)
    assert tiers.count(0) == 1
    assert tiers.count(1) == 1
    assert tiers.count(2) == 9
    assert tiers.count(3) == 89
    assert set(tiers) == {0, 1, 2, 3}


def test_score_tier_counts_custom_fractions() -> None:
    n0, n1, n2, n3 = score_tiers.score_tier_counts(10, fractions=(0.1, 0.2, 0.3))
    assert (n0, n1, n2, n3) == (1, 2, 3, 4)


def test_resolve_tier_fractions_from_ladder() -> None:
    resolved = score_tiers.resolve_tier_fractions([0.001, 0.01, 0.09, 1.0])
    assert resolved == (0.001, 0.01, 0.09)


def test_score_tier_counts_rejects_invalid_fractions() -> None:
    for fractions in [(0.5, 0.4, 0.2), (0.1, 0.2), (1.2, 0.2, 0.1)]:
        try:
            score_tiers.score_tier_counts(10, fractions=fractions)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid tier fractions should raise ValueError.")


def test_rank_scored_sequences_dedup_and_tiebreak() -> None:
    scored = [
        ("BBB", 5.0),
        ("AAA", 5.0),
        ("CCC", 7.0),
        ("AAA", 6.0),
        ("DDD", 5.0),
    ]
    ranked = stage_a_summary._rank_scored_sequences(scored)
    assert ranked == [
        ("CCC", 7.0),
        ("AAA", 6.0),
        ("BBB", 5.0),
        ("DDD", 5.0),
    ]


def test_ranked_sequence_positions_are_one_based() -> None:
    ranked = [("CCC", 7.0), ("AAA", 6.0), ("BBB", 5.0)]
    positions = stage_a_summary._ranked_sequence_positions(ranked)
    assert positions == {"CCC": 1, "AAA": 2, "BBB": 3}
