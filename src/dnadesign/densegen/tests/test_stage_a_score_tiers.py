"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_a_score_tiers.py

Score-based tiering and retention ordering for Stage-A PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.densegen.src.adapters.sources import pwm_sampling


def test_score_tier_counts_partition() -> None:
    assert pwm_sampling._score_tier_counts(0) == (0, 0, 0)
    assert pwm_sampling._score_tier_counts(1) == (1, 0, 0)
    assert pwm_sampling._score_tier_counts(99) == (1, 9, 89)
    assert pwm_sampling._score_tier_counts(100) == (1, 9, 90)
    for total in (1, 5, 42, 99, 100, 101):
        n0, n1, n2 = pwm_sampling._score_tier_counts(total)
        assert n0 >= 1
        assert n0 + n1 + n2 == total


def test_assign_score_tiers_partitions_ranked_list() -> None:
    ranked = [(f"seq{i:03d}", float(100 - i)) for i in range(100)]
    tiers = pwm_sampling._assign_score_tiers(ranked)
    assert len(tiers) == len(ranked)
    assert tiers.count(0) == 1
    assert tiers.count(1) == 9
    assert tiers.count(2) == 90
    assert set(tiers) == {0, 1, 2}


def test_rank_scored_sequences_dedup_and_tiebreak() -> None:
    scored = [
        ("BBB", 5.0),
        ("AAA", 5.0),
        ("CCC", 7.0),
        ("AAA", 6.0),
        ("DDD", 5.0),
    ]
    ranked = pwm_sampling._rank_scored_sequences(scored)
    assert ranked == [
        ("CCC", 7.0),
        ("AAA", 6.0),
        ("BBB", 5.0),
        ("DDD", 5.0),
    ]
