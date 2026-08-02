"""Explicit rank aggregation for maximin sequence searches."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias, TypeVar

Identity = TypeVar("Identity")
ScoreValue: TypeAlias = int | Fraction


@dataclass(frozen=True, slots=True)
class RankAggregate:
    """Normalized descending ranks for one minimum/mean score pair."""

    minimum_rank_fraction: Fraction
    mean_rank_fraction: Fraction
    weighted_score_fraction: Fraction

    @property
    def minimum_rank(self) -> float:
        return float(self.minimum_rank_fraction)

    @property
    def mean_rank(self) -> float:
        return float(self.mean_rank_fraction)

    @property
    def weighted_score(self) -> float:
        return float(self.weighted_score_fraction)


def rank_aggregate_maximin(
    scores: dict[Identity, tuple[ScoreValue, ScoreValue]],
) -> dict[Identity, RankAggregate]:
    """Rank maximin candidates with the documented 1.0/0.5 policy.

    The Sidewinder methods describe weighted rank aggregation but do not define
    its tie handling or normalization. TriJunction v1 assigns equal values an
    equal dense rank, normalizes each descending rank to ``[0, 1]``, and combines
    minimum and mean ranks with weights ``1.0`` and ``0.5`` respectively.
    """

    if not scores:
        raise ValueError("rank aggregation requires at least one candidate")

    def normalized_ranks(values: tuple[ScoreValue, ...]) -> dict[ScoreValue, Fraction]:
        distinct = sorted(set(values), reverse=True)
        if len(distinct) == 1:
            return {distinct[0]: Fraction(1)}
        denominator = len(distinct) - 1
        return {value: Fraction(denominator - index, denominator) for index, value in enumerate(distinct)}

    minimum_ranks = normalized_ranks(tuple(score[0] for score in scores.values()))
    mean_ranks = normalized_ranks(tuple(score[1] for score in scores.values()))
    return {
        identity: RankAggregate(
            minimum_rank_fraction=minimum_ranks[score[0]],
            mean_rank_fraction=mean_ranks[score[1]],
            weighted_score_fraction=minimum_ranks[score[0]] + Fraction(1, 2) * mean_ranks[score[1]],
        )
        for identity, score in scores.items()
    }


__all__ = ["RankAggregate", "rank_aggregate_maximin"]
