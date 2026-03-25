"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/selection.py

Deterministic accepted-pool admission and hit-selection policies for cassette
solve workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dnadesign.cruncher.cassette.models import HairpinCassetteSpec
from dnadesign.cruncher.cassette.solve_models import (
    CandidateScoreBreakdown,
    PairwiseDistanceSummary,
    SearchSelectionSpec,
    SolveSelectionSummary,
)
from dnadesign.cruncher.core.selection.mmr import select_ranked_mmr

if TYPE_CHECKING:
    from dnadesign.cruncher.cassette.solve_models import SearchSettingsSpec


ScoreKey = tuple[tuple[int | float, ...], str]


@dataclass
class CandidateHitRecord:
    hit_id: str
    left_variant_id: str
    right_variant_id: str
    explicit_spec: HairpinCassetteSpec
    report: object
    cassette_sequence: str
    stem5p_arm: str
    loop: str
    gc_fraction: float
    extra_site_count: int
    score_breakdown: CandidateScoreBreakdown
    base_penalty_vector: tuple[int | float, ...]
    score_tuple: tuple[float | int | str, ...]
    left_nick_boundary: int
    right_nick_boundary: int
    bounded_segment_length: int


@dataclass(frozen=True)
class SelectedCandidate:
    record: CandidateHitRecord
    selection_rank_reason: str
    distance_to_previous_selected: int | None


@dataclass(frozen=True)
class AcceptedPoolSummary:
    pool_size: int
    final_size: int
    admitted_count: int
    rejected_count: int
    truncated: bool
    worst_score_at_close: tuple[float | int | str, ...] | None


@dataclass(frozen=True)
class _DescendingScoreKey:
    key: ScoreKey

    def __lt__(self, other: "_DescendingScoreKey") -> bool:
        return self.key > other.key


@dataclass(frozen=True)
class SelectionOutcome:
    accepted_pool: list[CandidateHitRecord]
    selected_hits: list[SelectedCandidate]
    pool_summary: AcceptedPoolSummary
    summary: SolveSelectionSummary


@dataclass
class _PairwiseDistanceCache:
    metric: str
    _distance_cache: dict[tuple[str, str], int] = field(default_factory=dict)

    @staticmethod
    def _cache_key(left: CandidateHitRecord, right: CandidateHitRecord) -> tuple[str, str]:
        left_id, right_id = sorted((left.hit_id, right.hit_id))
        return left_id, right_id

    def distance(self, left: CandidateHitRecord, right: CandidateHitRecord) -> int:
        if left.hit_id == right.hit_id:
            return 0
        key = self._cache_key(left, right)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached
        if self.metric != "hamming":
            raise ValueError(f"Unsupported cassette selection distance_metric: {self.metric}")
        cached = hamming_distance(left.cassette_sequence, right.cassette_sequence)
        self._distance_cache[key] = cached
        return cached

    def similarity(self, left: CandidateHitRecord, right: CandidateHitRecord) -> float:
        if left.hit_id == right.hit_id:
            return 1.0
        distance = self.distance(left, right)
        length = len(left.cassette_sequence)
        if length == 0:
            return 0.0
        return 1.0 - (distance / length)


def hamming_distance(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right, strict=True) if a != b)


def _distance(
    left: CandidateHitRecord,
    right: CandidateHitRecord,
    *,
    metric: str,
    cache: _PairwiseDistanceCache | None = None,
) -> int:
    if cache is not None:
        return cache.distance(left, right)
    if metric != "hamming":
        raise ValueError(f"Unsupported cassette selection distance_metric: {metric}")
    return hamming_distance(left.cassette_sequence, right.cassette_sequence)


def _similarity(
    left: CandidateHitRecord,
    right: CandidateHitRecord,
    *,
    metric: str,
    cache: _PairwiseDistanceCache | None = None,
) -> float:
    if cache is not None:
        return cache.similarity(left, right)
    distance = _distance(left, right, metric=metric)
    length = len(left.cassette_sequence)
    if length == 0:
        return 0.0
    return 1.0 - (distance / length)


def _admission_key(hit: CandidateHitRecord) -> ScoreKey:
    return (tuple(hit.base_penalty_vector), hit.cassette_sequence)


def _full_sort_key(hit: CandidateHitRecord) -> tuple[ScoreKey, str, str, str]:
    return (
        _admission_key(hit),
        hit.left_variant_id,
        hit.right_variant_id,
        hit.hit_id,
    )


def _score_tuple_from_key(key: ScoreKey) -> tuple[float | int | str, ...]:
    base_penalty_vector, tie_break_sequence = key
    return (*base_penalty_vector, tie_break_sequence)


@dataclass
class AcceptedCandidatePool:
    limit: int
    _records_by_key: dict[ScoreKey, CandidateHitRecord] = field(default_factory=dict)
    _worst_key_heap: list[_DescendingScoreKey] = field(default_factory=list)
    admitted_count: int = 0
    rejected_count: int = 0
    truncated: bool = False

    def _push_key(self, key: ScoreKey) -> None:
        heapq.heappush(self._worst_key_heap, _DescendingScoreKey(key))

    def _peek_worst_key(self) -> ScoreKey | None:
        while self._worst_key_heap and self._worst_key_heap[0].key not in self._records_by_key:
            heapq.heappop(self._worst_key_heap)
        if not self._worst_key_heap:
            return None
        return self._worst_key_heap[0].key

    def _pop_worst_key(self) -> ScoreKey | None:
        worst_key = self._peek_worst_key()
        if worst_key is None:
            return None
        heapq.heappop(self._worst_key_heap)
        return worst_key

    def consider(self, hit: CandidateHitRecord) -> None:
        key = _admission_key(hit)
        existing = self._records_by_key.get(key)
        if existing is not None:
            if _full_sort_key(hit) < _full_sort_key(existing):
                self._records_by_key[key] = hit
                self.admitted_count += 1
            else:
                self.rejected_count += 1
            return
        if len(self._records_by_key) < self.limit:
            self._records_by_key[key] = hit
            self.admitted_count += 1
            self._push_key(key)
            return
        self.truncated = True
        worst_key = self._peek_worst_key()
        assert worst_key is not None
        if key < worst_key:
            popped_worst_key = self._pop_worst_key()
            assert popped_worst_key is not None
            del self._records_by_key[popped_worst_key]
            self._records_by_key[key] = hit
            self.admitted_count += 1
            self._push_key(key)
            return
        self.rejected_count += 1

    def ranked_hits(self) -> list[CandidateHitRecord]:
        return sorted(self._records_by_key.values(), key=_full_sort_key)

    def summary(self) -> AcceptedPoolSummary:
        ranked_hits = self.ranked_hits()
        worst_score_at_close = list(ranked_hits[-1].score_tuple) if ranked_hits else None
        return AcceptedPoolSummary(
            pool_size=self.limit,
            final_size=len(ranked_hits),
            admitted_count=self.admitted_count,
            rejected_count=self.rejected_count,
            truncated=self.truncated,
            worst_score_at_close=tuple(worst_score_at_close) if worst_score_at_close is not None else None,
        )


def build_accepted_candidate_pool(*, pool_size: int) -> AcceptedCandidatePool:
    return AcceptedCandidatePool(limit=pool_size)


def _distance_to_previous_selected(
    selected: list[SelectedCandidate],
    candidate: CandidateHitRecord,
    *,
    metric: str,
    cache: _PairwiseDistanceCache | None = None,
) -> int | None:
    if not selected:
        return None
    return _distance(selected[-1].record, candidate, metric=metric, cache=cache)


def _pairwise_distance_summary(
    hits: list[CandidateHitRecord],
    *,
    metric: str,
    cache: _PairwiseDistanceCache | None = None,
) -> PairwiseDistanceSummary:
    distances: list[int] = []
    for index, left in enumerate(hits):
        for right in hits[index + 1 :]:
            distances.append(_distance(left, right, metric=metric, cache=cache))
    if not distances:
        return PairwiseDistanceSummary()
    return PairwiseDistanceSummary(
        min=float(min(distances)),
        max=float(max(distances)),
        mean=float(sum(distances) / len(distances)),
    )


def _select_ranked_hits_score_only(
    ranked_hits: list[CandidateHitRecord],
    *,
    max_hits: int,
    distance_metric: str,
    pairwise_cache: _PairwiseDistanceCache | None = None,
) -> list[SelectedCandidate]:
    selected: list[SelectedCandidate] = []
    for record in ranked_hits[:max_hits]:
        selected.append(
            SelectedCandidate(
                record=record,
                selection_rank_reason="score_only_rank",
                distance_to_previous_selected=_distance_to_previous_selected(
                    selected,
                    record,
                    metric=distance_metric,
                    cache=pairwise_cache,
                ),
            )
        )
    return selected


def select_hits_score_only(
    hits: list[CandidateHitRecord],
    *,
    max_hits: int,
    distance_metric: str,
) -> list[SelectedCandidate]:
    ranked_hits = sorted(hits, key=_full_sort_key)
    pairwise_cache = _PairwiseDistanceCache(metric=distance_metric)
    return _select_ranked_hits_score_only(
        ranked_hits,
        max_hits=max_hits,
        distance_metric=distance_metric,
        pairwise_cache=pairwise_cache,
    )


def _select_ranked_hits_greedy_hamming(
    ranked_hits: list[CandidateHitRecord],
    *,
    max_hits: int,
    distance_metric: str,
    min_pairwise_distance: int,
    pairwise_cache: _PairwiseDistanceCache | None = None,
) -> list[SelectedCandidate]:
    selected: list[SelectedCandidate] = []
    for record in ranked_hits:
        if all(
            _distance(
                record,
                existing.record,
                metric=distance_metric,
                cache=pairwise_cache,
            )
            >= min_pairwise_distance
            for existing in selected
        ):
            selected.append(
                SelectedCandidate(
                    record=record,
                    selection_rank_reason=(f"greedy_hamming_rank(min_pairwise_distance={min_pairwise_distance})"),
                    distance_to_previous_selected=_distance_to_previous_selected(
                        selected,
                        record,
                        metric=distance_metric,
                        cache=pairwise_cache,
                    ),
                )
            )
        if len(selected) >= max_hits:
            break
    return selected


def select_hits_greedy_hamming(
    hits: list[CandidateHitRecord],
    *,
    max_hits: int,
    distance_metric: str,
    min_pairwise_distance: int,
) -> list[SelectedCandidate]:
    ranked_hits = sorted(hits, key=_full_sort_key)
    pairwise_cache = _PairwiseDistanceCache(metric=distance_metric)
    return _select_ranked_hits_greedy_hamming(
        ranked_hits,
        max_hits=max_hits,
        distance_metric=distance_metric,
        min_pairwise_distance=min_pairwise_distance,
        pairwise_cache=pairwise_cache,
    )


def _tiered_relevance(hits: list[CandidateHitRecord]) -> dict[str, float]:
    relevance_by_hit_id: dict[str, float] = {}
    prior_penalty_vector: tuple[int | float, ...] | None = None
    tier_index = -1
    for record in hits:
        if record.base_penalty_vector != prior_penalty_vector:
            tier_index += 1
            prior_penalty_vector = tuple(record.base_penalty_vector)
        relevance_by_hit_id[record.hit_id] = 1.0 / (1.0 + tier_index)
    return relevance_by_hit_id


def _select_ranked_hits_mmr(
    ranked_hits: list[CandidateHitRecord],
    *,
    max_hits: int,
    distance_metric: str,
    diversity_weight: float,
    min_pairwise_distance: int,
    pairwise_cache: _PairwiseDistanceCache | None = None,
) -> list[SelectedCandidate]:
    if not ranked_hits:
        return []
    pairwise_cache = pairwise_cache or _PairwiseDistanceCache(metric=distance_metric)
    relevance_by_hit_id = _tiered_relevance(ranked_hits)
    mmr_result = select_ranked_mmr(
        item_count=len(ranked_hits),
        k=max_hits,
        alpha=(1.0 - diversity_weight),
        relevance=[relevance_by_hit_id[record.hit_id] for record in ranked_hits],
        similarity_at=lambda left_idx, right_idx: _similarity(
            ranked_hits[left_idx],
            ranked_hits[right_idx],
            metric=distance_metric,
            cache=pairwise_cache,
        ),
        min_distance=float(min_pairwise_distance) if min_pairwise_distance > 0 else None,
        distance_at=lambda left_idx, right_idx: float(
            _distance(
                ranked_hits[left_idx],
                ranked_hits[right_idx],
                metric=distance_metric,
                cache=pairwise_cache,
            )
        ),
    )
    selected: list[SelectedCandidate] = []
    for rank, choice in enumerate(mmr_result.choices):
        record = ranked_hits[choice.idx]
        selected.append(
            SelectedCandidate(
                record=record,
                selection_rank_reason=(
                    "mmr_seed_best_score_tier"
                    if rank == 0
                    else (
                        "mmr_rank("
                        f"relevance={relevance_by_hit_id[record.hit_id]:.3f},"
                        f"utility={choice.utility:.3f},"
                        f"max_similarity={float(choice.max_similarity):.3f})"
                    )
                ),
                distance_to_previous_selected=_distance_to_previous_selected(
                    selected,
                    record,
                    metric=distance_metric,
                    cache=pairwise_cache,
                ),
            )
        )
    return selected


def select_hits_mmr(
    hits: list[CandidateHitRecord],
    *,
    max_hits: int,
    distance_metric: str,
    diversity_weight: float,
    min_pairwise_distance: int,
) -> list[SelectedCandidate]:
    ranked_hits = sorted(hits, key=_full_sort_key)
    pairwise_cache = _PairwiseDistanceCache(metric=distance_metric)
    return _select_ranked_hits_mmr(
        ranked_hits,
        max_hits=max_hits,
        distance_metric=distance_metric,
        diversity_weight=diversity_weight,
        min_pairwise_distance=min_pairwise_distance,
        pairwise_cache=pairwise_cache,
    )


def _selection_non_exhaustive_reason(*, search_truncated: bool, pool_truncated: bool) -> str | None:
    if search_truncated and pool_truncated:
        return "search_bounded_and_pool_bounded"
    if search_truncated:
        return "search_bounded"
    if pool_truncated:
        return "pool_bounded"
    return None


def _policy_limited_hit_count(
    *,
    accepted_pool_size: int,
    selected_hit_count: int,
    max_hits: int,
) -> int:
    expected_upper_bound = min(max_hits, accepted_pool_size)
    if selected_hit_count >= expected_upper_bound:
        return 0
    return expected_upper_bound - selected_hit_count


def _summarize_selection(
    *,
    selection: SearchSelectionSpec,
    search_settings: SearchSettingsSpec,
    accepted_candidate_count: int,
    accepted_pool_summary: AcceptedPoolSummary,
    selected_hits: list[SelectedCandidate],
    search_truncated: bool,
    selection_policy_defaulted: bool,
    pairwise_cache: _PairwiseDistanceCache | None = None,
) -> SolveSelectionSummary:
    policy_limited_hit_count = _policy_limited_hit_count(
        accepted_pool_size=accepted_pool_summary.final_size,
        selected_hit_count=len(selected_hits),
        max_hits=search_settings.max_hits,
    )
    return SolveSelectionSummary(
        policy=selection.policy,
        distance_metric=selection.distance_metric,
        diversity_weight=selection.diversity_weight,
        max_hits=search_settings.max_hits,
        pool_size=accepted_pool_summary.pool_size,
        accepted_candidate_count=accepted_candidate_count,
        accepted_pool_size=accepted_pool_summary.final_size,
        accepted_pool_admitted_count=accepted_pool_summary.admitted_count,
        accepted_pool_rejected_count=accepted_pool_summary.rejected_count,
        accepted_pool_truncated=accepted_pool_summary.truncated,
        accepted_pool_worst_score_at_close=(
            list(accepted_pool_summary.worst_score_at_close)
            if accepted_pool_summary.worst_score_at_close is not None
            else None
        ),
        search_truncated=search_truncated,
        selected_hit_count=len(selected_hits),
        selected_hit_ids=[item.record.hit_id for item in selected_hits],
        selection_policy_defaulted=selection_policy_defaulted,
        selection_pool_non_exhaustive_reason=_selection_non_exhaustive_reason(
            search_truncated=search_truncated,
            pool_truncated=accepted_pool_summary.truncated,
        ),
        policy_limited_hit_count=policy_limited_hit_count,
        policy_underfilled=policy_limited_hit_count > 0,
        policy_underfilled_reason=(
            "selection_policy_constraints_filtered_pool" if policy_limited_hit_count > 0 else None
        ),
        pairwise_distance_summary=_pairwise_distance_summary(
            [item.record for item in selected_hits],
            metric=selection.distance_metric,
            cache=pairwise_cache,
        ),
    )


def select_hits(
    *,
    accepted_pool: AcceptedCandidatePool,
    search_settings: SearchSettingsSpec,
    accepted_candidate_count: int,
    search_truncated: bool,
) -> SelectionOutcome:
    selection = search_settings.selection
    ranked_hits = accepted_pool.ranked_hits()
    pairwise_cache = _PairwiseDistanceCache(metric=selection.distance_metric)
    if selection.policy == "score_only":
        selected_hits = _select_ranked_hits_score_only(
            ranked_hits,
            max_hits=search_settings.max_hits,
            distance_metric=selection.distance_metric,
            pairwise_cache=pairwise_cache,
        )
    elif selection.policy == "mmr":
        assert selection.diversity_weight is not None
        selected_hits = _select_ranked_hits_mmr(
            ranked_hits,
            max_hits=search_settings.max_hits,
            distance_metric=selection.distance_metric,
            diversity_weight=selection.diversity_weight,
            min_pairwise_distance=selection.min_pairwise_distance,
            pairwise_cache=pairwise_cache,
        )
    else:
        selected_hits = _select_ranked_hits_greedy_hamming(
            ranked_hits,
            max_hits=search_settings.max_hits,
            distance_metric=selection.distance_metric,
            min_pairwise_distance=selection.min_pairwise_distance,
            pairwise_cache=pairwise_cache,
        )
    pool_summary = accepted_pool.summary()
    return SelectionOutcome(
        accepted_pool=ranked_hits,
        selected_hits=selected_hits,
        pool_summary=pool_summary,
        summary=_summarize_selection(
            selection=selection,
            search_settings=search_settings,
            accepted_candidate_count=accepted_candidate_count,
            accepted_pool_summary=pool_summary,
            selected_hits=selected_hits,
            search_truncated=search_truncated,
            selection_policy_defaulted=search_settings.selection_policy_defaulted,
            pairwise_cache=pairwise_cache,
        ),
    )
