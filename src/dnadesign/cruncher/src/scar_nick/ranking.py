"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/ranking.py

Ranking and profile-bucket selection for scar-nick candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

from dnadesign.cruncher.scar_nick.models import CandidateRankingContext, ScarNickCandidate

_MISSING_REFERENCE_DISTANCE = 1_000_000


def _profile_bucket_index(profile: str, context: CandidateRankingContext) -> int:
    try:
        return context.target_profile_buckets.index(profile)
    except ValueError:
        return len(context.target_profile_buckets)


def ranking_key(candidate: ScarNickCandidate, context: CandidateRankingContext) -> tuple[object, ...]:
    """Return the deterministic ordering key for one evaluated candidate."""

    rejected = bool(candidate.rejection_reasons)
    gc_value = candidate.gc_fraction if context.reduce_gc_when_tied else 0.0
    middle_hard_tier_sum = (
        candidate.middle_hard_mismatch_tier_sum if context.prefer_lower_middle_hard_mismatch_tier else 0
    )
    hard_tier_sum = candidate.hard_mismatch_tier_sum if context.prefer_lower_hard_mismatch_tier else 0
    worst_hard_tier = candidate.worst_hard_mismatch_tier if context.prefer_lower_hard_mismatch_tier else 0
    reference_distance = (
        candidate.reference_control_distance
        if candidate.reference_control_distance is not None
        else _MISSING_REFERENCE_DISTANCE
    )
    return (
        int(rejected),
        candidate.nick_distance,
        candidate.profile_s3s2s1s0 not in context.target_profile_buckets,
        _profile_bucket_index(candidate.profile_s3s2s1s0, context),
        candidate.middle_hard_count > 1,
        middle_hard_tier_sum,
        hard_tier_sum,
        worst_hard_tier,
        candidate.tnna_flag,
        abs(candidate.effective_disruption - 1.5),
        -candidate.ligation_support,
        candidate.profile_s3s2s1s0,
        gc_value,
        reference_distance,
        candidate.left_base,
        candidate.right_base,
        candidate.nicked_strand or "",
    )


def candidate_sequence_key(candidate: ScarNickCandidate) -> tuple[object, ...]:
    """Return the uniqueness key for one retained scar/profile/enzyme route."""

    release = candidate.release_placement
    nickase = candidate.nickase_placement
    release_key = None
    if release is not None:
        release_key = (
            release.variant_id,
            release.orientation,
            release.recognition_sequence,
            release.recognition_site_start,
            release.recognition_site_end,
            release.top_cut_boundary,
            release.bottom_cut_boundary,
        )
    nickase_key = None
    if nickase is not None:
        nickase_key = (
            nickase.variant_id,
            nickase.orientation,
            nickase.motif_top_5to3,
            nickase.source_site_start,
            nickase.source_site_end,
            nickase.strand,
            nickase.boundary,
            nickase.terminal_boundary,
        )
    return (
        candidate.left_base,
        candidate.right_base,
        candidate.profile_s3s2s1s0,
        candidate.retained_product_sequence,
        candidate.terminal_boundary,
        candidate.nick_boundary,
        release_key,
        nickase_key,
    )


def unique_sequence_candidates(
    candidates: Iterable[ScarNickCandidate],
    *,
    limit: int | None = None,
) -> list[ScarNickCandidate]:
    unique: list[ScarNickCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in candidates:
        key = candidate_sequence_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if limit is not None and len(unique) >= limit:
            break
    return unique


def rank_pair_candidates(
    candidates: Iterable[ScarNickCandidate],
    *,
    context: CandidateRankingContext,
) -> list[ScarNickCandidate]:
    ranked = sorted(candidates, key=lambda candidate: ranking_key(candidate, context))
    return [
        candidate.model_copy(update={"rank": rank, "rank_key": list(ranking_key(candidate, context))})
        for rank, candidate in enumerate(ranked, start=1)
    ]


def select_profile_bucket_candidates(
    candidates: Iterable[ScarNickCandidate],
    *,
    context: CandidateRankingContext,
    limit: int,
) -> list[ScarNickCandidate]:
    pool = list(candidates)
    selected: list[ScarNickCandidate] = []
    seen: set[tuple[object, ...]] = set()

    def add(candidate: ScarNickCandidate) -> None:
        key = candidate_sequence_key(candidate)
        if key in seen:
            return
        seen.add(key)
        selected.append(candidate)

    for bucket in context.target_profile_buckets:
        for candidate in pool:
            if candidate.profile_s3s2s1s0 == bucket:
                add(candidate)
                break
        if len(selected) >= limit:
            return selected[:limit]

    for candidate in pool:
        add(candidate)
        if len(selected) >= limit:
            break
    return selected


__all__ = [
    "candidate_sequence_key",
    "rank_pair_candidates",
    "ranking_key",
    "select_profile_bucket_candidates",
    "unique_sequence_candidates",
]
