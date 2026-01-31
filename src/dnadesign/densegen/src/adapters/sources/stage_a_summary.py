"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_summary.py

Stage-A PWM sampling summary helpers and manifest-facing dataclasses.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ...core.score_tiers import score_tier_counts
from ...core.stage_a_constants import SCORE_HIST_BINS
from .stage_a_metrics import DiversitySummary


@dataclass(frozen=True)
class PWMSamplingSummary:
    input_name: Optional[str]
    regulator: str
    backend: str
    pwm_consensus: Optional[str]
    pwm_consensus_iupac: Optional[str]
    uniqueness_key: Optional[str]
    collapsed_by_core_identity: Optional[int]
    generated: int
    target: int
    target_sites: Optional[int]
    candidates_with_hit: Optional[int]
    eligible_raw: Optional[int]
    eligible_unique: int
    retained: int
    retained_len_min: Optional[int]
    retained_len_median: Optional[float]
    retained_len_mean: Optional[float]
    retained_len_max: Optional[int]
    retained_score_min: Optional[float]
    retained_score_median: Optional[float]
    retained_score_mean: Optional[float]
    retained_score_max: Optional[float]
    eligible_tier_counts: Optional[list[int]]
    retained_tier_counts: Optional[list[int]]
    tier0_score: Optional[float]
    tier1_score: Optional[float]
    tier2_score: Optional[float]
    tier_fractions: Optional[list[float]]
    tier_fractions_source: Optional[str]
    eligible_score_hist_edges: Optional[list[float]] = None
    eligible_score_hist_counts: Optional[list[int]] = None
    tier_target_fraction: Optional[float] = None
    tier_target_required_unique: Optional[int] = None
    tier_target_met: Optional[bool] = None
    selection_policy: Optional[str] = None
    selection_alpha: Optional[float] = None
    selection_similarity: Optional[str] = None
    selection_shortlist_k: Optional[int] = None
    selection_shortlist_min: Optional[int] = None
    selection_shortlist_factor: Optional[int] = None
    selection_shortlist_max: Optional[int] = None
    selection_shortlist_target: Optional[int] = None
    selection_shortlist_target_met: Optional[bool] = None
    selection_tier_fraction_used: Optional[float] = None
    selection_tier_limit: Optional[int] = None
    selection_pool_source: Optional[str] = None
    diversity_nearest_distance_mean: Optional[float] = None
    diversity_nearest_distance_min: Optional[float] = None
    diversity_nearest_similarity_mean: Optional[float] = None
    diversity: Optional[DiversitySummary] = None
    mining_audit: Optional[dict[str, object]] = None
    padding_audit: Optional[dict[str, object]] = None
    pwm_max_score: Optional[float] = None


def _summarize_lengths(
    lengths: Sequence[int],
) -> tuple[Optional[int], Optional[float], Optional[float], Optional[int]]:
    if not lengths:
        return None, None, None, None
    arr = np.asarray(lengths, dtype=float)
    return int(arr.min()), float(np.median(arr)), float(arr.mean()), int(arr.max())


def _summarize_scores(
    scores: Sequence[float],
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not scores:
        return None, None, None, None
    arr = np.asarray(scores, dtype=float)
    return float(arr.min()), float(np.median(arr)), float(arr.mean()), float(arr.max())


def _rank_scored_sequences(scored: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    best_by_seq: dict[str, float] = {}
    for seq, score in scored:
        seq = str(seq)
        val = float(score)
        prev = best_by_seq.get(seq)
        if prev is None or val > prev:
            best_by_seq[seq] = val
    return sorted(best_by_seq.items(), key=lambda item: (-item[1], item[0]))


def _ranked_sequence_positions(ranked: Sequence[tuple[str, float]]) -> dict[str, int]:
    return {seq: idx + 1 for idx, (seq, _score) in enumerate(ranked)}


def _assign_score_tiers(
    ranked: Sequence[tuple[str, float]],
    *,
    fractions: Sequence[float] | None = None,
) -> list[int]:
    total = len(ranked)
    n0, n1, n2, _n3 = score_tier_counts(total, fractions=fractions)
    tiers: list[int] = []
    for idx in range(total):
        if idx < n0:
            tiers.append(0)
        elif idx < n0 + n1:
            tiers.append(1)
        elif idx < n0 + n1 + n2:
            tiers.append(2)
        else:
            tiers.append(3)
    return tiers


def _build_score_hist(
    scores: Sequence[float],
    *,
    bins: int = SCORE_HIST_BINS,
) -> tuple[list[float], list[int]]:
    vals = [float(v) for v in scores if v is not None]
    if not vals:
        return [], []
    lo = min(vals)
    hi = max(vals)
    lo = min(lo, 0.0)
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, num=int(bins) + 1)
    counts, _ = np.histogram(np.asarray(vals, dtype=float), bins=edges)
    return [float(v) for v in edges], [int(v) for v in counts]


def _build_summary(
    *,
    generated: int,
    target: int,
    target_sites: Optional[int],
    candidates_with_hit: Optional[int],
    eligible_raw: Optional[int],
    eligible_unique: Sequence[str],
    retained: Sequence[str],
    retained_scores: Optional[Sequence[float]] = None,
    uniqueness_key: Optional[str] = None,
    collapsed_by_core_identity: Optional[int] = None,
    eligible_tier_counts: Optional[Sequence[int]] = None,
    retained_tier_counts: Optional[Sequence[int]] = None,
    tier0_score: Optional[float] = None,
    tier1_score: Optional[float] = None,
    tier2_score: Optional[float] = None,
    tier_fractions: Optional[Sequence[float]] = None,
    tier_fractions_source: Optional[str] = None,
    eligible_score_hist_edges: Optional[Sequence[float]] = None,
    eligible_score_hist_counts: Optional[Sequence[int]] = None,
    tier_target_fraction: Optional[float] = None,
    tier_target_required_unique: Optional[int] = None,
    tier_target_met: Optional[bool] = None,
    selection_policy: Optional[str] = None,
    selection_alpha: Optional[float] = None,
    selection_similarity: Optional[str] = None,
    selection_shortlist_k: Optional[int] = None,
    selection_shortlist_min: Optional[int] = None,
    selection_shortlist_factor: Optional[int] = None,
    selection_shortlist_max: Optional[int] = None,
    selection_shortlist_target: Optional[int] = None,
    selection_shortlist_target_met: Optional[bool] = None,
    selection_tier_fraction_used: Optional[float] = None,
    selection_tier_limit: Optional[int] = None,
    selection_pool_source: Optional[str] = None,
    diversity_nearest_distance_mean: Optional[float] = None,
    diversity_nearest_distance_min: Optional[float] = None,
    diversity_nearest_similarity_mean: Optional[float] = None,
    diversity: Optional[DiversitySummary] = None,
    mining_audit: Optional[dict[str, object]] = None,
    padding_audit: Optional[dict[str, object]] = None,
    pwm_consensus: Optional[str] = None,
    pwm_consensus_iupac: Optional[str] = None,
    pwm_max_score: Optional[float] = None,
    input_name: Optional[str] = None,
    regulator: Optional[str] = None,
    backend: Optional[str] = None,
) -> PWMSamplingSummary:
    if diversity is not None and not isinstance(diversity, DiversitySummary):
        raise ValueError("Stage-A diversity must be a DiversitySummary instance.")
    lengths = [len(seq) for seq in retained]
    min_len, median_len, mean_len, max_len = _summarize_lengths(lengths)
    score_min, score_median, score_mean, score_max = _summarize_scores(retained_scores or [])
    return PWMSamplingSummary(
        input_name=input_name,
        regulator=str(regulator or ""),
        backend=str(backend or ""),
        pwm_consensus=str(pwm_consensus) if pwm_consensus is not None else None,
        pwm_consensus_iupac=str(pwm_consensus_iupac) if pwm_consensus_iupac is not None else None,
        uniqueness_key=str(uniqueness_key) if uniqueness_key is not None else None,
        collapsed_by_core_identity=int(collapsed_by_core_identity) if collapsed_by_core_identity is not None else None,
        generated=int(generated),
        target=int(target),
        target_sites=int(target_sites) if target_sites is not None else None,
        candidates_with_hit=int(candidates_with_hit) if candidates_with_hit is not None else None,
        eligible_raw=int(eligible_raw) if eligible_raw is not None else None,
        eligible_unique=int(len(eligible_unique)),
        retained=int(len(retained)),
        retained_len_min=min_len,
        retained_len_median=median_len,
        retained_len_mean=mean_len,
        retained_len_max=max_len,
        retained_score_min=score_min,
        retained_score_median=score_median,
        retained_score_mean=score_mean,
        retained_score_max=score_max,
        eligible_tier_counts=list(eligible_tier_counts) if eligible_tier_counts is not None else None,
        retained_tier_counts=list(retained_tier_counts) if retained_tier_counts is not None else None,
        tier0_score=float(tier0_score) if tier0_score is not None else None,
        tier1_score=float(tier1_score) if tier1_score is not None else None,
        tier2_score=float(tier2_score) if tier2_score is not None else None,
        tier_fractions=[float(v) for v in tier_fractions] if tier_fractions is not None else None,
        tier_fractions_source=str(tier_fractions_source) if tier_fractions_source is not None else None,
        eligible_score_hist_edges=list(eligible_score_hist_edges) if eligible_score_hist_edges is not None else None,
        eligible_score_hist_counts=list(eligible_score_hist_counts) if eligible_score_hist_counts is not None else None,
        tier_target_fraction=float(tier_target_fraction) if tier_target_fraction is not None else None,
        tier_target_required_unique=int(tier_target_required_unique)
        if tier_target_required_unique is not None
        else None,
        tier_target_met=bool(tier_target_met) if tier_target_met is not None else None,
        selection_policy=str(selection_policy) if selection_policy is not None else None,
        selection_alpha=float(selection_alpha) if selection_alpha is not None else None,
        selection_similarity=str(selection_similarity) if selection_similarity is not None else None,
        selection_shortlist_k=int(selection_shortlist_k) if selection_shortlist_k is not None else None,
        selection_shortlist_min=int(selection_shortlist_min) if selection_shortlist_min is not None else None,
        selection_shortlist_factor=int(selection_shortlist_factor) if selection_shortlist_factor is not None else None,
        selection_shortlist_max=int(selection_shortlist_max) if selection_shortlist_max is not None else None,
        selection_shortlist_target=int(selection_shortlist_target) if selection_shortlist_target is not None else None,
        selection_shortlist_target_met=bool(selection_shortlist_target_met)
        if selection_shortlist_target_met is not None
        else None,
        selection_tier_fraction_used=float(selection_tier_fraction_used)
        if selection_tier_fraction_used is not None
        else None,
        selection_tier_limit=int(selection_tier_limit) if selection_tier_limit is not None else None,
        selection_pool_source=str(selection_pool_source) if selection_pool_source is not None else None,
        diversity_nearest_distance_mean=float(diversity_nearest_distance_mean)
        if diversity_nearest_distance_mean is not None
        else None,
        diversity_nearest_distance_min=float(diversity_nearest_distance_min)
        if diversity_nearest_distance_min is not None
        else None,
        diversity_nearest_similarity_mean=float(diversity_nearest_similarity_mean)
        if diversity_nearest_similarity_mean is not None
        else None,
        diversity=diversity,
        mining_audit=dict(mining_audit) if mining_audit is not None else None,
        padding_audit=dict(padding_audit) if padding_audit is not None else None,
        pwm_max_score=float(pwm_max_score) if pwm_max_score is not None else None,
    )
