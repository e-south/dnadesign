"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/stage_a/stage_a_pipeline.py

Stage-A PWM mining, selection, metrics, and summary orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..score_tiers import score_tier_counts
from .stage_a_candidate_store import write_candidate_records, write_fimo_debug_tsv
from .stage_a_diversity import _diversity_summary
from .stage_a_encoding import CoreEncodingStore
from .stage_a_metadata import TFBSMeta
from .stage_a_metrics import _mmr_objective, _tail_unique_slope
from .stage_a_mining import mine_pwm_candidates
from .stage_a_progress import _PwmSamplingProgress, finalize_mining_phase, log_stage_a_milestone
from .stage_a_sampling_utils import (
    _pwm_theoretical_max_score,
    _ranges_overlap,
    build_log_odds,
    select_pwm_window_by_length,
)
from .stage_a_selection import (
    SelectionDiagnostics,
    SelectionMeta,
    _collapse_by_core_identity,
    _core_sequence,
    _pwm_tolerant_weights,
    _score_percentile_norm,
    _select_by_mmr,
    _select_diversity_candidate_pool,
    _select_diversity_global_candidates,
    _select_diversity_top_candidates,
    _select_diversity_upper_bound_candidates,
)
from .stage_a_summary import (
    PWMSamplingSummary,
    _assign_score_tiers,
    _build_score_hist,
    _build_summary,
    _ranked_sequence_positions,
)
from .stage_a_tier_targets import evaluate_tier_target
from .stage_a_types import PWMMotif

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageAPipelineResult:
    sequences: list[str]
    meta_by_seq: dict[str, TFBSMeta]
    summary: Optional[PWMSamplingSummary]


def _evaluate_tier_target(*, n_sites: int, target_tier_fraction: float, eligible_unique: int) -> tuple[int, bool]:
    return evaluate_tier_target(
        n_sites=n_sites,
        target_tier_fraction=target_tier_fraction,
        eligible_unique=eligible_unique,
    )


def _cap_label(
    cap_applied: bool,
    time_limited: bool,
    *,
    budget_max_seconds: float | None,
    budget_max_candidates: int | None,
) -> str:
    cap_label = ""
    if time_limited and budget_max_seconds is not None:
        cap_label = f" (max_seconds={budget_max_seconds})"
    if cap_applied and budget_max_candidates is not None:
        cap_label = (
            f"{cap_label}; max_candidates={budget_max_candidates}"
            if cap_label
            else (f" (max_candidates={budget_max_candidates})")
        )
    return cap_label


def _score_norm_by_tier(
    scores_by_seq: dict[str, float],
    tier_by_seq: dict[str, int],
    *,
    denominator: float | None = None,
    denominator_by_seq: dict[str, float] | None = None,
) -> dict[str, dict[str, float]] | None:
    if not scores_by_seq:
        return None
    if denominator_by_seq is None:
        if denominator is None:
            raise ValueError("score_norm summaries require a denominator.")
        denom = float(denominator)
        if denom <= 0.0:
            raise ValueError("pwm_theoretical_max_score must be > 0 for score_norm summaries.")
    tier_scores: dict[str, list[float]] = {"tier0": [], "tier1": [], "tier2": [], "rest": []}
    for seq, score in scores_by_seq.items():
        tier_idx = int(tier_by_seq.get(seq, 3))
        label = "rest" if tier_idx >= 3 else f"tier{tier_idx}"
        if denominator_by_seq is None:
            score_norm = float(score) / denom
        else:
            denom_val = denominator_by_seq.get(seq)
            if denom_val is None:
                raise ValueError("score_norm denominator missing entry for Stage-A summary.")
            denom_val = float(denom_val)
            if denom_val <= 0.0:
                raise ValueError("score_norm denominator must be > 0 for Stage-A summary.")
            score_norm = float(score) / denom_val
        tier_scores[label].append(score_norm)
    summary: dict[str, dict[str, float]] = {}
    for label, values in tier_scores.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        summary[label] = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }
    return summary or None


def _score_norm_denominator_by_seq(
    sequences: Sequence[str],
    *,
    matrix: Sequence[dict[str, float]],
    background: dict[str, float],
) -> dict[str, float]:
    if not sequences:
        return {}
    matrix_list = list(matrix)
    width = len(matrix_list)
    if width <= 0:
        raise ValueError("PWM matrix must have positive width for score_norm denominators.")
    log_odds = build_log_odds(matrix_list, background, smoothing_alpha=0.0)
    full_max = _pwm_theoretical_max_score(log_odds)
    if full_max <= 0.0:
        raise ValueError("PWM theoretical max must be > 0 for score_norm denominators.")
    by_length: dict[int, float] = {}
    denom_by_seq: dict[str, float] = {}
    for seq in sequences:
        length = len(seq)
        if length <= 0:
            raise ValueError("PWM sampling generated empty sequence; cannot normalize.")
        if length >= width:
            denom = full_max
        else:
            denom = by_length.get(length)
            if denom is None:
                window = select_pwm_window_by_length(
                    matrix=matrix_list,
                    log_odds=log_odds,
                    length=int(length),
                )
                denom = _pwm_theoretical_max_score(window.log_odds)
                if denom <= 0.0:
                    raise ValueError("PWM window theoretical max must be > 0 for score_norm denominators.")
                by_length[int(length)] = float(denom)
        denom_by_seq[str(seq)] = float(denom)
    return denom_by_seq


def _rank_candidates(
    candidates: Sequence[object],
    *,
    rank_by: str,
    score_norm_denominator_by_seq: dict[str, float] | None = None,
) -> list[object]:
    if not candidates:
        return []
    mode = str(rank_by or "score").lower()
    if mode == "score":
        return sorted(candidates, key=lambda cand: (-float(cand.score), str(cand.seq)))
    if mode != "score_norm":
        raise ValueError("selection.rank_by must be 'score' or 'score_norm'.")
    if score_norm_denominator_by_seq is None:
        raise ValueError("score_norm denominators are required for rank_by=score_norm.")
    ranked: list[tuple[float, float, str, object]] = []
    for cand in candidates:
        denom = score_norm_denominator_by_seq.get(str(cand.seq))
        if denom is None:
            raise ValueError("score_norm denominator missing entry for rank_by=score_norm.")
        denom_val = float(denom)
        if denom_val <= 0.0:
            raise ValueError("score_norm denominator must be > 0 for rank_by=score_norm.")
        score_norm = float(cand.score) / denom_val
        ranked.append((score_norm, float(cand.score), str(cand.seq), cand))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in ranked]


def _coerce_pool_max_candidates(policy: str, value: int | None) -> int | None:
    if policy != "mmr":
        return None
    if value is None:
        return None
    return int(value)


def _context(
    *,
    motif: PWMMotif,
    width: int,
    strategy: str,
    length_label: str,
    length_observed: str | None,
    window_label: str,
    score_label: str,
    n_sites: int,
    budget_mode: str,
    budget_target_tier_fraction: float | None,
    requested: int,
    generated: int,
    cap_applied: bool,
    time_limited: bool,
    budget_max_seconds: float | None,
    budget_max_candidates: int | None,
    mining_batch_size: int,
    mining_log_every: int,
    mining_time_limited: bool,
) -> dict[str, object]:
    return {
        "motif_id": motif.motif_id,
        "width": width,
        "strategy": strategy,
        "length_label": length_label,
        "length_observed": length_observed,
        "window_label": window_label,
        "score_label": score_label,
        "n_sites": n_sites,
        "budget_mode": budget_mode,
        "target_tier_fraction": budget_target_tier_fraction,
        "requested_candidates": requested,
        "generated_candidates": generated,
        "cap_applied": cap_applied,
        "cap_label": _cap_label(
            cap_applied,
            time_limited,
            budget_max_seconds=budget_max_seconds,
            budget_max_candidates=budget_max_candidates,
        ),
        "time_limited": time_limited,
        "mining_batch_size": mining_batch_size,
        "mining_max_seconds": budget_max_seconds,
        "mining_log_every_batches": mining_log_every,
        "mining_time_limited": mining_time_limited,
    }


def run_stage_a_pipeline(
    *,
    rng: np.random.Generator,
    motif: PWMMotif,
    matrix: list[dict[str, float]],
    background_cdf: np.ndarray,
    matrix_cdf: np.ndarray,
    width: int,
    strategy: str,
    length_policy: str,
    length_range: Optional[Sequence[int]],
    mining_batch_size: int,
    mining_log_every: int,
    budget_mode: str,
    budget_growth_factor: float,
    budget_max_candidates: Optional[int],
    budget_min_candidates: Optional[int],
    budget_max_seconds: Optional[float],
    budget_target_tier_fraction: Optional[float],
    n_candidates: int,
    requested: int,
    n_sites: int,
    bgfile: Optional[Path],
    keep_all_candidates_debug: bool,
    include_matched_sequence: bool,
    debug_output_dir: Optional[Path],
    debug_label: Optional[str],
    motif_hash: Optional[str],
    input_name: Optional[str],
    run_id: Optional[str],
    scoring_backend: str,
    uniqueness_key: str,
    progress: _PwmSamplingProgress | None,
    selection_policy: str,
    selection_rank_by: str,
    selection_alpha: float,
    selection_pool_min_score_norm: float | None,
    selection_pool_max_candidates: int | None,
    selection_relevance_norm: str | None,
    tier_fractions: Sequence[float],
    tier_fractions_source: str,
    pwm_consensus: str,
    pwm_consensus_iupac: str,
    pwm_consensus_score: float | None,
    pwm_theoretical_max_score: float | None,
    length_label: str,
    window_label: str,
    trim_window_length: Optional[int],
    trim_window_strategy: Optional[str],
    trim_window_start: Optional[int],
    trim_window_score: Optional[float],
    score_label: str,
    return_summary: bool,
    provided_sequences: Optional[List[str]] = None,
    intended_core_by_seq: Optional[dict[str, tuple[int, int]]] = None,
    core_offset_by_seq: Optional[dict[str, int]] = None,
) -> StageAPipelineResult:
    mining_result = mine_pwm_candidates(
        rng=rng,
        motif=motif,
        matrix=matrix,
        background_cdf=background_cdf,
        matrix_cdf=matrix_cdf,
        width=width,
        strategy=strategy,
        length_policy=length_policy,
        length_range=length_range,
        mining_batch_size=int(mining_batch_size),
        mining_log_every=int(mining_log_every),
        budget_mode=budget_mode,
        budget_growth_factor=float(budget_growth_factor),
        budget_max_candidates=budget_max_candidates,
        budget_min_candidates=budget_min_candidates,
        budget_max_seconds=budget_max_seconds,
        budget_target_tier_fraction=budget_target_tier_fraction,
        n_candidates=int(n_candidates),
        requested=int(requested),
        n_sites=int(n_sites),
        bgfile=bgfile,
        keep_all_candidates_debug=keep_all_candidates_debug,
        include_matched_sequence=include_matched_sequence,
        debug_output_dir=debug_output_dir,
        debug_label=debug_label,
        motif_hash=motif_hash,
        input_name=input_name,
        run_id=run_id,
        scoring_backend=scoring_backend,
        uniqueness_key=uniqueness_key,
        progress=progress,
        provided_sequences=provided_sequences,
        intended_core_by_seq=intended_core_by_seq,
        core_offset_by_seq=core_offset_by_seq,
    )
    candidates_by_seq = mining_result.candidates_by_seq
    candidates_with_hit = mining_result.candidates_with_hit
    eligible_raw = mining_result.eligible_raw
    lengths_all = mining_result.lengths_all
    generated_total = mining_result.generated_total
    time_limited = mining_result.time_limited
    mining_time_limited = mining_result.mining_time_limited
    cap_applied = mining_result.cap_applied
    batches = mining_result.batches
    unique_by_batch = mining_result.unique_by_batch
    generated_by_batch = mining_result.generated_by_batch
    candidate_records = mining_result.candidate_records
    debug_dir = mining_result.debug_dir
    debug_tsv_lines = mining_result.debug_tsv_lines
    debug_tsv_path = mining_result.debug_tsv_path
    requested_final = mining_result.requested_final
    intended_core_by_seq = mining_result.intended_core_by_seq
    core_offset_by_seq = mining_result.core_offset_by_seq

    length_obs = "-"
    if lengths_all:
        length_obs = (
            f"{min(lengths_all)}..{max(lengths_all)}" if min(lengths_all) != max(lengths_all) else str(lengths_all[0])
        )

    context = _context(
        motif=motif,
        width=width,
        strategy=strategy,
        length_label=length_label,
        window_label=window_label,
        score_label=score_label,
        n_sites=int(n_sites),
        budget_mode=budget_mode,
        budget_target_tier_fraction=budget_target_tier_fraction,
        requested=int(requested_final),
        generated=int(generated_total),
        cap_applied=cap_applied,
        time_limited=bool(time_limited or mining_time_limited),
        budget_max_seconds=budget_max_seconds,
        budget_max_candidates=budget_max_candidates,
        mining_batch_size=int(mining_batch_size),
        mining_log_every=int(mining_log_every),
        mining_time_limited=bool(mining_time_limited),
        length_observed=length_obs,
    )

    selection_rank_by = str(selection_rank_by or "score")
    candidates = list(candidates_by_seq.values())
    score_norm_denominator_by_seq = (
        _score_norm_denominator_by_seq(
            [cand.seq for cand in candidates],
            matrix=matrix,
            background=motif.background,
        )
        if candidates
        else {}
    )
    collapsed_by_core_identity = 0
    if uniqueness_key == "core":
        collapsed, collapsed_by_core_identity = _collapse_by_core_identity(candidates)
        ranked = _rank_candidates(
            collapsed,
            rank_by=selection_rank_by,
            score_norm_denominator_by_seq=score_norm_denominator_by_seq,
        )
    else:
        ranked = _rank_candidates(
            candidates,
            rank_by=selection_rank_by,
            score_norm_denominator_by_seq=score_norm_denominator_by_seq,
        )
    eligible_unique = len(ranked)
    mining_audit = _tail_unique_slope(generated_by_batch, unique_by_batch, window=5)
    postprocess_start = finalize_mining_phase(
        progress=progress,
        motif_id=motif.motif_id,
        generated=generated_total,
        accepted=eligible_unique,
        batches=batches,
        selection_policy=selection_policy,
        collapsed_by_core_identity=collapsed_by_core_identity,
    )
    ranked_pairs = [(cand.seq, cand.score) for cand in ranked]
    tiers = _assign_score_tiers(ranked_pairs, fractions=tier_fractions)
    rank_by_seq = _ranked_sequence_positions(ranked_pairs)
    tier_by_seq = {cand.seq: tiers[idx] for idx, cand in enumerate(ranked)}
    eligible_score_norm_by_tier: dict[str, dict[str, float]] | None = None
    if ranked:
        scores_by_seq = {cand.seq: float(cand.score) for cand in ranked}
        eligible_score_norm_by_tier = _score_norm_by_tier(
            scores_by_seq,
            tier_by_seq,
            denominator_by_seq=score_norm_denominator_by_seq,
        )
    eligible_tier_counts = [0, 0, 0, 0]
    for tier in tiers:
        eligible_tier_counts[tier] += 1
    selection_meta: dict[str, SelectionMeta] = {}
    selection_diag: SelectionDiagnostics | None = None
    encoding_store = CoreEncodingStore()
    if selection_policy == "mmr":
        picked, selection_meta, selection_diag = _select_by_mmr(
            ranked,
            matrix=matrix,
            background=motif.background,
            n_sites=int(n_sites),
            alpha=float(selection_alpha),
            pool_min_score_norm=selection_pool_min_score_norm,
            pool_max_candidates=selection_pool_max_candidates,
            relevance_norm=selection_relevance_norm or "minmax_raw_score",
            tier_fractions=tier_fractions,
            pwm_theoretical_max_score=pwm_theoretical_max_score,
            score_norm_denominator_by_seq=score_norm_denominator_by_seq,
            rank_by=selection_rank_by,
            encoding_store=encoding_store,
        )
    else:
        picked = ranked[: int(n_sites)]
        for idx, cand in enumerate(picked):
            selection_meta[cand.seq] = SelectionMeta(
                selection_rank=idx + 1,
                selection_utility=None,
                nearest_selected_similarity=None,
            )
        selection_diag = SelectionDiagnostics(
            selection_pool_size_final=int(len(ranked)),
            selection_pool_rung_fraction_used=None,
            selection_pool_min_score_norm_used=None,
            selection_pool_capped=False,
            selection_pool_cap_value=None,
        )
    retained_tier_counts = [0, 0, 0, 0]
    for cand in picked:
        retained_tier_counts[tier_by_seq[cand.seq]] += 1
    candidate_pool = _select_diversity_candidate_pool(
        ranked,
        selection_policy=selection_policy,
        selection_diag=selection_diag,
    )
    top_candidates = _select_diversity_top_candidates(
        ranked,
        selection_policy=selection_policy,
        selection_diag=selection_diag,
        n_sites=int(n_sites),
    )
    distance_weights = _pwm_tolerant_weights(matrix, background=motif.background)
    upper_bound_candidates = _select_diversity_upper_bound_candidates(
        ranked,
        selection_policy=selection_policy,
        selection_diag=selection_diag,
        n_sites=int(n_sites),
        weights=distance_weights,
        encoding_store=encoding_store,
    )
    top_candidates_global = _select_diversity_global_candidates(ranked, n_sites=int(n_sites))
    top_candidates_cores = [(_core_sequence(cand)) for cand in top_candidates if cand.matched_sequence]
    diversified_candidates_cores = [(_core_sequence(cand)) for cand in picked if cand.matched_sequence]
    top_candidates_global_cores = [(_core_sequence(cand)) for cand in top_candidates_global if cand.matched_sequence]
    max_diversity_upper_bound_cores = [
        (_core_sequence(cand)) for cand in upper_bound_candidates if cand.matched_sequence
    ]
    top_candidates_scores = [float(cand.score) for cand in top_candidates]
    diversified_candidates_scores = [float(cand.score) for cand in picked]
    top_candidates_global_scores = [float(cand.score) for cand in top_candidates_global]
    max_diversity_upper_bound_scores = [float(cand.score) for cand in upper_bound_candidates]
    objective_top_candidates = None
    objective_diversified_candidates = None
    if selection_policy == "mmr" and candidate_pool:
        relevance_norm = selection_relevance_norm or "minmax_raw_score"
        score_norm_by_seq: dict[str, float] = {}
        if relevance_norm == "percentile":
            scores_norm_map = _score_percentile_norm([float(cand.score) for cand in candidate_pool])
            score_norm_by_seq = {
                cand.seq: float(scores_norm_map.get(float(cand.score), 1.0)) for cand in candidate_pool
            }
        else:
            if not score_norm_denominator_by_seq:
                raise ValueError("score_norm denominators are required for minmax_raw_score relevance_norm.")
            for cand in candidate_pool:
                denom = score_norm_denominator_by_seq.get(cand.seq)
                if denom is None:
                    raise ValueError("score_norm denominator missing entry for MMR objective.")
                denom_val = float(denom)
                if denom_val <= 0.0:
                    raise ValueError("score_norm denominator must be > 0 for MMR objective.")
                score_norm_by_seq[cand.seq] = float(np.clip(float(cand.score) / denom_val, 0.0, 1.0))
        top_candidates_objective = [cand for cand in top_candidates if cand.matched_sequence]
        diversified_candidates_objective = [cand for cand in picked if cand.matched_sequence]
        top_scores_objective = [float(cand.score) for cand in top_candidates_objective]
        diversified_scores_objective = [float(cand.score) for cand in diversified_candidates_objective]
        top_scores_norm = [score_norm_by_seq[cand.seq] for cand in top_candidates_objective]
        diversified_scores_norm = [score_norm_by_seq[cand.seq] for cand in diversified_candidates_objective]
        objective_top_candidates = _mmr_objective(
            cores=top_candidates_cores,
            scores=top_scores_objective,
            scores_norm=top_scores_norm,
            alpha=float(selection_alpha),
            distance_weights=distance_weights,
            encoding_store=encoding_store,
        )
        objective_diversified_candidates = _mmr_objective(
            cores=diversified_candidates_cores,
            scores=diversified_scores_objective,
            scores_norm=diversified_scores_norm,
            alpha=float(selection_alpha),
            distance_weights=distance_weights,
            encoding_store=encoding_store,
        )
    candidate_pool_size = None
    if selection_diag is not None:
        candidate_pool_size = int(selection_diag.pool_size())
    if candidate_pool_size is None and candidate_pool:
        candidate_pool_size = len(candidate_pool)
    diversity_max_n = 2500
    log_stage_a_milestone(
        motif_id=motif.motif_id,
        phase="diversity",
        detail=(
            f"top={len(top_candidates_cores)} diversified={len(diversified_candidates_cores)} "
            f"global={len(top_candidates_global_cores)} cap={diversity_max_n}"
        ),
    )
    diversity_start = time.monotonic()
    diversity = _diversity_summary(
        top_candidates_cores=top_candidates_cores,
        diversified_candidates_cores=diversified_candidates_cores,
        top_candidates_scores=top_candidates_scores,
        diversified_candidates_scores=diversified_candidates_scores,
        top_candidates_global_cores=top_candidates_global_cores,
        top_candidates_global_scores=top_candidates_global_scores,
        max_diversity_upper_bound_cores=max_diversity_upper_bound_cores,
        max_diversity_upper_bound_scores=max_diversity_upper_bound_scores,
        pwm_theoretical_max_score=pwm_theoretical_max_score,
        objective_top_candidates=objective_top_candidates,
        objective_diversified_candidates=objective_diversified_candidates,
        uniqueness_key=uniqueness_key,
        candidate_pool_size=int(candidate_pool_size) if candidate_pool_size is not None else None,
        label=motif.motif_id,
        max_n=diversity_max_n,
        distance_weights=distance_weights,
        encoding_store=encoding_store,
    )
    if diversity is None:
        raise ValueError("Stage-A diversity metrics missing; ensure core sequences are available.")
    log_stage_a_milestone(
        motif_id=motif.motif_id,
        phase="diversity complete",
        elapsed=time.monotonic() - diversity_start,
    )
    padding_audit = None
    if intended_core_by_seq:
        overlap_total = 0
        overlap_hits = 0
        offset_counts: dict[int, int] = {}
        for seq, cand in candidates_by_seq.items():
            intended = intended_core_by_seq.get(seq)
            if intended is None:
                continue
            offset = core_offset_by_seq.get(seq) if core_offset_by_seq is not None else None
            if offset is None:
                continue
            overlap_total += 1
            if _ranges_overlap(intended[0], intended[1], int(cand.start), int(cand.stop)):
                overlap_hits += 1
            offset_counts[int(offset)] = offset_counts.get(int(offset), 0) + 1
        if overlap_total > 0 and offset_counts:
            bins = sorted(offset_counts)
            counts = [offset_counts[b] for b in bins]
            padding_audit = {
                "best_hit_overlaps_intended_core_fraction": float(overlap_hits) / float(overlap_total),
                "core_offset_histogram": {"bins": bins, "counts": counts},
                "core_offset_n": int(overlap_total),
            }
    if len(ranked) < n_sites:
        msg_lines = [
            (
                "Stage-A PWM sampling shortfall for motif "
                f"'{context.get('motif_id')}' "
                f"(width={context.get('width')}, strategy={context.get('strategy')}, "
                f"length={context.get('length_label')}, window={context.get('window_label')}, "
                f"score={context.get('score_label')})."
            ),
            (
                f"Requested n_sites={context.get('n_sites')} "
                f"-> candidates requested={context.get('requested_candidates')} "
                f"generated={context.get('generated_candidates')}"
                f"{context.get('cap_label')}."
            ),
            (f"Eligible unique sequences={len(ranked)} (need {n_sites})."),
        ]
        if context.get("length_observed"):
            msg_lines.append(f"Observed candidate lengths={context.get('length_observed')}.")
        suggestions = [
            "reduce n_sites",
            "increase mining.budget.max_candidates",
        ]
        if context.get("mining_max_seconds") is not None and context.get("mining_time_limited"):
            suggestions.append("increase mining.budget.max_seconds")
        if context.get("width") is not None and int(context.get("width")) <= 6:
            suggestions.append("try length.policy=range with a longer length.range")
        msg_lines.append("Try next: " + "; ".join(suggestions) + ".")
        log.warning(" ".join(msg_lines))
    tier_target_required_unique = None
    tier_target_met = None
    if budget_mode == "tier_target" and budget_target_tier_fraction is not None:
        tier_target_required_unique, tier_target_met = _evaluate_tier_target(
            n_sites=int(n_sites),
            target_tier_fraction=float(budget_target_tier_fraction),
            eligible_unique=len(ranked),
        )
        if not tier_target_met:
            reason_bits = []
            if cap_applied and budget_max_candidates is not None:
                reason_bits.append(f"max_candidates={int(budget_max_candidates)}")
            if mining_time_limited and budget_max_seconds is not None:
                reason_bits.append(f"max_seconds={float(budget_max_seconds):g}")
            reason_label = ", ".join(reason_bits) if reason_bits else "mining limits"
            suggestions = [
                "increase mining.budget.max_candidates",
                "relax mining.budget.target_tier_fraction",
                "reduce n_sites",
            ]
            if mining_time_limited and budget_max_seconds is not None:
                suggestions.insert(1, "increase mining.budget.max_seconds")
            warn_lines = [
                f"Stage-A tier target unmet for motif '{motif.motif_id}': "
                f"eligible_unique={len(ranked)} < required_unique={tier_target_required_unique} "
                f"for target_tier_fraction={budget_target_tier_fraction}.",
                (
                    "tier_target unmet; reached "
                    f"{reason_label}; retained from best available; consider mining.budget.mode=fixed_candidates."
                ),
                "Retained set will spill beyond the target tier.",
                "Try next: " + "; ".join(suggestions) + ".",
            ]
            if mining_audit is None:
                mining_audit = {}
            mining_audit["tier_target_unmet_reason"] = reason_label
            mining_audit["tier_target_unmet_cap_applied"] = bool(cap_applied)
            mining_audit["tier_target_unmet_time_limited"] = bool(mining_time_limited)
            log.warning(" ".join(warn_lines))
    n0, n1, n2, _n3 = score_tier_counts(len(ranked), fractions=tier_fractions)
    tier0_score = ranked[n0 - 1].score if n0 > 0 else None
    tier1_score = ranked[n0 + n1 - 1].score if n1 > 0 else None
    tier2_score = ranked[n0 + n1 + n2 - 1].score if n2 > 0 else None
    eligible_scores = [cand.score for cand in ranked]
    hist_edges, hist_counts = _build_score_hist(eligible_scores)
    log_stage_a_milestone(
        motif_id=motif.motif_id,
        phase="postprocess complete",
        elapsed=time.monotonic() - postprocess_start,
    )
    log.info(
        "FIMO yield for motif %s: eligible_unique=%d retained=%d",
        motif.motif_id,
        eligible_unique,
        len(picked),
        extra={"suppress_stdout": True},
    )
    meta_by_seq: dict[str, TFBSMeta] = {}
    for cand in picked:
        selection_meta_row = selection_meta.get(cand.seq)
        if selection_meta_row is None:
            raise ValueError(f"Selection metadata missing for retained TFBS {cand.seq}.")
        meta_by_seq[cand.seq] = TFBSMeta(
            best_hit_score=float(cand.score),
            rank_within_regulator=int(rank_by_seq[cand.seq]),
            tier=int(tier_by_seq[cand.seq]),
            fimo_start=int(cand.start),
            fimo_stop=int(cand.stop),
            fimo_strand=str(cand.strand),
            tfbs_core=_core_sequence(cand),
            fimo_matched_sequence=str(cand.matched_sequence) if cand.matched_sequence else None,
            selection_meta=selection_meta_row,
            selection_policy=str(selection_policy),
            selection_alpha=float(selection_alpha) if selection_policy == "mmr" else None,
            selection_similarity="weighted_hamming_tolerant" if selection_policy == "mmr" else None,
            selection_relevance_norm=str(selection_relevance_norm) if selection_policy == "mmr" else None,
            selection_pool_size_final=selection_diag.selection_pool_size_final if selection_diag else None,
            selection_pool_rung_fraction_used=(
                selection_diag.selection_pool_rung_fraction_used if selection_diag else None
            ),
            selection_pool_min_score_norm_used=selection_diag.selection_pool_min_score_norm_used
            if selection_diag
            else None,
            selection_pool_capped=selection_diag.selection_pool_capped if selection_diag else None,
            selection_pool_cap_value=selection_diag.selection_pool_cap_value if selection_diag else None,
            tier_target_fraction=budget_target_tier_fraction,
            tier_target_required_unique=tier_target_required_unique,
            tier_target_met=tier_target_met,
            tier_target_eligible_unique=int(len(ranked)),
        )
    if candidate_records is not None and debug_dir is not None:
        selected_set = {c.seq for c in picked}
        for row in candidate_records:
            if row.get("sequence") in selected_set:
                row["selected"] = True
                row["reject_reason"] = None
            elif row.get("accepted"):
                row["reject_reason"] = "not_selected"
        path = write_candidate_records(
            candidate_records,
            debug_output_dir=debug_dir,
            debug_label=debug_label or motif.motif_id,
            motif_id=motif.motif_id,
            motif_hash=motif_hash,
        )
        log.info("FIMO candidate records written: %s", path)
    if debug_tsv_lines is not None and debug_tsv_path is not None:
        path = write_fimo_debug_tsv(debug_tsv_lines, debug_path=debug_tsv_path)
        log.info("FIMO debug TSV written: %s", path)
    nearest_sims = [
        float(meta.nearest_selected_similarity)
        for meta in selection_meta.values()
        if int(meta.selection_rank) > 1 and meta.nearest_selected_similarity is not None
    ]
    diversity_nearest_similarity_mean = None
    diversity_nearest_distance_mean = None
    diversity_nearest_distance_min = None
    if nearest_sims:
        diversity_nearest_similarity_mean = float(np.mean(nearest_sims))
        diversity_nearest_distance_mean = float(np.mean([(1.0 / sim) - 1.0 for sim in nearest_sims if sim > 0]))
        diversity_nearest_distance_min = float(min((1.0 / sim) - 1.0 for sim in nearest_sims if sim > 0))
    max_observed_score = None
    if ranked:
        max_observed_score = float(max(cand.score for cand in ranked))
    summary = None
    if return_summary:
        if selection_diag is None:
            raise ValueError("Stage-A selection diagnostics missing for summary.")
        motif_width = len(motif.matrix)
        trimmed_width = width
        trim_window_applied = None
        motif_width_value = None
        trimmed_width_value = None
        if trim_window_length is not None:
            motif_width_value = motif_width
            trimmed_width_value = trimmed_width
            trim_window_applied = trimmed_width != motif_width
        summary = _build_summary(
            generated=generated_total,
            target=requested,
            target_sites=n_sites,
            candidates_with_hit=candidates_with_hit,
            eligible_raw=eligible_raw,
            eligible_unique=[cand.seq for cand in ranked],
            retained=[c.seq for c in picked],
            retained_scores=[cand.score for cand in picked],
            uniqueness_key=uniqueness_key,
            collapsed_by_core_identity=collapsed_by_core_identity,
            eligible_tier_counts=eligible_tier_counts,
            retained_tier_counts=retained_tier_counts,
            tier0_score=tier0_score,
            tier1_score=tier1_score,
            tier2_score=tier2_score,
            tier_fractions=tier_fractions,
            tier_fractions_source=tier_fractions_source,
            eligible_score_hist_edges=hist_edges,
            eligible_score_hist_counts=hist_counts,
            tier_target_fraction=budget_target_tier_fraction,
            tier_target_required_unique=tier_target_required_unique,
            tier_target_met=tier_target_met,
            selection_policy=selection_policy,
            selection_alpha=selection_alpha if selection_policy == "mmr" else None,
            selection_similarity="weighted_hamming_tolerant" if selection_policy == "mmr" else None,
            selection_relevance_norm=str(selection_relevance_norm) if selection_policy == "mmr" else None,
            selection_pool_size_final=selection_diag.selection_pool_size_final if selection_diag is not None else None,
            selection_pool_rung_fraction_used=selection_diag.selection_pool_rung_fraction_used
            if selection_diag is not None
            else None,
            selection_pool_min_score_norm_used=selection_diag.selection_pool_min_score_norm_used
            if selection_diag is not None
            else None,
            selection_pool_capped=selection_diag.selection_pool_capped if selection_diag is not None else None,
            selection_pool_cap_value=selection_diag.selection_pool_cap_value if selection_diag is not None else None,
            selection_score_norm_max_raw=selection_diag.selection_score_norm_max_raw
            if selection_diag is not None
            else None,
            selection_score_norm_clipped=selection_diag.selection_score_norm_clipped
            if selection_diag is not None
            else None,
            diversity_nearest_similarity_mean=diversity_nearest_similarity_mean,
            diversity_nearest_distance_mean=diversity_nearest_distance_mean,
            diversity_nearest_distance_min=diversity_nearest_distance_min,
            diversity=diversity,
            eligible_score_norm_by_tier=eligible_score_norm_by_tier,
            mining_audit=mining_audit,
            padding_audit=padding_audit,
            pwm_consensus=pwm_consensus,
            pwm_consensus_iupac=pwm_consensus_iupac,
            pwm_consensus_score=pwm_consensus_score,
            pwm_theoretical_max_score=pwm_theoretical_max_score,
            max_observed_score=max_observed_score,
            input_name=input_name,
            regulator=motif.motif_id,
            backend=scoring_backend,
            motif_width=motif_width_value,
            trim_window_length=trim_window_length,
            trim_window_strategy=trim_window_strategy,
            trim_window_start=trim_window_start,
            trim_window_score=trim_window_score,
            trimmed_width=trimmed_width_value,
            trim_window_applied=trim_window_applied,
        )
    return StageAPipelineResult(
        sequences=[c.seq for c in picked],
        meta_by_seq=meta_by_seq,
        summary=summary,
    )
