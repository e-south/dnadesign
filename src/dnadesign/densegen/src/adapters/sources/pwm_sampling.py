"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/pwm_sampling.py

Shared Stage-A PWM sampling utilities.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ...config import PWMMiningConfig, PWMSamplingConfig, PWMSelectionConfig, PWMSelectionTierWidening
from ...core.score_tiers import resolve_tier_fractions, score_tier_counts
from .stage_a_diversity import _diversity_summary
from .stage_a_metrics import _tail_unique_slope
from .stage_a_mining import mine_pwm_candidates, write_candidate_records
from .stage_a_progress import StageAProgressManager, _format_stage_a_milestone, _PwmSamplingProgress
from .stage_a_sampling_utils import (
    _background_cdf,
    _matrix_cdf,
    _pwm_consensus,
    _ranges_overlap,
    _sample_from_background_cdf,
    _select_pwm_window,
    build_log_odds,
)
from .stage_a_selection import (
    SelectionDiagnostics,
    _collapse_by_core_identity,
    _core_sequence,
    _pwm_tolerant_weights,
    _select_by_mmr,
    _select_diversity_baseline_candidates,
    _select_diversity_candidate_pool,
    _select_diversity_global_candidates,
    _select_diversity_upper_bound_candidates,
)
from .stage_a_summary import (
    PWMSamplingSummary,
    _assign_score_tiers,
    _build_score_hist,
    _build_summary,
    _ranked_sequence_positions,
)
from .stage_a_types import PWMMotif

log = logging.getLogger(__name__)
_BASES = np.array(["A", "C", "G", "T"])


def sampling_kwargs_from_config(sampling: PWMSamplingConfig) -> dict:
    if not isinstance(sampling, PWMSamplingConfig):
        raise ValueError("pwm.sampling config must be a PWMSamplingConfig instance.")
    mining = sampling.mining
    length_cfg = sampling.length
    trimming_cfg = sampling.trimming
    uniqueness_cfg = sampling.uniqueness
    return {
        "strategy": str(sampling.strategy),
        "n_sites": int(sampling.n_sites),
        "mining": mining,
        "bgfile": sampling.bgfile,
        "keep_all_candidates_debug": bool(sampling.keep_all_candidates_debug),
        "include_matched_sequence": bool(sampling.include_matched_sequence),
        "uniqueness_key": str(uniqueness_cfg.key),
        "selection": sampling.selection,
        "length_policy": str(length_cfg.policy),
        "length_range": length_cfg.range,
        "trim_window_length": trimming_cfg.window_length,
        "trim_window_strategy": str(trimming_cfg.window_strategy),
    }


def _evaluate_tier_target(*, n_sites: int, target_tier_fraction: float, eligible_unique: int) -> tuple[int, bool]:
    if target_tier_fraction <= 0 or target_tier_fraction > 1:
        raise ValueError("target_tier_fraction must be in (0, 1].")
    required_unique = int(np.ceil(float(n_sites) / float(target_tier_fraction)))
    return required_unique, int(eligible_unique) >= required_unique


def sample_pwm_sites(
    rng: np.random.Generator,
    motif: PWMMotif,
    *,
    input_name: Optional[str] = None,
    motif_hash: str | None = None,
    run_id: str | None = None,
    strategy: str,
    n_sites: int,
    mining: PWMMiningConfig,
    bgfile: Optional[str | Path] = None,
    keep_all_candidates_debug: bool = False,
    include_matched_sequence: bool = True,
    uniqueness_key: str = "sequence",
    selection: PWMSelectionConfig,
    debug_output_dir: Optional[Path] = None,
    debug_label: Optional[str] = None,
    length_policy: str = "exact",
    length_range: Optional[Sequence[int]] = None,
    trim_window_length: Optional[int] = None,
    trim_window_strategy: str = "max_info",
    progress_manager: StageAProgressManager | None = None,
    return_metadata: bool = False,
    return_summary: bool = False,
) -> Union[
    List[str],
    Tuple[List[str], dict[str, dict]],
    Tuple[List[str], PWMSamplingSummary],
    Tuple[List[str], dict[str, dict], Optional[PWMSamplingSummary]],
]:
    if n_sites <= 0:
        raise ValueError("n_sites must be > 0")
    if not isinstance(mining, PWMMiningConfig):
        raise ValueError("pwm.sampling.mining must be a PWMMiningConfig instance.")
    if not isinstance(selection, PWMSelectionConfig):
        raise ValueError("pwm.sampling.selection must be a PWMSelectionConfig instance.")
    scoring_backend = "fimo"
    uniqueness_key = str(uniqueness_key or "sequence").lower()
    if uniqueness_key not in {"sequence", "core"}:
        raise ValueError(f"Stage-A PWM sampling uniqueness.key must be 'sequence' or 'core', got '{uniqueness_key}'.")
    if keep_all_candidates_debug and run_id is None:
        raise ValueError("Stage-A PWM sampling keep_all_candidates_debug requires run_id to be set.")
    if strategy == "consensus" and n_sites != 1:
        raise ValueError("Stage-A PWM sampling strategy 'consensus' requires n_sites=1")

    width = len(motif.matrix)
    if width <= 0:
        raise ValueError(f"PWM motif '{motif.motif_id}' has zero width.")
    if length_policy not in {"exact", "range"}:
        raise ValueError(f"Unsupported pwm length.policy: {length_policy}")
    log_odds = motif.log_odds or build_log_odds(motif.matrix, motif.background)
    window_label = "full"
    if trim_window_length is not None:
        matrix, log_odds, window_start, window_score = _select_pwm_window(
            matrix=motif.matrix,
            log_odds=log_odds,
            length=int(trim_window_length),
            strategy=str(trim_window_strategy),
        )
        width = len(matrix)
        window_label = f"{width}@{window_start}"
        log.debug(
            "Stage-A PWM sampling trimmed motif %s to window length %d (start=%d, score=%.3f).",
            motif.motif_id,
            width,
            window_start,
            window_score,
        )
    else:
        matrix = motif.matrix
    matrix_cdf = _matrix_cdf(matrix)
    background_cdf = _background_cdf(motif.background)
    pwm_consensus = _pwm_consensus(matrix)

    score_label = "best_hit_score"
    length_label = str(length_policy)
    if length_policy == "range" and length_range is not None and len(length_range) == 2:
        length_label = f"{length_policy}({length_range[0]}..{length_range[1]})"

    selection_policy = str(selection.policy or "top_score").lower()
    if selection_policy not in {"top_score", "mmr"}:
        raise ValueError(f"Stage-A selection.policy must be 'top_score' or 'mmr', got '{selection_policy}'.")
    selection_alpha = float(selection.alpha)
    selection_shortlist_min = int(selection.shortlist_min)
    selection_shortlist_factor = int(selection.shortlist_factor)
    selection_shortlist_max = int(selection.shortlist_max) if selection.shortlist_max is not None else None
    selection_tier_widening: Optional[Sequence[float]] = None
    if selection_policy == "mmr":
        selection_alpha = float(selection_alpha)
        if selection_alpha <= 0.0 or selection_alpha > 1.0:
            raise ValueError("selection.alpha must be in (0, 1].")
        if int(selection_shortlist_min) <= 0:
            raise ValueError("selection.shortlist_min must be > 0.")
        if int(selection_shortlist_factor) <= 0:
            raise ValueError("selection.shortlist_factor must be > 0.")
        if selection_shortlist_max is not None and int(selection_shortlist_max) <= 0:
            raise ValueError("selection.shortlist_max must be > 0 when set.")
        if selection_shortlist_max is not None and int(selection_shortlist_max) < int(selection_shortlist_min):
            raise ValueError("selection.shortlist_max must be >= selection.shortlist_min.")
        if selection_shortlist_max is not None and int(selection_shortlist_max) < int(n_sites):
            raise ValueError("selection.shortlist_max must be >= n_sites when selection.policy=mmr.")

    tier_cfg = selection.tier_widening
    if isinstance(tier_cfg, PWMSelectionTierWidening) and tier_cfg.enabled:
        selection_tier_widening = list(tier_cfg.ladder)

    include_matched_sequence = bool(include_matched_sequence)
    tier_fractions = list(resolve_tier_fractions(selection_tier_widening))
    tier_fractions_source = "tier_widening" if selection_tier_widening else "default"

    budget = mining.budget
    budget_mode = str(budget.mode or "fixed_candidates").lower()
    if budget_mode not in {"tier_target", "fixed_candidates"}:
        raise ValueError(
            f"pwm.sampling.mining.budget.mode must be 'tier_target' or 'fixed_candidates', got '{budget_mode}'."
        )
    budget_target_tier_fraction = budget.target_tier_fraction
    budget_candidates = budget.candidates
    budget_max_candidates = budget.max_candidates
    budget_min_candidates = budget.min_candidates
    budget_max_seconds = budget.max_seconds
    budget_growth_factor = float(budget.growth_factor)
    if budget_max_candidates is not None and int(budget_max_candidates) <= 0:
        raise ValueError("pwm.sampling.mining.budget.max_candidates must be > 0 when set.")
    if budget_min_candidates is not None and int(budget_min_candidates) <= 0:
        raise ValueError("pwm.sampling.mining.budget.min_candidates must be > 0 when set.")
    if (
        budget_min_candidates is not None
        and budget_max_candidates is not None
        and int(budget_min_candidates) > int(budget_max_candidates)
    ):
        raise ValueError("pwm.sampling.mining.budget.min_candidates must be <= max_candidates.")
    if budget_max_seconds is not None and float(budget_max_seconds) <= 0:
        raise ValueError("pwm.sampling.mining.budget.max_seconds must be > 0 when set.")
    if budget_growth_factor <= 1.0:
        raise ValueError("pwm.sampling.mining.budget.growth_factor must be > 1.0")
    if budget_mode == "fixed_candidates":
        if budget_candidates is None:
            raise ValueError("pwm.sampling.mining.budget.candidates must be set when mode=fixed_candidates.")
        if int(budget_candidates) <= 0:
            raise ValueError("pwm.sampling.mining.budget.candidates must be > 0.")
    else:
        if budget_target_tier_fraction is None:
            raise ValueError("pwm.sampling.mining.budget.target_tier_fraction is required for mode=tier_target.")
        if float(budget_target_tier_fraction) <= 0 or float(budget_target_tier_fraction) > 1:
            raise ValueError("pwm.sampling.mining.budget.target_tier_fraction must be in (0, 1].")
        if budget_max_candidates is None and budget_max_seconds is None:
            raise ValueError("pwm.sampling.mining.budget.mode=tier_target requires max_candidates or max_seconds.")

    progress_target_fraction = None
    progress_accepted_target = None
    if budget_mode == "tier_target" and budget_target_tier_fraction is not None:
        progress_target_fraction = float(budget_target_tier_fraction)
        progress_accepted_target = int(np.ceil(float(n_sites) / progress_target_fraction))

    def _cap_label(cap_applied: bool, time_limited: bool) -> str:
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

    def _context(length_obs: str, cap_applied: bool, requested: int, generated: int, time_limited: bool) -> dict:
        return {
            "motif_id": motif.motif_id,
            "width": width,
            "strategy": strategy,
            "length_label": length_label,
            "window_label": window_label,
            "length_observed": length_obs,
            "score_label": score_label,
            "n_sites": n_sites,
            "budget_mode": budget_mode,
            "target_tier_fraction": budget_target_tier_fraction,
            "requested_candidates": requested,
            "generated_candidates": generated,
            "cap_applied": cap_applied,
            "cap_label": _cap_label(cap_applied, time_limited),
            "time_limited": time_limited,
            "mining_batch_size": int(mining.batch_size),
            "mining_max_seconds": budget_max_seconds,
            "mining_log_every_batches": int(mining.log_every_batches),
        }

    def _resolve_length() -> int:
        if length_policy == "exact":
            return width
        if length_range is None or len(length_range) != 2:
            raise ValueError("pwm.sampling.length.range must be provided when length.policy=range")
        lo, hi = int(length_range[0]), int(length_range[1])
        if lo <= 0 or hi <= 0:
            raise ValueError("pwm.sampling.length.range values must be > 0")
        if lo > hi:
            raise ValueError("pwm.sampling.length.range must be min <= max")
        if lo < width:
            raise ValueError(f"pwm.sampling.length.range min must be >= motif width ({width}), got {lo}")
        return int(rng.integers(lo, hi + 1))

    def _embed_with_background(seq: str, target_len: int) -> tuple[str, int]:
        if target_len == len(seq):
            return seq, 0
        extra = target_len - len(seq)
        left_len = int(rng.integers(0, extra + 1))
        right_len = extra - left_len
        left = _sample_from_background_cdf(rng, background_cdf, left_len)
        right = _sample_from_background_cdf(rng, background_cdf, right_len)
        return f"{left}{seq}{right}", int(left_len)

    progress: _PwmSamplingProgress | None = None

    def _score_with_fimo(
        *,
        n_candidates: int,
        requested: int,
        sequences: Optional[List[str]] = None,
        intended_core_by_seq: Optional[dict[str, tuple[int, int]]] = None,
        core_offset_by_seq: Optional[dict[str, int]] = None,
    ) -> tuple[List[str], dict[str, dict]]:
        mining_batch_size = int(mining.batch_size)
        mining_max_seconds = budget_max_seconds
        mining_log_every = int(mining.log_every_batches)
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
            mining_batch_size=mining_batch_size,
            mining_log_every=mining_log_every,
            budget_mode=budget_mode,
            budget_growth_factor=budget_growth_factor,
            budget_max_candidates=budget_max_candidates,
            budget_min_candidates=budget_min_candidates,
            budget_max_seconds=budget_max_seconds,
            budget_target_tier_fraction=budget_target_tier_fraction,
            n_candidates=n_candidates,
            requested=requested,
            n_sites=n_sites,
            bgfile=Path(bgfile) if bgfile is not None else None,
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
            provided_sequences=sequences,
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
        requested_final = mining_result.requested_final
        intended_core_by_seq = mining_result.intended_core_by_seq
        core_offset_by_seq = mining_result.core_offset_by_seq

        length_obs = "-"
        if lengths_all:
            length_obs = (
                f"{min(lengths_all)}..{max(lengths_all)}"
                if min(lengths_all) != max(lengths_all)
                else str(lengths_all[0])
            )

        context = _context(
            length_obs,
            cap_applied,
            requested_final,
            generated_total,
            time_limited or mining_time_limited,
        )
        context["mining_batch_size"] = mining_batch_size
        context["mining_max_seconds"] = mining_max_seconds
        context["mining_time_limited"] = mining_time_limited
        ranked = sorted(candidates_by_seq.values(), key=lambda cand: (-cand.score, cand.seq))
        collapsed_by_core_identity = 0
        if uniqueness_key == "core":
            ranked, collapsed_by_core_identity = _collapse_by_core_identity(ranked)
        eligible_unique = len(ranked)
        mining_audit = _tail_unique_slope(generated_by_batch, unique_by_batch, window=5)
        if progress is not None:
            progress.update(
                generated=generated_total,
                accepted=eligible_unique,
                batch_index=batches if batches > 0 else None,
                batch_total=None,
                force=True,
            )
            progress.finish()
        postprocess_start = time.monotonic()
        log.info(
            _format_stage_a_milestone(
                motif_id=motif.motif_id,
                phase="postprocess",
                detail=(
                    f"eligible_unique={eligible_unique} collapsed={collapsed_by_core_identity} "
                    f"selection={selection_policy}"
                ),
            )
        )
        ranked_pairs = [(cand.seq, cand.score) for cand in ranked]
        tiers = _assign_score_tiers(ranked_pairs, fractions=tier_fractions)
        rank_by_seq = _ranked_sequence_positions(ranked_pairs)
        tier_by_seq = {cand.seq: tiers[idx] for idx, cand in enumerate(ranked)}
        eligible_tier_counts = [0, 0, 0, 0]
        for tier in tiers:
            eligible_tier_counts[tier] += 1
        selection_meta: dict[str, dict] = {}
        selection_diag: SelectionDiagnostics | None = None
        if selection_policy == "mmr":
            picked, selection_meta, selection_diag = _select_by_mmr(
                ranked,
                matrix=matrix,
                n_sites=int(n_sites),
                alpha=float(selection_alpha),
                shortlist_min=int(selection_shortlist_min),
                shortlist_factor=int(selection_shortlist_factor),
                shortlist_max=int(selection_shortlist_max) if selection_shortlist_max is not None else None,
                tier_widening=selection_tier_widening,
            )
        else:
            picked = ranked[: int(n_sites)]
            for idx, cand in enumerate(picked):
                selection_meta[cand.seq] = {
                    "selection_rank": idx + 1,
                    "selection_utility": None,
                    "nearest_selected_similarity": None,
                }
            selection_diag = SelectionDiagnostics(
                shortlist_k=0,
                shortlist_target=0,
                shortlist_target_met=False,
                tier_fraction_used=None,
                tier_limit=int(len(ranked)),
                pool_source="eligible_unique",
            )
        retained_tier_counts = [0, 0, 0, 0]
        for cand in picked:
            retained_tier_counts[tier_by_seq[cand.seq]] += 1
        candidate_pool = _select_diversity_candidate_pool(
            ranked,
            selection_policy=selection_policy,
            selection_diag=selection_diag,
        )
        baseline_candidates = _select_diversity_baseline_candidates(
            ranked,
            selection_policy=selection_policy,
            selection_diag=selection_diag,
            n_sites=int(n_sites),
        )
        distance_weights = _pwm_tolerant_weights(matrix)
        upper_bound_candidates = _select_diversity_upper_bound_candidates(
            ranked,
            selection_policy=selection_policy,
            selection_diag=selection_diag,
            n_sites=int(n_sites),
            weights=distance_weights,
        )
        baseline_global_candidates = _select_diversity_global_candidates(ranked, n_sites=int(n_sites))
        baseline_cores = [(_core_sequence(cand)) for cand in baseline_candidates if cand.matched_sequence]
        actual_cores = [(_core_sequence(cand)) for cand in picked if cand.matched_sequence]
        baseline_global_cores = [(_core_sequence(cand)) for cand in baseline_global_candidates if cand.matched_sequence]
        upper_bound_cores = [(_core_sequence(cand)) for cand in upper_bound_candidates if cand.matched_sequence]
        baseline_scores = [float(cand.score) for cand in baseline_candidates]
        actual_scores = [float(cand.score) for cand in picked]
        baseline_global_scores = [float(cand.score) for cand in baseline_global_candidates]
        upper_bound_scores = [float(cand.score) for cand in upper_bound_candidates]
        candidate_pool_size = None
        shortlist_target = None
        if selection_diag is not None:
            shortlist_target = int(selection_diag.shortlist_target)
            candidate_pool_size = int(selection_diag.pool_size())
        if candidate_pool_size is None and candidate_pool:
            candidate_pool_size = len(candidate_pool)
        diversity_max_n = 2500
        log.info(
            _format_stage_a_milestone(
                motif_id=motif.motif_id,
                phase="diversity",
                detail=(
                    f"baseline={len(baseline_cores)} actual={len(actual_cores)} "
                    f"global={len(baseline_global_cores)} cap={diversity_max_n}"
                ),
            )
        )
        diversity_start = time.monotonic()
        diversity = _diversity_summary(
            baseline_cores=baseline_cores,
            actual_cores=actual_cores,
            baseline_scores=baseline_scores,
            actual_scores=actual_scores,
            baseline_global_cores=baseline_global_cores,
            baseline_global_scores=baseline_global_scores,
            upper_bound_cores=upper_bound_cores,
            upper_bound_scores=upper_bound_scores,
            uniqueness_key=uniqueness_key,
            candidate_pool_size=int(candidate_pool_size) if candidate_pool_size is not None else None,
            shortlist_target=int(shortlist_target) if shortlist_target is not None else None,
            label=motif.motif_id,
            max_n=diversity_max_n,
            distance_weights=distance_weights,
        )
        log.info(
            _format_stage_a_milestone(
                motif_id=motif.motif_id,
                phase="diversity complete",
                elapsed=time.monotonic() - diversity_start,
            )
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
                offset = core_offset_by_seq.get(seq)
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
                    "Retained set will spill beyond the target tier.",
                    "Try next: " + "; ".join(suggestions) + ".",
                ]
                log.warning(" ".join(warn_lines))
        n0, n1, n2, _n3 = score_tier_counts(len(ranked), fractions=tier_fractions)
        tier0_score = ranked[n0 - 1].score if n0 > 0 else None
        tier1_score = ranked[n0 + n1 - 1].score if n1 > 0 else None
        tier2_score = ranked[n0 + n1 + n2 - 1].score if n2 > 0 else None
        eligible_scores = [cand.score for cand in ranked]
        hist_edges, hist_counts = _build_score_hist(eligible_scores)
        log.info(
            _format_stage_a_milestone(
                motif_id=motif.motif_id,
                phase="postprocess complete",
                elapsed=time.monotonic() - postprocess_start,
            )
        )
        log.info(
            "FIMO yield for motif %s: eligible_unique=%d retained=%d",
            motif.motif_id,
            eligible_unique,
            len(picked),
            extra={"suppress_stdout": True},
        )
        meta_by_seq: dict[str, dict] = {}
        for cand in picked:
            meta = {
                "best_hit_score": cand.score,
                "rank_within_regulator": rank_by_seq[cand.seq],
                "tier": tier_by_seq[cand.seq],
                "fimo_start": cand.start,
                "fimo_stop": cand.stop,
                "fimo_strand": cand.strand,
            }
            meta["tfbs_core"] = _core_sequence(cand)
            if cand.matched_sequence:
                meta["fimo_matched_sequence"] = cand.matched_sequence
            selection_meta_row = selection_meta.get(cand.seq, {})
            meta["selection_rank"] = selection_meta_row.get("selection_rank")
            meta["selection_utility"] = selection_meta_row.get("selection_utility")
            meta["nearest_selected_similarity"] = selection_meta_row.get("nearest_selected_similarity")
            meta["selection_policy"] = selection_policy
            meta["selection_alpha"] = selection_alpha if selection_policy == "mmr" else None
            meta["selection_similarity"] = "weighted_hamming_tolerant" if selection_policy == "mmr" else None
            meta["selection_shortlist_min"] = selection_shortlist_min if selection_policy == "mmr" else None
            meta["selection_shortlist_factor"] = selection_shortlist_factor if selection_policy == "mmr" else None
            meta["selection_shortlist_max"] = selection_shortlist_max if selection_policy == "mmr" else None
            if selection_diag is not None:
                meta["selection_tier_fraction_used"] = selection_diag.tier_fraction_used
                meta["selection_tier_limit"] = selection_diag.tier_limit
                meta["shortlist_k"] = selection_diag.shortlist_k
                meta["selection_pool_source"] = selection_diag.pool_source
            else:
                meta["selection_tier_fraction_used"] = None
                meta["selection_tier_limit"] = None
                meta["shortlist_k"] = None
                meta["selection_pool_source"] = None
            meta["tier_target_fraction"] = budget_target_tier_fraction
            meta["tier_target_required_unique"] = tier_target_required_unique
            meta["tier_target_met"] = tier_target_met
            meta["tier_target_eligible_unique"] = int(len(ranked))
            meta_by_seq[cand.seq] = meta
        if candidate_records is not None and debug_dir is not None:
            selected_set = {c.seq for c in picked}
            for row in candidate_records:
                if row.get("sequence") in selected_set:
                    row["selected"] = True
                    row["reject_reason"] = None
                elif row.get("accepted"):
                    row["reject_reason"] = "not_selected"
            try:
                path = write_candidate_records(
                    candidate_records,
                    debug_output_dir=debug_dir,
                    debug_label=debug_label or motif.motif_id,
                    motif_id=motif.motif_id,
                    motif_hash=motif_hash,
                )
                log.info("FIMO candidate records written: %s", path)
            except Exception:
                log.warning("Failed to write FIMO candidate records.", exc_info=True)
        nearest_sims = [
            float(meta.get("nearest_selected_similarity"))
            for meta in selection_meta.values()
            if meta.get("selection_rank") is not None
            and int(meta.get("selection_rank")) > 1
            and meta.get("nearest_selected_similarity") is not None
        ]
        diversity_nearest_similarity_mean = None
        diversity_nearest_distance_mean = None
        diversity_nearest_distance_min = None
        if nearest_sims:
            diversity_nearest_similarity_mean = float(np.mean(nearest_sims))
            diversity_nearest_distance_mean = float(np.mean([(1.0 / sim) - 1.0 for sim in nearest_sims if sim > 0]))
            diversity_nearest_distance_min = float(min((1.0 / sim) - 1.0 for sim in nearest_sims if sim > 0))
        summary = None
        if return_summary:
            if selection_diag is None:
                raise ValueError("Stage-A selection diagnostics missing for summary.")
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
                selection_shortlist_k=selection_diag.shortlist_k if selection_diag is not None else None,
                selection_shortlist_min=selection_shortlist_min if selection_policy == "mmr" else None,
                selection_shortlist_factor=selection_shortlist_factor if selection_policy == "mmr" else None,
                selection_shortlist_max=selection_shortlist_max if selection_policy == "mmr" else None,
                selection_shortlist_target=selection_diag.shortlist_target if selection_diag is not None else None,
                selection_shortlist_target_met=(
                    selection_diag.shortlist_target_met if selection_diag is not None else None
                ),
                selection_tier_fraction_used=selection_diag.tier_fraction_used if selection_diag is not None else None,
                selection_tier_limit=selection_diag.tier_limit if selection_diag is not None else None,
                selection_pool_source=selection_diag.pool_source if selection_diag is not None else None,
                diversity_nearest_similarity_mean=diversity_nearest_similarity_mean,
                diversity_nearest_distance_mean=diversity_nearest_distance_mean,
                diversity_nearest_distance_min=diversity_nearest_distance_min,
                diversity=diversity,
                mining_audit=mining_audit,
                padding_audit=padding_audit,
                pwm_consensus=pwm_consensus,
                input_name=input_name,
                regulator=motif.motif_id,
                backend=scoring_backend,
            )
        return [c.seq for c in picked], meta_by_seq, summary

    if strategy == "consensus":
        progress = _PwmSamplingProgress(
            motif_id=motif.motif_id,
            backend=scoring_backend,
            target=1,
            accepted_target=progress_accepted_target,
            stream=sys.stdout,
            tier_fractions=tier_fractions,
            manager=progress_manager,
            target_fraction=progress_target_fraction,
        )
        seq = "".join(max(row.items(), key=lambda kv: kv[1])[0] for row in matrix)
        target_len = _resolve_length()
        full_seq, left_len = _embed_with_background(seq, target_len)
        intended_start = int(left_len) + 1
        intended_stop = int(left_len) + int(width)
        selected, meta, summary = _score_with_fimo(
            n_candidates=1,
            requested=1,
            sequences=[full_seq],
            intended_core_by_seq={full_seq: (intended_start, intended_stop)},
            core_offset_by_seq={full_seq: int(left_len)},
        )
        if return_metadata and return_summary:
            return selected, meta, summary
        if return_metadata:
            return selected, meta
        if return_summary:
            return selected, summary
        return selected

    if budget_mode == "fixed_candidates":
        requested_candidates = max(1, int(budget_candidates))
    else:
        base_target = int(mining.batch_size)
        if budget_min_candidates is not None:
            base_target = max(int(base_target), int(budget_min_candidates))
        requested_candidates = max(1, int(base_target))
        if budget_max_candidates is not None:
            requested_candidates = min(requested_candidates, int(budget_max_candidates))
    n_candidates = max(1, int(requested_candidates))
    progress = _PwmSamplingProgress(
        motif_id=motif.motif_id,
        backend=scoring_backend,
        target=requested_candidates,
        accepted_target=progress_accepted_target,
        stream=sys.stdout,
        tier_fractions=tier_fractions,
        manager=progress_manager,
        target_fraction=progress_target_fraction,
    )
    selected, meta, summary = _score_with_fimo(
        requested=requested_candidates,
        n_candidates=n_candidates,
    )
    if return_metadata and return_summary:
        return selected, meta, summary
    if return_metadata:
        return selected, meta
    if return_summary:
        return selected, summary
    return selected
