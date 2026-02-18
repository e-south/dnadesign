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

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ...config import PWMMiningConfig, PWMSamplingConfig, PWMSelectionConfig
from ...core.score_tiers import resolve_tier_fractions
from ...core.stage_a.stage_a_metadata import TFBSMeta
from ...core.stage_a.stage_a_pipeline import (
    StageAMiningRequest,
    StageASelectionRequest,
    StageASummaryRequest,
    run_stage_a_pipeline,
)
from ...core.stage_a.stage_a_progress import StageAProgressManager, _PwmSamplingProgress
from ...core.stage_a.stage_a_sampling_utils import (
    _background_cdf,
    _matrix_cdf,
    _pwm_consensus,
    _pwm_consensus_iupac,
    _pwm_theoretical_max_score,
    _sample_from_background_cdf,
    _select_pwm_window,
    build_log_odds,
    parse_bgfile,
    score_sequence,
    select_pwm_window_by_length,
)
from ...core.stage_a.stage_a_summary import PWMSamplingSummary
from ...core.stage_a.stage_a_types import PWMMotif

log = logging.getLogger(__name__)
_BASES = np.array(["A", "C", "G", "T"])


def validate_mmr_core_length(
    *,
    motif_id: str,
    motif_width: int,
    selection_policy: str,
    length_policy: str,
    length_range: Optional[Sequence[int]],
    trim_window_length: Optional[int],
) -> None:
    if not isinstance(selection_policy, str) or not selection_policy.strip():
        raise ValueError("pwm.sampling.selection.policy must be a non-empty string.")
    policy = selection_policy.strip().lower()
    if policy != "mmr":
        return
    if not isinstance(length_policy, str) or not length_policy.strip():
        raise ValueError("pwm.sampling.length.policy must be a non-empty string.")
    length_policy_value = length_policy.strip().lower()
    if length_policy_value != "range":
        return
    if length_range is None or len(length_range) != 2:
        raise ValueError("pwm.sampling.length.range must be provided when length.policy=range")
    min_len = int(min(length_range))
    max_len = int(max(length_range))
    if trim_window_length is not None:
        trim_len = int(trim_window_length)
        if trim_len > motif_width:
            raise ValueError(f"pwm.sampling.trimming.window_length={trim_len} exceeds motif width {motif_width}")
        if trim_len > min_len:
            raise ValueError(
                "Stage-A MMR requires trim_window_length <= minimum length when length.policy=range "
                "and min length is below the motif width. "
                f"(motif={motif_id} width={motif_width} range={min_len}..{max_len} trim_window_length={trim_len})"
            )
    if min_len < motif_width and trim_window_length is None:
        raise ValueError(
            "Stage-A MMR requires a fixed trim window when length.policy=range and min length is below "
            "the motif width. "
            f"(motif={motif_id} width={motif_width} range={min_len}..{max_len}) "
            "Set pwm.sampling.trimming.window_length to <= the minimum length or increase length.range."
        )


def enforce_cross_regulator_core_collisions(
    rows: Sequence[dict[str, object]],
    *,
    mode: str,
    input_name: str,
    source_kind: str,
) -> None:
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("pwm.sampling.uniqueness.cross_regulator_core_collisions must be a non-empty string.")
    mode_value = mode.strip().lower()
    if mode_value not in {"allow", "warn", "error"}:
        raise ValueError("pwm.sampling.uniqueness.cross_regulator_core_collisions must be one of: allow, warn, error.")
    if mode_value == "allow":
        return

    regulators_by_core: dict[str, set[str]] = {}
    rows_by_core: dict[str, int] = {}
    for row in rows:
        core_raw = row.get("tfbs_core")
        tf_raw = row.get("tf")
        if core_raw is None or tf_raw is None:
            continue
        core = str(core_raw).strip()
        tf = str(tf_raw).strip()
        if not core or not tf:
            continue
        regulators_by_core.setdefault(core, set()).add(tf)
        rows_by_core[core] = int(rows_by_core.get(core, 0)) + 1

    collisions = [
        (core, sorted(regulators), int(rows_by_core.get(core, 0)))
        for core, regulators in regulators_by_core.items()
        if len(regulators) > 1
    ]
    if not collisions:
        return

    collisions.sort(key=lambda item: (-len(item[1]), -item[2], item[0]))
    preview = [
        {"tfbs_core": core, "regulators": regulators, "rows": rows} for core, regulators, rows in collisions[:10]
    ]
    payload = {
        "input_name": str(input_name),
        "source_kind": str(source_kind),
        "collision_core_count": int(len(collisions)),
        "collision_row_count": int(sum(rows for _, _, rows in collisions)),
        "preview": preview,
    }
    message = "Stage-A run-global core collision across regulators detected: " + json.dumps(payload, sort_keys=True)
    if mode_value == "error":
        raise ValueError(message)
    log.warning(message)


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
        "tier_fractions": sampling.tier_fractions,
        "uniqueness_key": str(uniqueness_cfg.key),
        "cross_regulator_core_collisions": str(uniqueness_cfg.cross_regulator_core_collisions),
        "selection": sampling.selection,
        "length_policy": str(length_cfg.policy),
        "length_range": length_cfg.range,
        "trim_window_length": trimming_cfg.window_length,
        "trim_window_strategy": str(trimming_cfg.window_strategy),
    }


@dataclass(frozen=True)
class StageARequestBundle:
    mining_request: StageAMiningRequest
    selection_request: StageASelectionRequest
    summary_request: StageASummaryRequest


def _build_stage_a_requests(
    *,
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
) -> StageARequestBundle:
    mining_request = StageAMiningRequest(
        matrix=matrix,
        background_cdf=background_cdf,
        matrix_cdf=matrix_cdf,
        width=int(width),
        strategy=str(strategy),
        length_policy=str(length_policy),
        length_range=length_range,
        mining_batch_size=int(mining_batch_size),
        mining_log_every=int(mining_log_every),
        budget_mode=str(budget_mode),
        budget_growth_factor=float(budget_growth_factor),
        budget_max_candidates=budget_max_candidates,
        budget_min_candidates=budget_min_candidates,
        budget_max_seconds=budget_max_seconds,
        budget_target_tier_fraction=budget_target_tier_fraction,
        n_candidates=int(n_candidates),
        requested=int(requested),
        n_sites=int(n_sites),
        bgfile=bgfile,
        keep_all_candidates_debug=bool(keep_all_candidates_debug),
        include_matched_sequence=bool(include_matched_sequence),
        debug_output_dir=debug_output_dir,
        debug_label=debug_label,
        motif_hash=motif_hash,
        input_name=input_name,
        run_id=run_id,
        scoring_backend=str(scoring_backend),
        uniqueness_key=str(uniqueness_key),
        progress=progress,
        provided_sequences=provided_sequences,
        intended_core_by_seq=intended_core_by_seq,
        core_offset_by_seq=core_offset_by_seq,
    )
    selection_request = StageASelectionRequest(
        selection_policy=str(selection_policy),
        selection_rank_by=str(selection_rank_by),
        selection_alpha=float(selection_alpha),
        selection_pool_min_score_norm=selection_pool_min_score_norm,
        selection_pool_max_candidates=selection_pool_max_candidates,
        selection_relevance_norm=selection_relevance_norm,
        tier_fractions=tier_fractions,
        tier_fractions_source=str(tier_fractions_source),
    )
    summary_request = StageASummaryRequest(
        pwm_consensus=str(pwm_consensus),
        pwm_consensus_iupac=str(pwm_consensus_iupac),
        pwm_consensus_score=pwm_consensus_score,
        pwm_theoretical_max_score=pwm_theoretical_max_score,
        length_label=str(length_label),
        window_label=str(window_label),
        trim_window_length=trim_window_length,
        trim_window_strategy=trim_window_strategy,
        trim_window_start=trim_window_start,
        trim_window_score=trim_window_score,
        score_label=str(score_label),
        return_summary=bool(return_summary),
    )
    return StageARequestBundle(
        mining_request=mining_request,
        selection_request=selection_request,
        summary_request=summary_request,
    )


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
    tier_fractions: Optional[Sequence[float]] = None,
    progress_manager: StageAProgressManager | None = None,
    return_metadata: bool = False,
    return_summary: bool = False,
) -> Union[
    List[str],
    Tuple[List[str], dict[str, TFBSMeta]],
    Tuple[List[str], PWMSamplingSummary],
    Tuple[List[str], dict[str, TFBSMeta], Optional[PWMSamplingSummary]],
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
    original_width = width
    if width <= 0:
        raise ValueError(f"PWM motif '{motif.motif_id}' has zero width.")
    if length_policy not in {"exact", "range"}:
        raise ValueError(f"Unsupported pwm length.policy: {length_policy}")
    effective_background = motif.background
    if bgfile is not None:
        effective_background = parse_bgfile(bgfile)
    if effective_background != motif.background:
        motif = PWMMotif(
            motif_id=motif.motif_id,
            matrix=motif.matrix,
            background=effective_background,
            log_odds=motif.log_odds,
        )
    log_odds = build_log_odds(motif.matrix, effective_background, smoothing_alpha=0.0)
    window_label = "full"
    trim_window_start = None
    trim_window_score = None
    if trim_window_length is not None:
        trim_window_length = int(trim_window_length)
        log.info(
            "Stage-A PWM trim configured for motif %s: window_length=%d strategy=%s width=%d",
            motif.motif_id,
            trim_window_length,
            trim_window_strategy,
            original_width,
        )
        matrix, log_odds, window_start, window_score = _select_pwm_window(
            matrix=motif.matrix,
            log_odds=log_odds,
            length=int(trim_window_length),
            strategy=str(trim_window_strategy),
        )
        trim_window_start = int(window_start)
        trim_window_score = float(window_score)
        width = len(matrix)
        window_label = f"{width}@{window_start}"
        if width != original_width:
            log.info(
                "Stage-A PWM trim applied for motif %s: width %d -> %d (start=%d, score=%.3f)",
                motif.motif_id,
                original_width,
                width,
                trim_window_start,
                trim_window_score,
            )
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
    background_cdf = _background_cdf(effective_background)
    pwm_consensus = _pwm_consensus(matrix)
    pwm_consensus_iupac = _pwm_consensus_iupac(matrix)
    pwm_consensus_score = score_sequence(
        pwm_consensus,
        matrix,
        log_odds=log_odds,
        background=effective_background,
    )
    pwm_theoretical_max_score = _pwm_theoretical_max_score(log_odds)

    score_label = "best_hit_score"
    length_label = str(length_policy)
    if length_policy == "range" and length_range is not None and len(length_range) == 2:
        length_label = f"{length_policy}({length_range[0]}..{length_range[1]})"

    selection_policy = str(selection.policy).lower()
    if selection_policy not in {"top_score", "mmr"}:
        raise ValueError(f"Stage-A selection.policy must be 'top_score' or 'mmr', got '{selection_policy}'.")
    selection_alpha = float(selection.alpha)
    selection_pool_min_score_norm = None
    selection_pool_max_candidates = None
    selection_relevance_norm = None
    selection_pool = selection.pool if selection_policy == "mmr" else None
    if selection_pool is not None:
        selection_pool_min_score_norm = selection_pool.min_score_norm
        selection_pool_max_candidates = selection_pool.max_candidates
        selection_relevance_norm = str(selection_pool.relevance_norm)
    selection_rank_by = str(selection.rank_by)
    if selection_policy == "mmr":
        selection_alpha = float(selection_alpha)
        if selection_alpha <= 0.0 or selection_alpha > 1.0:
            raise ValueError("selection.alpha must be in (0, 1].")
        if selection_pool is None:
            raise ValueError("selection.pool must be set when selection.policy=mmr.")
    validate_mmr_core_length(
        motif_id=motif.motif_id,
        motif_width=original_width,
        selection_policy=selection_policy,
        length_policy=length_policy,
        length_range=length_range,
        trim_window_length=trim_window_length,
    )

    include_matched_sequence = bool(include_matched_sequence)
    tier_fractions = list(resolve_tier_fractions(tier_fractions))
    tier_fractions_source = "sampling.tier_fractions" if tier_fractions else "default"

    budget = mining.budget
    budget_mode = str(budget.mode).lower()
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
    mining_batch_size = int(mining.batch_size)
    mining_log_every = int(mining.log_every_batches)

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
        target_len = _resolve_length()
        if length_policy == "range" and length_range is not None and target_len < width:
            window = select_pwm_window_by_length(
                matrix=matrix,
                log_odds=log_odds,
                length=int(target_len),
                strategy=str(trim_window_strategy),
            )
            matrix = window.matrix
            log_odds = window.log_odds
            width = len(matrix)
            if window_label == "full":
                window_label = f"{width}@{window.start}"
            else:
                window_label = f"{window_label}|{width}@{window.start}"
            matrix_cdf = _matrix_cdf(matrix)
            pwm_consensus = _pwm_consensus(matrix)
            pwm_consensus_iupac = _pwm_consensus_iupac(matrix)
            pwm_consensus_score = score_sequence(
                pwm_consensus,
                matrix,
                log_odds=log_odds,
                background=effective_background,
            )
            pwm_theoretical_max_score = _pwm_theoretical_max_score(log_odds)
        seq = str(pwm_consensus)
        full_seq, left_len = _embed_with_background(seq, target_len)
        intended_start = int(left_len) + 1
        intended_stop = int(left_len) + int(width)
        request_bundle = _build_stage_a_requests(
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
            n_candidates=1,
            requested=1,
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
            selection_policy=selection_policy,
            selection_rank_by=selection_rank_by,
            selection_alpha=selection_alpha,
            selection_pool_min_score_norm=selection_pool_min_score_norm,
            selection_pool_max_candidates=selection_pool_max_candidates,
            selection_relevance_norm=selection_relevance_norm,
            tier_fractions=tier_fractions,
            tier_fractions_source=tier_fractions_source,
            pwm_consensus=pwm_consensus,
            pwm_consensus_iupac=pwm_consensus_iupac,
            pwm_consensus_score=pwm_consensus_score,
            pwm_theoretical_max_score=pwm_theoretical_max_score,
            length_label=length_label,
            window_label=window_label,
            trim_window_length=trim_window_length,
            trim_window_strategy=str(trim_window_strategy) if trim_window_length is not None else None,
            trim_window_start=trim_window_start,
            trim_window_score=trim_window_score,
            score_label=score_label,
            return_summary=return_summary,
            provided_sequences=[full_seq],
            intended_core_by_seq={full_seq: (intended_start, intended_stop)},
            core_offset_by_seq={full_seq: int(left_len)},
        )
        result = run_stage_a_pipeline(
            rng=rng,
            motif=motif,
            mining_request=request_bundle.mining_request,
            selection_request=request_bundle.selection_request,
            summary_request=request_bundle.summary_request,
        )
        selected = result.sequences
        meta = result.meta_by_seq
        summary = result.summary
        if return_summary and summary is None:
            raise ValueError("Stage-A summary missing from pipeline result.")
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
    if budget_mode == "fixed_candidates":
        progress_target = int(budget_candidates)
    elif budget_max_candidates is not None:
        progress_target = int(budget_max_candidates)
    elif budget_max_seconds is not None:
        progress_target = 0
    else:
        progress_target = int(requested_candidates)
    progress = _PwmSamplingProgress(
        motif_id=motif.motif_id,
        backend=scoring_backend,
        target=progress_target,
        accepted_target=progress_accepted_target,
        stream=sys.stdout,
        tier_fractions=tier_fractions,
        manager=progress_manager,
        target_fraction=progress_target_fraction,
    )
    request_bundle = _build_stage_a_requests(
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
        requested=requested_candidates,
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
        selection_policy=selection_policy,
        selection_rank_by=selection_rank_by,
        selection_alpha=selection_alpha,
        selection_pool_min_score_norm=selection_pool_min_score_norm,
        selection_pool_max_candidates=selection_pool_max_candidates,
        selection_relevance_norm=selection_relevance_norm,
        tier_fractions=tier_fractions,
        tier_fractions_source=tier_fractions_source,
        pwm_consensus=pwm_consensus,
        pwm_consensus_iupac=pwm_consensus_iupac,
        pwm_consensus_score=pwm_consensus_score,
        pwm_theoretical_max_score=pwm_theoretical_max_score,
        length_label=length_label,
        window_label=window_label,
        trim_window_length=trim_window_length,
        trim_window_strategy=str(trim_window_strategy) if trim_window_length is not None else None,
        trim_window_start=trim_window_start,
        trim_window_score=trim_window_score,
        score_label=score_label,
        return_summary=return_summary,
    )
    result = run_stage_a_pipeline(
        rng=rng,
        motif=motif,
        mining_request=request_bundle.mining_request,
        selection_request=request_bundle.selection_request,
        summary_request=request_bundle.summary_request,
    )
    selected = result.sequences
    meta = result.meta_by_seq
    summary = result.summary
    if return_summary and summary is None:
        raise ValueError("Stage-A summary missing from pipeline result.")
    if return_metadata and return_summary:
        return selected, meta, summary
    if return_metadata:
        return selected, meta
    if return_summary:
        return selected, summary
    return selected
