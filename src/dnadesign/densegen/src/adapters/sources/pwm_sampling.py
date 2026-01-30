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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ...config import PWMMiningConfig, PWMSamplingConfig, PWMSelectionConfig, PWMSelectionTierWidening
from ...core.artifacts.ids import hash_candidate_id
from ...core.score_tiers import score_tier_counts
from ...core.stage_a_constants import FIMO_REPORT_THRESH
from .stage_a_diversity import _diversity_summary
from .stage_a_progress import _format_stage_a_milestone, _PwmSamplingProgress
from .stage_a_selection import (
    _collapse_by_core_identity,
    _core_sequence,
    _pwm_tolerant_weights,
    _select_by_mmr,
    _select_diversity_baseline_candidates,
    _select_diversity_candidate_pool,
    _select_diversity_global_candidates,
    _select_diversity_upper_bound_candidates,
)

SMOOTHING_ALPHA = 1e-6
SCORE_HIST_BINS = 60
log = logging.getLogger(__name__)
_BASES = np.array(["A", "C", "G", "T"])
_SAFE_LABEL_RE = None


def _safe_label(text: str) -> str:
    global _SAFE_LABEL_RE
    if _SAFE_LABEL_RE is None:
        import re

        _SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")
    cleaned = _SAFE_LABEL_RE.sub("_", str(text).strip())
    return cleaned or "motif"


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


@dataclass(frozen=True)
class FimoCandidate:
    seq: str
    score: float
    start: int
    stop: int
    strand: str
    matched_sequence: Optional[str] = None


def _evaluate_tier_target(*, n_sites: int, target_tier_fraction: float, eligible_unique: int) -> tuple[int, bool]:
    if target_tier_fraction <= 0 or target_tier_fraction > 1:
        raise ValueError("target_tier_fraction must be in (0, 1].")
    required_unique = int(np.ceil(float(n_sites) / float(target_tier_fraction)))
    return required_unique, int(eligible_unique) >= required_unique


def _write_candidate_records(
    records: list[dict],
    *,
    debug_output_dir: Path,
    debug_label: str,
    motif_id: str,
    motif_hash: str | None = None,
) -> Path:
    suffix = ""
    if motif_hash:
        suffix = f"__{_safe_label(motif_hash[:10])}"
    safe_label = f"{_safe_label(debug_label or motif_id)}{suffix}"
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    path = debug_output_dir / f"candidates__{safe_label}.parquet"
    df = pd.DataFrame(records)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            if "candidate_id" not in existing.columns or "candidate_id" not in df.columns:
                raise ValueError(
                    f"Candidate append requires candidate_id in {path}. "
                    "Clear outputs/pools/candidates or use --fresh to reset."
                )
            if set(existing.columns) != set(df.columns):
                raise ValueError(
                    f"Candidate schema mismatch for {path}. Clear outputs/pools/candidates or use --fresh to reset."
                )
            df = df[existing.columns]
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["candidate_id"], keep="last")
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(f"Failed to append candidate records to {path}") from exc
    df.to_parquet(path, index=False)
    return path


@dataclass(frozen=True)
class PWMMotif:
    motif_id: str
    matrix: List[dict[str, float]]  # per-position A/C/G/T probabilities
    background: dict[str, float]
    log_odds: Optional[List[dict[str, float]]] = None


@dataclass(frozen=True)
class PWMSamplingSummary:
    input_name: Optional[str]
    regulator: str
    backend: str
    pwm_consensus: Optional[str]
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
    eligible_tier_counts: Optional[List[int]]
    retained_tier_counts: Optional[List[int]]
    tier0_score: Optional[float]
    tier1_score: Optional[float]
    tier2_score: Optional[float]
    eligible_score_hist_edges: Optional[List[float]] = None
    eligible_score_hist_counts: Optional[List[int]] = None
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
    diversity_nearest_distance_mean: Optional[float] = None
    diversity_nearest_distance_min: Optional[float] = None
    diversity_nearest_similarity_mean: Optional[float] = None
    diversity: Optional[dict[str, object]] = None
    mining_audit: Optional[dict[str, object]] = None
    padding_audit: Optional[dict[str, object]] = None


def normalize_background(bg: Optional[dict[str, float]]) -> dict[str, float]:
    if not bg:
        return {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    total = sum(bg.values())
    if total <= 0:
        raise ValueError("Background frequencies must sum to > 0.")
    return {k: float(v) / total for k, v in bg.items()}


def build_log_odds(
    matrix: List[dict[str, float]],
    background: dict[str, float],
    *,
    smoothing_alpha: float = SMOOTHING_ALPHA,
) -> List[dict[str, float]]:
    # Alignment (3): smooth PWM probabilities to avoid -inf log-odds on zeros.
    bg = normalize_background(background)
    log_odds: List[dict[str, float]] = []
    for row in matrix:
        lod_row: dict[str, float] = {}
        for base in ("A", "C", "G", "T"):
            p = float(row.get(base, 0.0))
            b = float(bg.get(base, 0.0))
            if smoothing_alpha > 0.0:
                p = (1.0 - smoothing_alpha) * p + smoothing_alpha * b
            if p <= 0.0 or b <= 0.0:
                lod_row[base] = float("-inf")
            else:
                lod_row[base] = float(np.log(p / b))
        log_odds.append(lod_row)
    return log_odds


def score_sequence(
    seq: str,
    matrix: List[dict[str, float]],
    *,
    log_odds: Optional[List[dict[str, float]]] = None,
    background: Optional[dict[str, float]] = None,
) -> float:
    score = 0.0
    if log_odds is None:
        if background is None:
            background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        log_odds = build_log_odds(matrix, background)
    for base, probs in zip(seq, log_odds):
        lod = probs.get(base, float("-inf"))
        if lod == float("-inf"):
            return float("-inf")
        score += float(lod)
    return score


def _summarize_lengths(lengths: Sequence[int]) -> tuple[Optional[int], Optional[float], Optional[float], Optional[int]]:
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


def _tail_unique_slope(
    generated_by_batch: Sequence[int],
    unique_by_batch: Sequence[int],
    *,
    window: int = 5,
) -> dict[str, object] | None:
    if not generated_by_batch or len(generated_by_batch) < 2:
        return None
    if len(generated_by_batch) != len(unique_by_batch):
        raise ValueError("Generated/unique batch lengths must match for tail slope.")
    window = min(int(window), len(generated_by_batch))
    if window < 2:
        return None
    start_idx = max(0, len(generated_by_batch) - window)
    delta_gen = int(generated_by_batch[-1]) - int(generated_by_batch[start_idx])
    delta_unique = int(unique_by_batch[-1]) - int(unique_by_batch[start_idx])
    if delta_gen <= 0:
        return None
    return {
        "unique_slope": float(delta_unique) / float(delta_gen),
        "unique_slope_window": int(window),
        "unique_slope_generated": int(delta_gen),
    }


def _background_cdf(probs: dict[str, float]) -> np.ndarray:
    bases = ["A", "C", "G", "T"]
    weights = np.array([float(probs.get(b, 0.0)) for b in bases], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Background frequencies must sum to > 0.")
    weights = weights / total
    return np.cumsum(weights)


def _matrix_cdf(matrix: List[dict[str, float]]) -> np.ndarray:
    arr = _matrix_array(matrix)
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("PWM matrix rows must sum to > 0.")
    weights = arr / row_sums
    return np.cumsum(weights, axis=1)


def _pwm_consensus(matrix: Sequence[dict[str, float]]) -> str:
    if not matrix:
        raise ValueError("PWM matrix is required to compute consensus.")
    bases = ("A", "C", "G", "T")
    consensus: list[str] = []
    for row in matrix:
        probs = [float(row.get(base, 0.0)) for base in bases]
        total = float(sum(probs))
        if total <= 0:
            raise ValueError("PWM probabilities must sum to > 0 to compute consensus.")
        max_val = max(probs)
        best_idx = probs.index(max_val)
        consensus.append(bases[best_idx])
    return "".join(consensus)


def _sample_from_background_cdf(rng: np.random.Generator, cdf: np.ndarray, length: int) -> str:
    idx = np.searchsorted(cdf, rng.random(int(length)))
    return "".join(_BASES[idx])


def _sample_pwm_batch(
    rng: np.random.Generator,
    cdf: np.ndarray,
    *,
    count: int,
) -> list[str]:
    width = int(cdf.shape[0])
    draws = rng.random((int(count), width))
    idx = (draws[:, :, None] <= cdf[None, :, :]).argmax(axis=2)
    return ["".join(_BASES[row]) for row in idx]


def _sample_background_batch(
    rng: np.random.Generator,
    cdf: np.ndarray,
    *,
    count: int,
    length: int,
) -> list[str]:
    draws = rng.random((int(count), int(length)))
    idx = np.searchsorted(cdf, draws)
    return ["".join(_BASES[row]) for row in idx]


def _ranges_overlap(a_start: int, a_stop: int, b_start: int, b_stop: int) -> bool:
    return int(a_start) <= int(b_stop) and int(b_start) <= int(a_stop)


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


def _score_tier_counts(total: int) -> tuple[int, int, int, int]:
    return score_tier_counts(total)


def _assign_score_tiers(ranked: Sequence[tuple[str, float]]) -> list[int]:
    total = len(ranked)
    n0, n1, n2, _n3 = _score_tier_counts(total)
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
    diversity_nearest_distance_mean: Optional[float] = None,
    diversity_nearest_distance_min: Optional[float] = None,
    diversity_nearest_similarity_mean: Optional[float] = None,
    diversity: Optional[dict[str, object]] = None,
    mining_audit: Optional[dict[str, object]] = None,
    padding_audit: Optional[dict[str, object]] = None,
    pwm_consensus: Optional[str] = None,
    input_name: Optional[str] = None,
    regulator: Optional[str] = None,
    backend: Optional[str] = None,
) -> PWMSamplingSummary:
    lengths = [len(seq) for seq in retained]
    min_len, median_len, mean_len, max_len = _summarize_lengths(lengths)
    score_min, score_median, score_mean, score_max = _summarize_scores(retained_scores or [])
    return PWMSamplingSummary(
        input_name=input_name,
        regulator=str(regulator or ""),
        backend=str(backend or ""),
        pwm_consensus=str(pwm_consensus) if pwm_consensus is not None else None,
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
        diversity_nearest_distance_mean=float(diversity_nearest_distance_mean)
        if diversity_nearest_distance_mean is not None
        else None,
        diversity_nearest_distance_min=float(diversity_nearest_distance_min)
        if diversity_nearest_distance_min is not None
        else None,
        diversity_nearest_similarity_mean=float(diversity_nearest_similarity_mean)
        if diversity_nearest_similarity_mean is not None
        else None,
        diversity=dict(diversity) if diversity is not None else None,
        mining_audit=dict(mining_audit) if mining_audit is not None else None,
        padding_audit=dict(padding_audit) if padding_audit is not None else None,
    )


def sample_sequence_from_background(rng: np.random.Generator, probs: dict[str, float], length: int) -> str:
    return _sample_from_background_cdf(rng, _background_cdf(probs), int(length))


def sample_sequence_from_pwm(rng: np.random.Generator, matrix: List[dict[str, float]]) -> str:
    cdf = _matrix_cdf(matrix)
    return _sample_pwm_batch(rng, cdf, count=1)[0]


def _matrix_array(matrix: List[dict[str, float]]) -> np.ndarray:
    rows = []
    for row in matrix:
        rows.append(
            [
                float(row.get("A", 0.0)),
                float(row.get("C", 0.0)),
                float(row.get("G", 0.0)),
                float(row.get("T", 0.0)),
            ]
        )
    return np.asarray(rows, dtype=float)


def _information_bits(matrix: np.ndarray) -> float:
    p = matrix + 1e-9
    return float((2 + (p * np.log2(p)).sum(axis=1)).sum())


def _select_pwm_window(
    *,
    matrix: List[dict[str, float]],
    log_odds: List[dict[str, float]],
    length: int,
    strategy: str,
) -> tuple[List[dict[str, float]], List[dict[str, float]], int, float]:
    if length < 1:
        raise ValueError("pwm.sampling.trim_window_length must be >= 1")
    if length > len(matrix):
        raise ValueError(f"pwm.sampling.trim_window_length={length} exceeds motif width {len(matrix)}")
    if strategy != "max_info":
        raise ValueError("pwm.sampling.trim_window_strategy must be 'max_info'")
    if length == len(matrix):
        return matrix, log_odds, 0, 0.0
    arr = _matrix_array(matrix)
    best_start = 0
    best_score = float("-inf")
    last_start = len(matrix) - length
    for start in range(last_start + 1):
        window = arr[start : start + length]
        score = _information_bits(window)
        if score > best_score:
            best_score = score
            best_start = start
    return (
        matrix[best_start : best_start + length],
        log_odds[best_start : best_start + length],
        best_start,
        best_score,
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
        import tempfile

        from .pwm_fimo import (
            aggregate_best_hits,
            build_candidate_records,
            run_fimo,
            write_candidates_fasta,
            write_minimal_meme_motif,
        )

        mining_batch_size = int(mining.batch_size)
        mining_max_seconds = budget_max_seconds
        mining_log_every = int(mining.log_every_batches)
        log.info(
            "FIMO mining config for %s: mode=%s target=%d batch=%d max_seconds=%s max_candidates=%s thresh=%s",
            motif.motif_id,
            budget_mode,
            n_candidates,
            mining_batch_size,
            str(mining_max_seconds) if mining_max_seconds is not None else "-",
            str(budget_max_candidates) if budget_max_candidates is not None else "-",
            FIMO_REPORT_THRESH,
            extra={"suppress_stdout": True},
        )
        debug_path: Optional[Path] = None
        debug_dir = debug_output_dir
        if keep_all_candidates_debug:
            if debug_dir is None:
                tmp_dir = tempfile.mkdtemp(prefix="densegen-fimo-")
                debug_dir = Path(tmp_dir)
                log.warning(
                    "Stage-A PWM sampling keep_all_candidates_debug enabled without outputs_root; "
                    "writing FIMO debug TSVs to %s",
                    debug_dir,
                )
            debug_dir.mkdir(parents=True, exist_ok=True)
            label = _safe_label(debug_label or motif.motif_id)
            debug_path = debug_dir / f"{label}__fimo.tsv"

        def _merge_tsv(existing: list[str], text: str) -> None:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                return
            if not existing:
                existing.extend(lines)
                return
            header_skipped = False
            for ln in lines:
                if ln.lstrip().startswith("#"):
                    continue
                if not header_skipped:
                    header_skipped = True
                    continue
                existing.append(ln)

        def _generate_batch(count: int) -> tuple[list[str], list[int], bool]:
            batch_start = time.monotonic()
            sequences: list[str] = []
            lengths: list[int] = []
            time_limited = False
            target_lengths = []
            for _ in range(count):
                if mining_max_seconds is not None and target_lengths:
                    if (time.monotonic() - batch_start) >= float(mining_max_seconds):
                        time_limited = True
                        break
                target_len = _resolve_length()
                target_lengths.append(int(target_len))
            if not target_lengths:
                return sequences, lengths, time_limited
            lengths.extend(target_lengths)
            if strategy == "background":
                cores = _sample_background_batch(rng, background_cdf, count=len(target_lengths), length=width)
            else:
                cores = _sample_pwm_batch(rng, matrix_cdf, count=len(target_lengths))
            for core, target_len in zip(cores, target_lengths):
                full_seq, left_len = _embed_with_background(core, int(target_len))
                intended_start = int(left_len) + 1
                intended_stop = int(left_len) + int(width)
                intended_core_by_seq[full_seq] = (intended_start, intended_stop)
                core_offset_by_seq[full_seq] = int(left_len)
                sequences.append(full_seq)
            return sequences, lengths, time_limited

        candidates_by_seq: dict[str, FimoCandidate] = {}
        candidates_with_hit = 0
        eligible_raw = 0
        lengths_all: list[int] = []
        generated_total = 0
        time_limited = False
        mining_time_limited = False
        cap_applied = False
        batches = 0
        unique_by_batch: list[int] = []
        generated_by_batch: list[int] = []
        tsv_lines: list[str] = []
        provided_sequences = sequences
        requested_final = int(requested)
        candidate_records: list[dict] | None = [] if keep_all_candidates_debug else None
        intended_core_by_seq = dict(intended_core_by_seq or {})
        core_offset_by_seq = dict(core_offset_by_seq or {})

        def _record_candidate(
            *,
            seq: str,
            hit,
            accepted: bool,
            reject_reason: str | None,
        ) -> None:
            if candidate_records is None:
                return
            resolved_motif_id = motif_hash or motif.motif_id
            candidate_id = hash_candidate_id(
                input_name=input_name,
                motif_id=resolved_motif_id,
                sequence=seq,
                scoring_backend=scoring_backend,
            )
            candidate_records.append(
                {
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "input_name": input_name,
                    "motif_id": resolved_motif_id,
                    "motif_label": motif.motif_id,
                    "scoring_backend": scoring_backend,
                    "sequence": seq,
                    "best_hit_score": None if hit is None else hit.score,
                    "start": None if hit is None else hit.start,
                    "stop": None if hit is None else hit.stop,
                    "strand": None if hit is None else hit.strand,
                    "matched_sequence": None if hit is None else hit.matched_sequence,
                    "accepted": bool(accepted),
                    "selected": False,
                    "reject_reason": reject_reason,
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meme_path = tmp_path / "motif.meme"
            motif_for_fimo = PWMMotif(motif_id=motif.motif_id, matrix=matrix, background=motif.background)
            # FIMO background uses the MEME motif background unless bgfile is provided.
            write_minimal_meme_motif(motif_for_fimo, meme_path)
            if provided_sequences is not None:
                lengths_all = [len(seq) for seq in provided_sequences]
                fasta_path = tmp_path / "candidates.fasta"
                records = build_candidate_records(motif.motif_id, provided_sequences, start_index=0)
                write_candidates_fasta(records, fasta_path)
                rows, raw_tsv = run_fimo(
                    meme_motif_path=meme_path,
                    fasta_path=fasta_path,
                    bgfile=Path(bgfile) if bgfile is not None else None,
                    thresh=FIMO_REPORT_THRESH,
                    norc=True,
                    include_matched_sequence=include_matched_sequence or keep_all_candidates_debug,
                    return_tsv=debug_path is not None,
                )
                if debug_path is not None and raw_tsv is not None:
                    _merge_tsv(tsv_lines, raw_tsv)
                best_hits = aggregate_best_hits(rows)
                for rec_id, seq in records:
                    hit = best_hits.get(rec_id)
                    if hit is None:
                        _record_candidate(
                            seq=seq,
                            hit=None,
                            accepted=False,
                            reject_reason="no_hit",
                        )
                        continue
                    candidates_with_hit += 1
                    if hit.score <= 0:
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            accepted=False,
                            reject_reason="score_non_positive",
                        )
                        continue
                    eligible_raw += 1
                    _record_candidate(
                        seq=seq,
                        hit=hit,
                        accepted=True,
                        reject_reason=None,
                    )
                    prev = candidates_by_seq.get(seq)
                    if (
                        prev is None
                        or hit.score > prev.score
                        or (hit.score == prev.score and (hit.start, hit.stop) < (prev.start, prev.stop))
                    ):
                        candidates_by_seq[seq] = FimoCandidate(
                            seq=seq,
                            score=hit.score,
                            start=hit.start,
                            stop=hit.stop,
                            strand=hit.strand,
                            matched_sequence=hit.matched_sequence,
                        )
                generated_total = len(provided_sequences)
                batches = 1
                unique_by_batch.append(len(candidates_by_seq))
                generated_by_batch.append(generated_total)
                if progress is not None:
                    progress.update(generated=generated_total, accepted=len(candidates_by_seq), force=True)
            else:
                mining_start = time.monotonic()
                target_candidates = int(n_candidates)
                cap_applied = False
                core_by_seq: dict[str, str] = {}
                core_counts: dict[str, int] = {}

                def _eligible_unique_count() -> int:
                    if uniqueness_key == "core":
                        return int(len(core_counts))
                    return int(len(candidates_by_seq))

                def _record_core(seq: str, cand: FimoCandidate) -> None:
                    core = _core_sequence(cand)
                    prev_core = core_by_seq.get(seq)
                    if prev_core is not None:
                        core_counts[prev_core] = core_counts.get(prev_core, 0) - 1
                        if core_counts.get(prev_core) == 0:
                            core_counts.pop(prev_core, None)
                    core_by_seq[seq] = core
                    core_counts[core] = core_counts.get(core, 0) + 1

                while True:
                    if budget_max_seconds is not None and (time.monotonic() - mining_start) >= float(
                        budget_max_seconds
                    ):
                        mining_time_limited = True
                        break
                    if generated_total >= target_candidates:
                        if budget_mode == "fixed_candidates":
                            break
                        if budget_mode == "tier_target":
                            if budget_min_candidates is None or generated_total >= int(budget_min_candidates):
                                if budget_target_tier_fraction is not None:
                                    required_unique = int(np.ceil(float(n_sites) / float(budget_target_tier_fraction)))
                                    if _eligible_unique_count() >= required_unique:
                                        break
                        if budget_max_candidates is not None and generated_total >= int(budget_max_candidates):
                            cap_applied = True
                            break
                        next_target = int(np.ceil(target_candidates * budget_growth_factor))
                        if budget_max_candidates is not None:
                            next_target = min(next_target, int(budget_max_candidates))
                        if next_target <= target_candidates:
                            cap_applied = True
                            break
                        target_candidates = next_target
                        if progress is not None:
                            progress.target = target_candidates
                        continue
                    remaining = int(target_candidates) - generated_total
                    if remaining <= 0:
                        continue
                    batch_target = min(int(mining_batch_size), remaining)
                    sequences, lengths, batch_limited = _generate_batch(batch_target)
                    if batch_limited:
                        time_limited = True
                    if not sequences:
                        break
                    lengths_all.extend(lengths)
                    fasta_path = tmp_path / "candidates.fasta"
                    records = build_candidate_records(motif.motif_id, sequences, start_index=generated_total)
                    write_candidates_fasta(records, fasta_path)
                    rows, raw_tsv = run_fimo(
                        meme_motif_path=meme_path,
                        fasta_path=fasta_path,
                        bgfile=Path(bgfile) if bgfile is not None else None,
                        thresh=FIMO_REPORT_THRESH,
                        norc=True,
                        include_matched_sequence=include_matched_sequence or keep_all_candidates_debug,
                        return_tsv=debug_path is not None,
                    )
                    if debug_path is not None and raw_tsv is not None:
                        _merge_tsv(tsv_lines, raw_tsv)
                    best_hits = aggregate_best_hits(rows)
                    for rec_id, seq in records:
                        hit = best_hits.get(rec_id)
                        if hit is None:
                            _record_candidate(
                                seq=seq,
                                hit=None,
                                accepted=False,
                                reject_reason="no_hit",
                            )
                            continue
                        candidates_with_hit += 1
                        if hit.score <= 0:
                            _record_candidate(
                                seq=seq,
                                hit=hit,
                                accepted=False,
                                reject_reason="score_non_positive",
                            )
                            continue
                        eligible_raw += 1
                        _record_candidate(
                            seq=seq,
                            hit=hit,
                            accepted=True,
                            reject_reason=None,
                        )
                        prev = candidates_by_seq.get(seq)
                        if (
                            prev is None
                            or hit.score > prev.score
                            or (hit.score == prev.score and (hit.start, hit.stop) < (prev.start, prev.stop))
                        ):
                            cand = FimoCandidate(
                                seq=seq,
                                score=hit.score,
                                start=hit.start,
                                stop=hit.stop,
                                strand=hit.strand,
                                matched_sequence=hit.matched_sequence,
                            )
                            candidates_by_seq[seq] = cand
                            if uniqueness_key == "core":
                                _record_core(seq, cand)
                    generated_total += len(sequences)
                    batches += 1
                    unique_by_batch.append(_eligible_unique_count())
                    generated_by_batch.append(generated_total)
                    if progress is not None:
                        progress.update(
                            generated=generated_total,
                            accepted=_eligible_unique_count(),
                            batch_index=batches,
                            batch_total=None,
                        )
                    if mining_log_every > 0 and batches % mining_log_every == 0:
                        log.info(
                            "FIMO mining %s batch %d/%s: generated=%d/%d eligible_unique=%d",
                            motif.motif_id,
                            batches,
                            "-",
                            generated_total,
                            target_candidates,
                            _eligible_unique_count(),
                        )
                    if budget_mode == "tier_target" and budget_target_tier_fraction is not None:
                        if budget_min_candidates is None or generated_total >= int(budget_min_candidates):
                            required_unique = int(np.ceil(float(n_sites) / float(budget_target_tier_fraction)))
                            if _eligible_unique_count() >= required_unique:
                                break
                requested_final = int(target_candidates)

        if debug_path is not None and tsv_lines:
            debug_path.write_text("\n".join(tsv_lines) + "\n")
            log.info("FIMO debug TSV written: %s", debug_path)

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
        tiers = _assign_score_tiers(ranked_pairs)
        rank_by_seq = _ranked_sequence_positions(ranked_pairs)
        tier_by_seq = {cand.seq: tiers[idx] for idx, cand in enumerate(ranked)}
        eligible_tier_counts = [0, 0, 0, 0]
        for tier in tiers:
            eligible_tier_counts[tier] += 1
        selection_meta: dict[str, dict] = {}
        selection_diag: dict = {}
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
            selection_diag = {
                "shortlist_k": None,
                "shortlist_target": None,
                "shortlist_target_met": None,
                "tier_fraction_used": None,
                "tier_limit": None,
            }
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
        if isinstance(selection_diag, dict):
            shortlist_target = selection_diag.get("shortlist_target")
            candidate_pool_size = selection_diag.get("shortlist_k")
            if candidate_pool_size is None:
                candidate_pool_size = selection_diag.get("tier_limit")
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
        n0, n1, n2, _n3 = _score_tier_counts(len(ranked))
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
            meta["selection_tier_fraction_used"] = selection_diag.get("tier_fraction_used")
            meta["selection_tier_limit"] = selection_diag.get("tier_limit")
            meta["shortlist_k"] = selection_diag.get("shortlist_k")
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
                path = _write_candidate_records(
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
                eligible_score_hist_edges=hist_edges,
                eligible_score_hist_counts=hist_counts,
                tier_target_fraction=budget_target_tier_fraction,
                tier_target_required_unique=tier_target_required_unique,
                tier_target_met=tier_target_met,
                selection_policy=selection_policy,
                selection_alpha=selection_alpha if selection_policy == "mmr" else None,
                selection_similarity="weighted_hamming_tolerant" if selection_policy == "mmr" else None,
                selection_shortlist_k=selection_diag.get("shortlist_k"),
                selection_shortlist_min=selection_shortlist_min if selection_policy == "mmr" else None,
                selection_shortlist_factor=selection_shortlist_factor if selection_policy == "mmr" else None,
                selection_shortlist_max=selection_shortlist_max if selection_policy == "mmr" else None,
                selection_shortlist_target=selection_diag.get("shortlist_target"),
                selection_shortlist_target_met=selection_diag.get("shortlist_target_met"),
                selection_tier_fraction_used=selection_diag.get("tier_fraction_used"),
                selection_tier_limit=selection_diag.get("tier_limit"),
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
