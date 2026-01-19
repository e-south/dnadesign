"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/pwm_sampling.py

Shared PWM sampling utilities.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

SMOOTHING_ALPHA = 1e-6
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PWMMotif:
    motif_id: str
    matrix: List[dict[str, float]]  # per-position A/C/G/T probabilities
    background: dict[str, float]
    log_odds: Optional[List[dict[str, float]]] = None


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


def sample_sequence_from_background(rng: np.random.Generator, probs: dict[str, float], length: int) -> str:
    bases = ["A", "C", "G", "T"]
    weights = np.array([probs[b] for b in bases], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(bases), size=length, replace=True, p=weights)
    return "".join(bases[i] for i in idx)


def sample_sequence_from_pwm(rng: np.random.Generator, matrix: List[dict[str, float]]) -> str:
    seq = []
    for probs in matrix:
        bases = ["A", "C", "G", "T"]
        weights = np.array([probs[b] for b in bases], dtype=float)
        weights = weights / weights.sum()
        seq.append(bases[int(rng.choice(len(bases), p=weights))])
    return "".join(seq)


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


def select_by_score(
    candidates: List[Tuple[str, float]],
    *,
    n_sites: int,
    threshold: Optional[float],
    percentile: Optional[float],
    keep_low: bool,
    context: Optional[dict] = None,
) -> List[str]:
    scores = np.array([s for _, s in candidates], dtype=float)
    if threshold is None:
        cutoff = np.percentile(scores, float(percentile))  # type: ignore[arg-type]
    else:
        cutoff = float(threshold)
    if context is not None:
        if threshold is None and percentile is not None:
            context = dict(context)
            context["score_label"] = f"percentile={percentile} (cutoff={cutoff:.4g})"
        elif threshold is not None:
            context = dict(context)
            context["score_label"] = f"threshold={threshold}"
    if keep_low:
        picked = [seq for seq, score in candidates if score <= cutoff]
    else:
        picked = [seq for seq, score in candidates if score >= cutoff]
    unique = list(dict.fromkeys(picked))
    if len(unique) < n_sites:
        unique_total = len({seq for seq, _ in candidates})
        if context is None:
            raise ValueError(
                f"PWM sampling produced {len(unique)} unique sites after filtering; "
                f"need {n_sites}. Adjust thresholds or oversample_factor."
            )
        msg_lines = [
            (
                "PWM sampling failed for motif "
                f"'{context.get('motif_id')}' "
                f"(width={context.get('width')}, strategy={context.get('strategy')}, "
                f"length={context.get('length_label')}, window={context.get('window_label')}, "
                f"score={context.get('score_label')})."
            ),
            (
                f"Requested n_sites={context.get('n_sites')} oversample_factor={context.get('oversample_factor')} "
                f"-> candidates requested={context.get('requested_candidates')} "
                f"generated={context.get('generated_candidates')}"
                f"{context.get('cap_label')}."
            ),
            (f"Unique candidates before filtering={unique_total}, after filtering={len(unique)} (need {n_sites})."),
        ]
        if context.get("length_observed"):
            msg_lines.append(f"Observed candidate lengths={context.get('length_observed')}.")
        suggestions = [
            "reduce n_sites",
            "lower score_percentile (e.g., 90 → 80)",
            "increase oversample_factor",
        ]
        if context.get("cap_applied"):
            suggestions.append("increase max_candidates (cap was hit)")
        if context.get("time_limited"):
            suggestions.append("increase max_seconds (time limit was hit)")
        if context.get("width") is not None and int(context.get("width")) <= 6:
            suggestions.append("try length_policy=range with a longer length_range")
        msg_lines.append("Try next: " + "; ".join(suggestions) + ".")
        raise ValueError(" ".join(msg_lines))
    return unique[:n_sites]


def sample_pwm_sites(
    rng: np.random.Generator,
    motif: PWMMotif,
    *,
    strategy: str,
    n_sites: int,
    oversample_factor: int,
    max_candidates: Optional[int] = None,
    max_seconds: Optional[float] = None,
    score_threshold: Optional[float],
    score_percentile: Optional[float],
    length_policy: str = "exact",
    length_range: Optional[Sequence[int]] = None,
    trim_window_length: Optional[int] = None,
    trim_window_strategy: str = "max_info",
) -> List[str]:
    if n_sites <= 0:
        raise ValueError("n_sites must be > 0")
    if oversample_factor <= 0:
        raise ValueError("oversample_factor must be > 0")
    if max_seconds is not None and float(max_seconds) <= 0:
        raise ValueError("max_seconds must be > 0 when set")
    if (score_threshold is None) == (score_percentile is None):
        raise ValueError("PWM sampling requires exactly one of score_threshold or score_percentile")
    if strategy == "consensus" and n_sites != 1:
        raise ValueError("PWM sampling strategy 'consensus' requires n_sites=1")

    keep_low = strategy == "background"
    width = len(motif.matrix)
    if width <= 0:
        raise ValueError(f"PWM motif '{motif.motif_id}' has zero width.")
    if length_policy not in {"exact", "range"}:
        raise ValueError(f"Unsupported pwm length_policy: {length_policy}")
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
            "PWM sampling trimmed motif %s to window length %d (start=%d, score=%.3f).",
            motif.motif_id,
            width,
            window_start,
            window_score,
        )
    else:
        matrix = motif.matrix

    score_label = f"threshold={score_threshold}" if score_threshold is not None else f"percentile={score_percentile}"
    length_label = str(length_policy)
    if length_policy == "range" and length_range is not None and len(length_range) == 2:
        length_label = f"{length_policy}({length_range[0]}..{length_range[1]})"

    def _select(
        candidates: List[Tuple[str, float]],
        *,
        length_obs: str,
        cap_applied: bool,
        requested: int,
        generated: int,
        time_limited: bool,
    ):
        cap_label = ""
        if cap_applied and max_candidates is not None:
            cap_label = f" (capped by max_candidates={max_candidates})"
        if time_limited and max_seconds is not None:
            cap_label = f"{cap_label}; max_seconds={max_seconds}" if cap_label else f" (max_seconds={max_seconds})"
        return select_by_score(
            candidates,
            n_sites=n_sites,
            threshold=score_threshold,
            percentile=score_percentile,
            keep_low=keep_low,
            context={
                "motif_id": motif.motif_id,
                "width": width,
                "strategy": strategy,
                "length_label": length_label,
                "window_label": window_label,
                "length_observed": length_obs,
                "score_label": score_label,
                "n_sites": n_sites,
                "oversample_factor": oversample_factor,
                "requested_candidates": requested,
                "generated_candidates": generated,
                "cap_applied": cap_applied,
                "cap_label": cap_label,
                "time_limited": time_limited,
            },
        )

    def _resolve_length() -> int:
        if length_policy == "exact":
            return width
        if length_range is None or len(length_range) != 2:
            raise ValueError("pwm.sampling.length_range must be provided when length_policy=range")
        lo, hi = int(length_range[0]), int(length_range[1])
        if lo <= 0 or hi <= 0:
            raise ValueError("pwm.sampling.length_range values must be > 0")
        if lo > hi:
            raise ValueError("pwm.sampling.length_range must be min <= max")
        if lo < width:
            raise ValueError(f"pwm.sampling.length_range min must be >= motif width ({width}), got {lo}")
        return int(rng.integers(lo, hi + 1))

    def _embed_with_background(seq: str, target_len: int) -> str:
        if target_len == len(seq):
            return seq
        extra = target_len - len(seq)
        left_len = int(rng.integers(0, extra + 1))
        right_len = extra - left_len
        left = sample_sequence_from_background(rng, motif.background, left_len)
        right = sample_sequence_from_background(rng, motif.background, right_len)
        return f"{left}{seq}{right}"

    if strategy == "consensus":
        seq = "".join(max(row.items(), key=lambda kv: kv[1])[0] for row in matrix)
        target_len = _resolve_length()
        full_seq = _embed_with_background(seq, target_len)
        score = score_sequence(seq, matrix, log_odds=log_odds, background=motif.background)
        return _select(
            [(full_seq, score)],
            length_obs=str(target_len),
            cap_applied=False,
            requested=1,
            generated=1,
            time_limited=False,
        )

    requested_candidates = max(1, n_sites * oversample_factor)
    n_candidates = requested_candidates
    cap_applied = False
    if max_candidates is not None:
        cap_val = int(max_candidates)
        if cap_val <= 0:
            raise ValueError("max_candidates must be > 0 when set")
        if requested_candidates > cap_val:
            n_candidates = cap_val
            cap_applied = True
            log.warning(
                "PWM sampling capped candidate generation for motif %s: requested=%d max_candidates=%d",
                motif.motif_id,
                requested_candidates,
                cap_val,
            )
    n_candidates = max(1, n_candidates)
    candidates: List[Tuple[str, float]] = []
    lengths: List[int] = []
    start = time.monotonic()
    time_limited = False
    for _ in range(n_candidates):
        if max_seconds is not None and candidates:
            if (time.monotonic() - start) >= float(max_seconds):
                time_limited = True
                break
        target_len = _resolve_length()
        lengths.append(int(target_len))
        if strategy == "background":
            core = sample_sequence_from_background(rng, motif.background, width)
        else:
            core = sample_sequence_from_pwm(rng, matrix)
        full_seq = _embed_with_background(core, target_len)
        candidates.append(
            (
                full_seq,
                score_sequence(core, matrix, log_odds=log_odds, background=motif.background),
            )
        )
    if time_limited:
        log.warning(
            "PWM sampling hit max_seconds for motif %s: generated=%d requested=%d",
            motif.motif_id,
            len(candidates),
            requested_candidates,
        )
    length_obs = "-"
    if lengths:
        length_obs = f"{min(lengths)}..{max(lengths)}" if min(lengths) != max(lengths) else str(lengths[0])
    return _select(
        candidates,
        length_obs=length_obs,
        cap_applied=cap_applied,
        requested=requested_candidates,
        generated=len(candidates),
        time_limited=time_limited,
    )
