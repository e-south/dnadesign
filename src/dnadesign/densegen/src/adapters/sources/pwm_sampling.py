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

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PWMMotif:
    motif_id: str
    matrix: List[dict[str, float]]  # per-position A/C/G/T probabilities
    background: dict[str, float]


def normalize_background(bg: Optional[dict[str, float]]) -> dict[str, float]:
    if not bg:
        return {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    total = sum(bg.values())
    if total <= 0:
        raise ValueError("Background frequencies must sum to > 0.")
    return {k: float(v) / total for k, v in bg.items()}


def score_sequence(seq: str, matrix: List[dict[str, float]]) -> float:
    score = 0.0
    for base, probs in zip(seq, matrix):
        p = probs.get(base, 0.0)
        if p <= 0:
            return float("-inf")
        score += float(np.log(p))
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


def select_by_score(
    candidates: List[Tuple[str, float]],
    *,
    n_sites: int,
    threshold: Optional[float],
    percentile: Optional[float],
    keep_low: bool,
) -> List[str]:
    scores = np.array([s for _, s in candidates], dtype=float)
    if threshold is None:
        cutoff = np.percentile(scores, float(percentile))  # type: ignore[arg-type]
    else:
        cutoff = float(threshold)
    if keep_low:
        picked = [seq for seq, score in candidates if score <= cutoff]
    else:
        picked = [seq for seq, score in candidates if score >= cutoff]
    unique = list(dict.fromkeys(picked))
    if len(unique) < n_sites:
        raise ValueError(
            f"PWM sampling produced {len(unique)} unique sites after filtering; "
            f"need {n_sites}. Adjust thresholds or oversample_factor."
        )
    return unique[:n_sites]


def sample_pwm_sites(
    rng: np.random.Generator,
    motif: PWMMotif,
    *,
    strategy: str,
    n_sites: int,
    oversample_factor: int,
    score_threshold: Optional[float],
    score_percentile: Optional[float],
) -> List[str]:
    if n_sites <= 0:
        raise ValueError("n_sites must be > 0")
    if oversample_factor <= 0:
        raise ValueError("oversample_factor must be > 0")
    if (score_threshold is None) == (score_percentile is None):
        raise ValueError("PWM sampling requires exactly one of score_threshold or score_percentile")
    if strategy == "consensus" and n_sites != 1:
        raise ValueError("PWM sampling strategy 'consensus' requires n_sites=1")

    keep_low = strategy == "background"
    width = len(motif.matrix)
    if width <= 0:
        raise ValueError(f"PWM motif '{motif.motif_id}' has zero width.")

    if strategy == "consensus":
        seq = "".join(max(row.items(), key=lambda kv: kv[1])[0] for row in motif.matrix)
        score = score_sequence(seq, motif.matrix)
        return select_by_score(
            [(seq, score)],
            n_sites=n_sites,
            threshold=score_threshold,
            percentile=score_percentile,
            keep_low=keep_low,
        )

    n_candidates = max(1, n_sites * oversample_factor)
    candidates: List[Tuple[str, float]] = []
    for _ in range(n_candidates):
        if strategy == "background":
            seq = sample_sequence_from_background(rng, motif.background, width)
        else:
            seq = sample_sequence_from_pwm(rng, motif.matrix)
        candidates.append((seq, score_sequence(seq, motif.matrix)))
    return select_by_score(
        candidates,
        n_sites=n_sites,
        threshold=score_threshold,
        percentile=score_percentile,
        keep_low=keep_low,
    )
