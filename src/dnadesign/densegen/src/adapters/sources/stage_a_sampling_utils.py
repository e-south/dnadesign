"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_sampling_utils.py

PWM sampling utilities for Stage-A candidate generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

SMOOTHING_ALPHA = 1e-6
_BASES = np.array(["A", "C", "G", "T"])


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
    probs = arr / row_sums
    return np.cumsum(probs, axis=1)


def _pwm_consensus(matrix: Sequence[dict[str, float]]) -> str:
    return "".join(max(row.items(), key=lambda kv: kv[1])[0] for row in matrix)


def _sample_from_background_cdf(rng: np.random.Generator, cdf: np.ndarray, length: int) -> str:
    if length <= 0:
        return ""
    draw = rng.random(int(length))
    idx = np.searchsorted(cdf, draw, side="left")
    return "".join(_BASES[idx])


def _sample_pwm_batch(
    rng: np.random.Generator,
    matrix_cdf: np.ndarray,
    *,
    count: int,
) -> List[str]:
    draw = rng.random((int(count), matrix_cdf.shape[0]))
    idx = np.searchsorted(matrix_cdf, draw, side="left")
    return ["".join(_BASES[row]) for row in idx]


def _sample_background_batch(
    rng: np.random.Generator,
    background_cdf: np.ndarray,
    *,
    count: int,
    length: int,
) -> List[str]:
    if int(count) <= 0 or int(length) <= 0:
        return []
    draw = rng.random((int(count), int(length)))
    idx = np.searchsorted(background_cdf, draw, side="left")
    return ["".join(_BASES[row]) for row in idx]


def _ranges_overlap(a_start: int, a_stop: int, b_start: int, b_stop: int) -> bool:
    return int(a_start) <= int(b_stop) and int(b_start) <= int(a_stop)


def sample_sequence_from_background(
    rng: np.random.Generator,
    probs: dict[str, float],
    length: int,
) -> str:
    return _sample_from_background_cdf(rng, _background_cdf(probs), length)


def sample_sequence_from_pwm(rng: np.random.Generator, matrix: List[dict[str, float]]) -> str:
    return _sample_pwm_batch(rng, _matrix_cdf(matrix), count=1)[0]


def _matrix_array(matrix: List[dict[str, float]]) -> np.ndarray:
    arr = np.zeros((len(matrix), 4), dtype=float)
    for idx, row in enumerate(matrix):
        arr[idx] = [row.get(base, 0.0) for base in ("A", "C", "G", "T")]
    return arr


def _information_bits(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("PWM rows must sum to > 0 for information content.")
    probs = matrix / row_sums
    entropy = -(probs * np.where(probs > 0, np.log2(probs), 0.0)).sum(axis=1)
    info_bits = 2.0 - entropy
    return float(info_bits.sum())


def _select_pwm_window(
    matrix: List[dict[str, float]],
    *,
    window_length: int,
    window_strategy: str,
) -> tuple[List[dict[str, float]], int, int]:
    if window_length <= 0:
        raise ValueError("trimming.window_length must be > 0.")
    if window_length >= len(matrix):
        return matrix, 0, len(matrix)
    if window_strategy != "max_info":
        raise ValueError(f"Unsupported trimming.window_strategy: {window_strategy}")
    best_start = 0
    best_score = float("-inf")
    matrix_arr = _matrix_array(matrix)
    for start in range(0, len(matrix) - window_length + 1):
        window = matrix_arr[start : start + window_length]
        info_bits = _information_bits(window)
        if info_bits > best_score:
            best_score = info_bits
            best_start = start
    end = best_start + window_length
    return matrix[best_start:end], best_start, end
