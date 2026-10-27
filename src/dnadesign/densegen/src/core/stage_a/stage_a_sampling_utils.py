"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/stage_a/stage_a_sampling_utils.py

PWM sampling utilities for Stage-A candidate generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

SMOOTHING_ALPHA = 1e-6
_BASES = np.array(["A", "C", "G", "T"])


@dataclass(frozen=True)
class PWMWindowSelection:
    length: int
    start: int
    score: float
    matrix: List[dict[str, float]]
    log_odds: List[dict[str, float]]


def normalize_background(bg: Optional[dict[str, float]]) -> dict[str, float]:
    if not bg:
        return {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    total = sum(bg.values())
    if total <= 0:
        raise ValueError("Background frequencies must sum to > 0.")
    return {k: float(v) / total for k, v in bg.items()}


def parse_bgfile(bgfile: str | Path) -> dict[str, float]:
    path = Path(bgfile)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Stage-A PWM sampling bgfile not found: {path}")
    base_values: dict[str, float] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("background"):
            continue
        line = line.replace("=", " ").replace(":", " ")
        parts = [part for part in line.split() if part]
        if not parts:
            continue
        token = parts[0].upper()
        if token not in {"A", "C", "G", "T"}:
            continue
        if len(parts) != 2:
            raise ValueError(f"bgfile line must be '<base> <freq>', got: {raw_line!r}")
        if token in base_values:
            raise ValueError(f"bgfile contains duplicate base entries: {token}")
        try:
            value = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"bgfile value for base {token} must be numeric") from exc
        base_values[token] = value
    if set(base_values) != {"A", "C", "G", "T"}:
        missing = sorted({"A", "C", "G", "T"} - set(base_values))
        raise ValueError(f"bgfile missing base frequencies: {missing}")
    return normalize_background(base_values)


def build_log_odds(
    matrix: List[dict[str, float]],
    background: dict[str, float],
    *,
    smoothing_alpha: float = SMOOTHING_ALPHA,
    log_base: float = 2.0,
) -> List[dict[str, float]]:
    if log_base <= 0.0 or log_base == 1.0:
        raise ValueError("log_base must be > 0 and != 1.0.")
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
                ratio = float(p / b)
                if log_base == 2.0:
                    lod_row[base] = float(np.log2(ratio))
                else:
                    lod_row[base] = float(np.log(ratio) / np.log(log_base))
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
        log_odds = build_log_odds(matrix, background, log_base=2.0)
    for base, probs in zip(seq, log_odds):
        lod = probs.get(base, float("-inf"))
        if lod == float("-inf"):
            return float("-inf")
        score += float(lod)
    return score


def _pwm_theoretical_max_score(log_odds: List[dict[str, float]]) -> float:
    if not log_odds:
        return 0.0
    total = 0.0
    for row in log_odds:
        if not row:
            raise ValueError("PWM log-odds rows must be non-empty to compute theoretical max score.")
        best = max(float(row.get(base, float("-inf"))) for base in ("A", "C", "G", "T"))
        if not np.isfinite(best):
            return float("-inf")
        total += best
    return float(total)


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


def _pwm_consensus_iupac(matrix: Sequence[dict[str, float]]) -> str:
    iupac_map = {
        frozenset({"A"}): "A",
        frozenset({"C"}): "C",
        frozenset({"G"}): "G",
        frozenset({"T"}): "T",
        frozenset({"A", "G"}): "R",
        frozenset({"C", "T"}): "Y",
        frozenset({"G", "C"}): "S",
        frozenset({"A", "T"}): "W",
        frozenset({"G", "T"}): "K",
        frozenset({"A", "C"}): "M",
        frozenset({"C", "G", "T"}): "B",
        frozenset({"A", "G", "T"}): "D",
        frozenset({"A", "C", "T"}): "H",
        frozenset({"A", "C", "G"}): "V",
        frozenset({"A", "C", "G", "T"}): "N",
    }
    consensus: list[str] = []
    threshold = 0.25
    for row in matrix:
        probs = {base: float(row.get(base, 0.0)) for base in ("A", "C", "G", "T")}
        if any(val < 0 for val in probs.values()):
            raise ValueError("PWM probabilities must be >= 0 to compute IUPAC consensus.")
        total = float(sum(probs.values()))
        if total <= 0:
            raise ValueError("PWM probabilities must sum to > 0 to compute IUPAC consensus.")
        for base in probs:
            probs[base] = probs[base] / total
        tol = 1e-9
        selected = {base for base, val in probs.items() if val >= threshold - tol}
        if not selected:
            selected = {max(probs.items(), key=lambda kv: kv[1])[0]}
        code = iupac_map.get(frozenset(selected))
        if code is None:
            raise ValueError(f"Unsupported IUPAC consensus bases: {sorted(selected)}")
        consensus.append(code)
    return "".join(consensus)


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
    width = int(matrix_cdf.shape[0])
    draws = rng.random((int(count), width))
    idx = (draws[:, :, None] <= matrix_cdf[None, :, :]).argmax(axis=2)
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
    safe = np.clip(probs, 1e-9, 1.0)
    entropy = -(probs * np.log2(safe)).sum(axis=1)
    info_bits = 2.0 - entropy
    return float(info_bits.sum())


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


def select_pwm_window_by_length(
    *,
    matrix: List[dict[str, float]],
    log_odds: List[dict[str, float]],
    length: int,
    strategy: str = "max_info",
) -> PWMWindowSelection:
    window_matrix, window_log_odds, start, score = _select_pwm_window(
        matrix=matrix,
        log_odds=log_odds,
        length=int(length),
        strategy=str(strategy),
    )
    return PWMWindowSelection(
        length=len(window_matrix),
        start=int(start),
        score=float(score),
        matrix=window_matrix,
        log_odds=window_log_odds,
    )
