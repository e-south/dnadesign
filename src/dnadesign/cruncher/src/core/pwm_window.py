"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/pwm_window.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dnadesign.cruncher.core.pwm import PWM


@dataclass(frozen=True, slots=True)
class PWMWindowSelection:
    start: int
    length: int
    score: float


def _information_bits(matrix: np.ndarray) -> float:
    p = matrix + 1e-9
    return float((2 + (p * np.log2(p)).sum(axis=1)).sum())


def select_pwm_window(pwm: PWM, *, length: int, strategy: str = "max_info") -> PWM:
    if length < 1:
        raise ValueError("pwm window length must be >= 1")
    if length > pwm.length:
        raise ValueError(f"pwm window length {length} exceeds PWM length {pwm.length}")
    if strategy != "max_info":
        raise ValueError("pwm window strategy must be 'max_info'")
    if length == pwm.length:
        return pwm

    matrix = np.asarray(pwm.matrix, dtype=float)
    log_odds = pwm.log_odds_matrix
    best_start = 0
    best_score = float("-inf")
    last_start = pwm.length - length
    for start in range(last_start + 1):
        window = matrix[start : start + length]
        score = _information_bits(window)
        if score > best_score:
            best_score = score
            best_start = start

    base_start = pwm.window_start or 0
    source_length = pwm.source_length or pwm.length
    window_matrix = matrix[best_start : best_start + length]
    window_log_odds = None
    if log_odds is not None:
        window_log_odds = np.asarray(log_odds, dtype=float)[best_start : best_start + length]

    return PWM(
        name=pwm.name,
        matrix=window_matrix,
        alphabet=pwm.alphabet,
        nsites=pwm.nsites,
        evalue=pwm.evalue,
        log_odds_matrix=window_log_odds,
        source_length=source_length,
        window_start=base_start + best_start,
        window_strategy=strategy,
        window_score=best_score,
    )
