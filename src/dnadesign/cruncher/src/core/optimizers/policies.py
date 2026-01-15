"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/policies.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

MOVE_KINDS: Tuple[str, ...] = ("S", "B", "M", "L", "W", "I")


def move_probs_array(move_probs: Dict[str, float]) -> np.ndarray:
    return np.array([float(move_probs[k]) for k in MOVE_KINDS], dtype=float)


@dataclass
class MoveSchedule:
    start: np.ndarray
    end: Optional[np.ndarray] = None

    def probs(self, frac: float) -> np.ndarray:
        if self.end is None:
            return self.start
        frac = min(max(float(frac), 0.0), 1.0)
        probs = self.start + frac * (self.end - self.start)
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0:
            return self.start
        return probs / total


@dataclass
class TargetingPolicy:
    enabled: bool
    worst_tf_prob: float
    window_pad: int

    def maybe_target(
        self,
        *,
        seq_len: int,
        state: object,
        evaluator: object,
        rng: np.random.Generator,
    ) -> Optional[Tuple[str, int, int]]:
        if not self.enabled or rng.random() >= self.worst_tf_prob:
            return None
        per_tf = evaluator(state)
        if not per_tf:
            return None
        worst_tf = sorted(per_tf.items(), key=lambda item: (item[1], item[0]))[0][0]
        best_hits = evaluator.best_hits(state)
        if worst_tf not in best_hits:
            return None
        best_score, offset, strand = best_hits[worst_tf]
        if not math.isfinite(best_score):
            return None
        width = evaluator.pwm_width(worst_tf)
        if width <= 0:
            return None
        if strand == "-":
            offset = max(0, seq_len - width - offset)
        start = max(0, offset - self.window_pad)
        end = min(seq_len, offset + width + self.window_pad)
        if end <= start:
            return None
        return worst_tf, start, end


def targeted_start(
    *,
    seq_len: int,
    block_len: int,
    target: Optional[Tuple[int, int]],
    rng: np.random.Generator,
) -> int:
    if target is None or block_len >= seq_len:
        return int(rng.integers(0, max(1, seq_len - block_len + 1)))
    target_start, target_end = target
    center = (target_start + target_end) // 2
    lo = max(0, target_end - block_len)
    hi = min(seq_len - block_len, target_start)
    if lo <= hi:
        return int(rng.integers(lo, hi + 1))
    start = center - (block_len // 2)
    return max(0, min(seq_len - block_len, start))


@dataclass
class AdaptiveBetaController:
    target: float
    window: int
    k: float
    min_beta: float
    max_beta: float
    moves: Sequence[str]

    attempts: int = 0
    accepts: int = 0
    scale: float = 1.0

    def record(self, move_kind: str, accepted: bool) -> None:
        if move_kind not in self.moves:
            return
        self.attempts += 1
        if accepted:
            self.accepts += 1

    def update_scale(self) -> None:
        if self.attempts < self.window:
            return
        acc = self.accepts / float(self.attempts)
        self.scale *= math.exp(self.k * (acc - self.target))
        self.scale = max(self.scale, 1.0e-6)
        self.attempts = 0
        self.accepts = 0

    def beta(self, base_beta: float) -> float:
        beta = float(base_beta) * float(self.scale)
        if beta < self.min_beta:
            beta = self.min_beta
            self.scale = beta / float(base_beta)
        if beta > self.max_beta:
            beta = self.max_beta
            self.scale = beta / float(base_beta)
        return beta


@dataclass
class AdaptiveSwapController:
    target: float
    window: int
    k: float
    min_scale: float
    max_scale: float

    attempts: int = 0
    accepts: int = 0
    scale: float = 1.0

    def record(self, accepted: bool) -> None:
        self.attempts += 1
        if accepted:
            self.accepts += 1

    def update_scale(self) -> None:
        if self.attempts < self.window:
            return
        acc = self.accepts / float(self.attempts)
        self.scale *= math.exp(self.k * (acc - self.target))
        self.scale = max(self.min_scale, min(self.max_scale, self.scale))
        self.attempts = 0
        self.accepts = 0
