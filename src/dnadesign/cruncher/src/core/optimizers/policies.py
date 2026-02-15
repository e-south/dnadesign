"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/policies.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

MOVE_KINDS: Tuple[str, ...] = ("S", "B", "M", "L", "W", "I")


def move_probs_array(move_probs: Dict[str, float]) -> np.ndarray:
    raw = np.array([float(move_probs.get(k, 0.0)) for k in MOVE_KINDS], dtype=float)
    raw = np.clip(raw, 0.0, None)
    total = float(raw.sum())
    if total <= 0:
        return np.full(len(MOVE_KINDS), 1.0 / len(MOVE_KINDS), dtype=float)
    return raw / total


@dataclass
class MoveSchedule:
    start: np.ndarray
    end: Optional[np.ndarray] = None

    def probs(self, frac: float) -> np.ndarray:
        if self.end is None:
            return move_probs_array(dict(zip(MOVE_KINDS, self.start.tolist())))
        frac = min(max(float(frac), 0.0), 1.0)
        probs = self.start + frac * (self.end - self.start)
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0:
            return np.full_like(probs, 1.0 / len(probs))
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
        per_tf: Optional[Dict[str, float]] = None,
    ) -> Optional[Tuple[str, int, int]]:
        if not self.enabled or rng.random() >= self.worst_tf_prob:
            return None
        if per_tf is None:
            per_tf = evaluator(state)
        if not per_tf:
            return None
        worst_tf = sorted(per_tf.items(), key=lambda item: (item[1], item[0]))[0][0]
        best_score, offset, strand = evaluator.best_hit(state, worst_tf)
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


@dataclass
class AdaptiveMoveController:
    enabled: bool
    window: int
    k: float
    min_prob: float
    max_prob: float
    targets: Dict[str, float]
    kinds: Sequence[str]

    attempts: Dict[str, int] = field(default_factory=dict)
    accepts: Dict[str, int] = field(default_factory=dict)
    log_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for kind in self.kinds:
            self.attempts.setdefault(kind, 0)
            self.accepts.setdefault(kind, 0)
            self.log_weights.setdefault(kind, 0.0)

    def record(self, move_kind: str, *, accepted: bool) -> None:
        if not self.enabled or move_kind not in self.attempts:
            return
        self.attempts[move_kind] += 1
        if accepted:
            self.accepts[move_kind] += 1

    def adapt(self, base_probs: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return base_probs
        self._maybe_update()
        weights = np.ones(len(MOVE_KINDS), dtype=float)
        for kind, lw in self.log_weights.items():
            idx = MOVE_KINDS.index(kind)
            weights[idx] = math.exp(lw)
        probs = np.asarray(base_probs, dtype=float) * weights
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0:
            probs = np.full(len(MOVE_KINDS), 1.0 / len(MOVE_KINDS), dtype=float)
        else:
            probs /= total
        return self._constrain_prob_bounds(probs)

    def _maybe_update(self) -> None:
        for kind in list(self.attempts.keys()):
            attempted = int(self.attempts[kind])
            if attempted < self.window:
                continue
            accepted = int(self.accepts[kind])
            acc = accepted / float(max(1, attempted))
            target = float(self.targets.get(kind, 0.5))
            self.log_weights[kind] += self.k * (acc - target)
            self.log_weights[kind] = max(-6.0, min(6.0, self.log_weights[kind]))
            self.attempts[kind] = 0
            self.accepts[kind] = 0

    def _constrain_prob_bounds(self, probs: np.ndarray) -> np.ndarray:
        constrained = np.array(probs, dtype=float, copy=True)
        constrained_mask = np.zeros(len(MOVE_KINDS), dtype=bool)
        constrained_total = 0.0
        for kind in self.kinds:
            idx = MOVE_KINDS.index(kind)
            clipped = min(self.max_prob, max(self.min_prob, float(constrained[idx])))
            constrained[idx] = clipped
            constrained_mask[idx] = True
            constrained_total += clipped
        if constrained_total >= 1.0:
            return constrained / max(constrained_total, 1.0e-12)
        free_idx = np.where(~constrained_mask)[0]
        free_total = float(np.sum(probs[free_idx])) if free_idx.size else 0.0
        remaining = 1.0 - constrained_total
        if free_total <= 0.0:
            if free_idx.size:
                constrained[free_idx] = remaining / float(free_idx.size)
            else:
                constrained /= max(np.sum(constrained), 1.0e-12)
            return constrained
        constrained[free_idx] = probs[free_idx] * (remaining / free_total)
        norm = float(np.sum(constrained))
        if norm <= 0.0:
            return np.full(len(MOVE_KINDS), 1.0 / len(MOVE_KINDS), dtype=float)
        return constrained / norm


@dataclass
class AdaptiveProposalController:
    enabled: bool
    window: int
    step: float
    min_scale: float
    max_scale: float
    target_low: float
    target_high: float

    block_scale: float = 1.0
    multi_scale: float = 1.0
    attempts_b: int = 0
    accepts_b: int = 0
    attempts_m: int = 0
    accepts_m: int = 0

    def record(self, move_kind: str, *, accepted: bool) -> None:
        if not self.enabled:
            return
        if move_kind == "B":
            self.attempts_b += 1
            if accepted:
                self.accepts_b += 1
        elif move_kind == "M":
            self.attempts_m += 1
            if accepted:
                self.accepts_m += 1

    def current_ranges(
        self,
        base_block: tuple[int, int],
        base_multi: tuple[int, int],
        *,
        sequence_length: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.enabled:
            self._maybe_update()
        block = self._scaled_range(base_block, self.block_scale, sequence_length=sequence_length)
        multi = self._scaled_range(base_multi, self.multi_scale, sequence_length=sequence_length)
        return block, multi

    def _maybe_update(self) -> None:
        if self.attempts_b >= self.window:
            self.block_scale = self._updated_scale(self.block_scale, self.accepts_b / float(max(1, self.attempts_b)))
            self.attempts_b = 0
            self.accepts_b = 0
        if self.attempts_m >= self.window:
            self.multi_scale = self._updated_scale(self.multi_scale, self.accepts_m / float(max(1, self.attempts_m)))
            self.attempts_m = 0
            self.accepts_m = 0

    def _updated_scale(self, scale: float, acc: float) -> float:
        if acc > self.target_high:
            scale += self.step
        elif acc < self.target_low:
            scale -= self.step
        return max(self.min_scale, min(self.max_scale, scale))

    @staticmethod
    def _scaled_range(base_range: tuple[int, int], scale: float, *, sequence_length: int) -> tuple[int, int]:
        lo_base, hi_base = int(base_range[0]), int(base_range[1])
        lo = max(1, int(round(lo_base * scale)))
        hi = max(lo, int(round(hi_base * scale)))
        hi = min(hi, int(sequence_length))
        lo = min(lo, hi)
        return lo, hi


@dataclass
class AdaptiveSwapPairController:
    n_pairs: int
    enabled: bool
    target: float
    window: int
    k: float
    min_scale: float
    max_scale: float
    strict: bool
    saturation_windows: int

    attempts_total: int = 0
    accepts_total: int = 0
    attempts_window: list[int] = field(default_factory=list)
    accepts_window: list[int] = field(default_factory=list)
    pair_log_offsets: list[float] = field(default_factory=list)
    saturated_windows_seen: int = 0

    def __post_init__(self) -> None:
        if not self.attempts_window:
            self.attempts_window = [0 for _ in range(self.n_pairs)]
        if not self.accepts_window:
            self.accepts_window = [0 for _ in range(self.n_pairs)]
        if not self.pair_log_offsets:
            self.pair_log_offsets = [0.0 for _ in range(self.n_pairs)]

    @property
    def attempts(self) -> int:
        return int(self.attempts_total)

    @property
    def accepts(self) -> int:
        return int(self.accepts_total)

    @property
    def scale(self) -> float:
        if self.n_pairs <= 0:
            return 1.0
        mean_log = float(sum(self.pair_log_offsets)) / float(self.n_pairs)
        return float(math.exp(mean_log))

    @property
    def pair_scales(self) -> list[float]:
        return [float(math.exp(v)) for v in self.pair_log_offsets]

    def record(self, *, pair_idx: int, accepted: bool) -> None:
        if not self.enabled:
            return
        if pair_idx < 0 or pair_idx >= self.n_pairs:
            raise IndexError(f"pair index {pair_idx} out of bounds for n_pairs={self.n_pairs}")
        self.attempts_total += 1
        self.attempts_window[pair_idx] += 1
        if accepted:
            self.accepts_total += 1
            self.accepts_window[pair_idx] += 1

    def update(self) -> None:
        if not self.enabled:
            return
        changed = False
        for pair_idx in range(self.n_pairs):
            attempts = int(self.attempts_window[pair_idx])
            if attempts < self.window:
                continue
            accepts = int(self.accepts_window[pair_idx])
            acc = accepts / float(max(1, attempts))
            self.pair_log_offsets[pair_idx] += self.k * (acc - self.target)
            self.attempts_window[pair_idx] = 0
            self.accepts_window[pair_idx] = 0
            changed = True
        if not changed:
            return
        self._clamp_global_scale()
        if self.at_max_scale():
            self.saturated_windows_seen += 1
        else:
            self.saturated_windows_seen = 0

    def ladder_from_base(self, base_betas: Sequence[float]) -> list[float]:
        if not base_betas:
            return []
        if len(base_betas) <= 1 or not self.enabled:
            return [float(b) for b in base_betas]
        if len(base_betas) != self.n_pairs + 1:
            raise ValueError(
                f"base beta ladder length must equal n_pairs + 1 (got len={len(base_betas)} n_pairs={self.n_pairs})"
            )
        base = [float(b) for b in base_betas]
        betas = [base[0]]
        for pair_idx in range(self.n_pairs):
            gap_base = base[pair_idx + 1] / base[pair_idx]
            gap = gap_base * math.exp(self.pair_log_offsets[pair_idx])
            gap = max(gap, 1.0e-12)
            betas.append(betas[-1] * gap)
        return betas

    def at_max_scale(self, *, atol: float = 1.0e-9) -> bool:
        return self.scale >= (self.max_scale - atol)

    def tuning_limited(self) -> bool:
        return self.strict and self.saturated_windows_seen >= self.saturation_windows

    def _clamp_global_scale(self) -> None:
        if self.n_pairs <= 0:
            return
        mean_log = float(sum(self.pair_log_offsets)) / float(self.n_pairs)
        min_log = math.log(max(self.min_scale, 1.0e-12))
        max_log = math.log(max(self.max_scale, 1.0e-12))
        if mean_log < min_log:
            shift = min_log - mean_log
            self.pair_log_offsets = [v + shift for v in self.pair_log_offsets]
        elif mean_log > max_log:
            shift = max_log - mean_log
            self.pair_log_offsets = [v + shift for v in self.pair_log_offsets]
