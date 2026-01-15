"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/pt.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import arviz as az
import numpy as np
from dnadesign.cruncher.core.optimizers.base import Optimizer
from dnadesign.cruncher.core.optimizers.cooling import make_beta_ladder, make_beta_scheduler
from dnadesign.cruncher.core.optimizers.helpers import _replace_block, slide_window, swap_block
from dnadesign.cruncher.core.optimizers.policies import (
    MOVE_KINDS,
    AdaptiveSwapController,
    MoveSchedule,
    TargetingPolicy,
    move_probs_array,
    targeted_start,
)
from dnadesign.cruncher.core.state import SequenceState, make_seed
from dnadesign.cruncher.utils.run_status import RunStatusWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PTGibbsOptimizer(Optimizer):
    """Parallel-Tempered Gibbs/Metropolis sampler."""

    # Construction
    def __init__(
        self,
        evaluator: Any,  # SequenceEvaluator
        cfg: Dict[str, Any],
        rng: np.random.Generator,
        *,
        pwms: Dict[str, Any],
        init_cfg: Any,
        status_writer: RunStatusWriter | None = None,
    ) -> None:
        super().__init__(evaluator, cfg, rng)

        # Core dimensions
        self.draws: int = int(cfg["draws"])
        self.tune: int = int(cfg["tune"])
        self.chains: int = int(cfg["chains"])
        self.min_dist: int = int(cfg["min_dist"])
        self.top_k: int = int(cfg["top_k"])
        self.swap_prob: float = float(cfg["swap_prob"])
        self.record_tune: bool = bool(cfg.get("record_tune", False))
        self.progress_bar: bool = bool(cfg.get("progress_bar", True))
        self.progress_every: int = int(cfg.get("progress_every", 0))
        self.status_writer = status_writer

        # β‑ladder (base)
        cooling_cfg = {"kind": cfg["kind"], "beta": cfg["beta"]}
        self.beta_ladder_base: List[float] = make_beta_ladder(cooling_cfg)
        if len(self.beta_ladder_base) != self.chains:
            raise ValueError(
                "Length of beta ladder (%d) must match number of chains (%d)"
                % (len(self.beta_ladder_base), self.chains)
            )
        self.beta_ladder: List[float] = list(self.beta_ladder_base)
        self.sweeps_per_chain = self.tune + self.draws
        self.total_sweeps = self.sweeps_per_chain

        # Move configuration
        self.move_cfg: Dict[str, Tuple[int, int] | int] = {
            "block_len_range": tuple(cfg["block_len_range"]),
            "multi_k_range": tuple(cfg["multi_k_range"]),
            "slide_max_shift": int(cfg["slide_max_shift"]),
            "swap_len_range": tuple(cfg["swap_len_range"]),
        }
        self.move_probs_start = move_probs_array(cfg["move_probs"])
        move_sched_cfg = cfg.get("move_schedule") or {}
        if move_sched_cfg.get("enabled"):
            end_probs = move_probs_array(move_sched_cfg["end"])
        else:
            end_probs = None
        self.move_schedule = MoveSchedule(start=self.move_probs_start, end=end_probs)
        target_prob = float(cfg.get("target_worst_tf_prob", 0.0))
        target_pad = int(cfg.get("target_window_pad", 0))
        self.targeting = TargetingPolicy(
            enabled=target_prob > 0.0,
            worst_tf_prob=target_prob,
            window_pad=target_pad,
        )
        self.insertion_consensus_prob = float(cfg.get("insertion_consensus_prob", 0.5))

        # Soft-min schedule (independent of PT temperature ladder)
        softmin_cfg = cfg.get("softmin") or {}
        self.softmin_enabled = bool(softmin_cfg.get("enabled", False))
        if self.softmin_enabled:
            softmin_sched = {k: v for k, v in softmin_cfg.items() if k in ("kind", "beta", "stages")}
            self.softmin_of = make_beta_scheduler(softmin_sched, self.total_sweeps)
        else:
            self.softmin_of = None

        # Adaptive swap controller
        self.adaptive_swap_cfg = cfg.get("adaptive_swap") or {}
        self.adaptive_swap_enabled = bool(self.adaptive_swap_cfg.get("enabled", False))
        if self.adaptive_swap_enabled:
            self.swap_controller = AdaptiveSwapController(
                target=float(self.adaptive_swap_cfg.get("target_swap", 0.25)),
                window=int(self.adaptive_swap_cfg.get("window", 50)),
                k=float(self.adaptive_swap_cfg.get("k", 0.5)),
                min_scale=float(self.adaptive_swap_cfg.get("min_scale", 0.25)),
                max_scale=float(self.adaptive_swap_cfg.get("max_scale", 4.0)),
            )
        else:
            self.swap_controller = None

        # References needed during optimisation
        self.pwms = pwms
        self.init_cfg = init_cfg

        # Book‑keeping
        self.move_tally: Counter = Counter()
        self.accept_tally: Counter = Counter()
        self.swap_attempts: int = 0
        self.swap_accepts: int = 0
        self.all_samples: List[np.ndarray] = []
        self.all_meta: List[Tuple[int, int]] = []
        self.all_scores: List[Dict[str, float]] = []
        self.elites_meta: List[Tuple[int, int]] = []
        self.trace_idata = None  # filled after optimise()
        self.best_score: float | None = None
        self.best_meta: Tuple[int, int] | None = None

    # Public API
    def optimise(self) -> List[SequenceState]:  # noqa: C901  (long but readable)
        """Run PT-MCMC and return *k* diverse elite sequences."""

        rng = self.rng
        evaluator = self.scorer  # SequenceEvaluator
        C, T, D = self.chains, self.tune, self.draws
        logger.debug("Starting PT optimisation: chains=%d  tune=%d  draws=%d", C, T, D)

        # Seed each chain independently
        chain_states: List[np.ndarray] = [make_seed(self.init_cfg, self.pwms, rng).seq.copy() for _ in range(C)]

        # For trace: only *draw* phase (not tune) is stored per ArviZ convention.
        chain_scores: List[List[float]] = [[] for _ in range(C)]

        # Helper to record a state
        def _record(chain_id: int, draw_idx: int, seq_arr: np.ndarray):
            state = SequenceState(seq_arr.copy())
            per_tf = evaluator(state)
            self.all_samples.append(seq_arr.copy())
            self.all_scores.append(per_tf)
            self.all_meta.append((chain_id, draw_idx))

        stop_after_tune = bool(self.adaptive_swap_cfg.get("stop_after_tune", True))

        # Burn‑in sweeps (record like Gibbs for consistency)
        for t in tqdm(range(T), desc="burn‑in", leave=False, disable=not self.progress_bar):
            sweep_idx = t
            beta_softmin = self.softmin_of(sweep_idx) if self.softmin_of else None
            move_probs = self.move_schedule.probs(sweep_idx / max(self.total_sweeps - 1, 1))
            scale = self.swap_controller.scale if self.swap_controller else 1.0
            self.beta_ladder = [b * scale for b in self.beta_ladder_base]
            for c in range(C):
                self._single_chain_move(chain_states[c], self.beta_ladder[c], beta_softmin, evaluator, rng, move_probs)
                if self.record_tune:
                    _record(c, t, chain_states[c])

            # Optional swap attempts during tune (for adaptive swap calibration)
            if self.adaptive_swap_enabled:
                for c in range(C - 1):
                    if rng.random() >= self.swap_prob:
                        continue
                    self.swap_attempts += 1
                    s0, s1 = chain_states[c], chain_states[c + 1]
                    β0, β1 = self.beta_ladder[c], self.beta_ladder[c + 1]
                    f0 = evaluator.combined(SequenceState(s0.copy()), beta=beta_softmin)
                    f1 = evaluator.combined(SequenceState(s1.copy()), beta=beta_softmin)
                    Δ = (β1 - β0) * (f0 - f1)
                    accepted = False
                    if Δ >= 0 or np.log(rng.random()) < Δ:
                        chain_states[c], chain_states[c + 1] = s1, s0
                        self.swap_accepts += 1
                        accepted = True
                    if self.swap_controller is not None:
                        self.swap_controller.record(accepted)
                        if not stop_after_tune:
                            self.swap_controller.update_scale()
                if self.swap_controller is not None and stop_after_tune:
                    self.swap_controller.update_scale()

            self._maybe_log_progress("burn-in", t + 1, T)

        # Sampling sweeps + swap attempts
        for d in tqdm(range(D), desc="sampling", leave=False, disable=not self.progress_bar):
            sweep_idx = T + d
            beta_softmin = self.softmin_of(sweep_idx) if self.softmin_of else None
            move_probs = self.move_schedule.probs(sweep_idx / max(self.total_sweeps - 1, 1))
            scale = self.swap_controller.scale if self.swap_controller else 1.0
            self.beta_ladder = [b * scale for b in self.beta_ladder_base]

            # Within‑chain proposals
            for c in range(C):
                self._single_chain_move(
                    chain_states[c],
                    self.beta_ladder[c],
                    beta_softmin,
                    evaluator,
                    rng,
                    move_probs,
                )
                _record(c, T + d, chain_states[c])

                # store combined fitness only for draw phase
                comb = evaluator.combined(SequenceState(chain_states[c].copy()), beta=beta_softmin)
                chain_scores[c].append(comb)
                if self.best_score is None or comb > self.best_score:
                    self.best_score = comb
                    self.best_meta = (c, T + d)

            # Pair‑wise swaps
            for c in range(C - 1):
                if rng.random() >= self.swap_prob:
                    continue
                self.swap_attempts += 1
                s0, s1 = chain_states[c], chain_states[c + 1]
                β0, β1 = self.beta_ladder[c], self.beta_ladder[c + 1]
                f0 = evaluator.combined(SequenceState(s0.copy()), beta=beta_softmin)
                f1 = evaluator.combined(SequenceState(s1.copy()), beta=beta_softmin)
                Δ = (β1 - β0) * (f0 - f1)
                accepted = False
                if Δ >= 0 or np.log(rng.random()) < Δ:
                    chain_states[c], chain_states[c + 1] = s1, s0  # swap in‑place
                    self.swap_accepts += 1
                    accepted = True
                if self.swap_controller is not None:
                    self.swap_controller.record(accepted)
                    if not stop_after_tune:
                        self.swap_controller.update_scale()
            if self.swap_controller is not None and stop_after_tune:
                self.swap_controller.update_scale()

            current_scores = [scores[-1] for scores in chain_scores if scores]
            score_mean = float(np.mean(current_scores)) if current_scores else None
            score_std = float(np.std(current_scores)) if current_scores else None
            current_best = float(max(current_scores)) if current_scores else None
            self._maybe_log_progress(
                "sampling",
                d + 1,
                D,
                current_score=current_best,
                score_mean=score_mean,
                score_std=score_std,
                beta_softmin=beta_softmin,
            )

        logger.debug("PT optimisation finished. Move utilisation: %s", dict(self.move_tally))

        # Build ArviZ trace from draw phase only
        score_arr = np.asarray(chain_scores)  # (C, D)
        self.trace_idata = az.from_dict(posterior={"score": score_arr})

        # Rank all recorded sequences by combined fitness
        beta_softmin_final = self.softmin_of(self.total_sweeps - 1) if self.softmin_of else None
        ranked: List[Tuple[float, np.ndarray, int]] = []
        for idx, seq in enumerate(self.all_samples):
            val = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin_final)
            ranked.append((val, seq.copy(), idx))
        ranked.sort(key=lambda x: x[0], reverse=True)

        elites: List[SequenceState] = []
        picked_idx: List[int] = []
        for val, seq, idx in ranked:
            if len(elites) >= self.top_k:
                break
            if any(np.sum(seq != e.seq) < self.min_dist for e in elites):
                continue
            elites.append(SequenceState(seq))
            picked_idx.append(idx)
        self.elites_meta = [self.all_meta[i] for i in picked_idx]
        return elites

    # Low‑level helpers
    def _single_chain_move(
        self,
        seq: np.ndarray,
        β: float,
        beta_softmin: float | None,
        evaluator: Any,
        rng: np.random.Generator,
        move_probs: np.ndarray,
    ) -> tuple[str, bool]:
        """One Gibbs-style proposal/accept cycle at inverse-temperature β."""

        L = seq.size
        move_kind = self._sample_move_kind(rng, move_probs)
        self.move_tally[move_kind] += 1
        accepted = False
        target = self.targeting.maybe_target(seq_len=L, state=SequenceState(seq.copy()), evaluator=evaluator, rng=rng)
        target_tf = target[0] if target else None
        target_window = (target[1], target[2]) if target else None

        # Single‑base flip
        if move_kind == "S":
            if target_window is not None:
                i = rng.integers(target_window[0], target_window[1])
            else:
                i = rng.integers(L)
            old = seq[i]
            lods = np.empty(4, float)
            for b in range(4):
                seq[i] = b
                lods[b] = β * evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            seq[i] = old
            lods -= lods.max()
            probs = np.exp(lods)
            seq[i] = rng.choice(4, p=probs / probs.sum())
            self.accept_tally[move_kind] += 1
            accepted = True
            return move_kind, accepted

        # Contiguous block replace
        if move_kind == "B":
            mn, mx = self.move_cfg["block_len_range"]
            length = rng.integers(mn, mx + 1)
            if length > L:
                return move_kind, False
            start = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
            proposal = rng.integers(0, 4, size=length)
            old_block = seq[start : start + length].copy()

            _replace_block(seq, start, length, proposal)
            new_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            _replace_block(seq, start, length, old_block)
            old_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)

            Δ = β * (new_f - old_f)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                _replace_block(seq, start, length, proposal)
                self.accept_tally[move_kind] += 1
                accepted = True
            return move_kind, accepted

        # Multi‑site flip
        if move_kind == "M":
            kmin, kmax = self.move_cfg["multi_k_range"]
            k = rng.integers(kmin, kmax + 1)
            if k > L:
                k = L
            if target_window is not None and (target_window[1] - target_window[0]) >= k:
                idxs = rng.choice(np.arange(target_window[0], target_window[1]), size=k, replace=False)
            else:
                idxs = rng.choice(L, size=k, replace=False)
            old_bases = seq[idxs].copy()
            proposal = rng.integers(0, 4, size=k)

            seq[idxs] = proposal
            new_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            seq[idxs] = old_bases
            old_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)

            Δ = β * (new_f - old_f)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                seq[idxs] = proposal
                self.accept_tally[move_kind] += 1
                accepted = True
            return move_kind, accepted

        if move_kind == "L":
            min_len, max_len = self.move_cfg["swap_len_range"]
            length = rng.integers(min_len, max_len + 1)
            max_shift = int(self.move_cfg["slide_max_shift"])
            if max_shift < 1 or length >= L:
                return move_kind, False
            shift = rng.integers(-max_shift, max_shift + 1)
            if shift == 0:
                shift = 1 if rng.random() < 0.5 else -1
            if shift > 0:
                min_start, max_start = 0, L - length - shift
            else:
                min_start, max_start = -shift, L - length
            if max_start < min_start:
                return move_kind, False
            start = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
            start = max(min_start, min(max_start, start))
            slide_window(seq, start, length, int(shift))
            new_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            slide_window(seq, start + int(shift), length, int(-shift))
            old_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            Δ = β * (new_f - old_f)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                slide_window(seq, start, length, int(shift))
                self.accept_tally[move_kind] += 1
                accepted = True
            return move_kind, accepted

        if move_kind == "W":
            min_len, max_len = self.move_cfg["swap_len_range"]
            length = rng.integers(min_len, max_len + 1)
            if length >= L:
                return move_kind, False
            start_a = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
            max_tries = 10
            start_b = start_a
            for _ in range(max_tries):
                start_b = rng.integers(0, L - length + 1)
                if abs(start_b - start_a) >= length:
                    break
            if abs(start_b - start_a) < length:
                return move_kind, False
            swap_block(seq, start_a, start_b, length)
            new_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            swap_block(seq, start_a, start_b, length)
            old_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            Δ = β * (new_f - old_f)
            if Δ >= 0 or np.log(rng.random()) < Δ:
                swap_block(seq, start_a, start_b, length)
                self.accept_tally[move_kind] += 1
                accepted = True
            return move_kind, accepted

        # Motif insertion proposal
        tf_names = list(self.pwms.keys())
        if not tf_names:
            return move_kind, False
        tf_name = target_tf if target_tf in self.pwms else rng.choice(tf_names)
        pwm = self.pwms[tf_name]
        width = pwm.length
        if width > L:
            return move_kind, False
        start = targeted_start(seq_len=L, block_len=width, target=target_window, rng=rng)
        if rng.random() < self.insertion_consensus_prob:
            proposal = np.argmax(pwm.matrix, axis=1).astype(np.int8)
        else:
            proposal = np.array(
                [rng.choice(4, p=row / row.sum()) for row in pwm.matrix],
                dtype=np.int8,
            )
        old_block = seq[start : start + width].copy()
        _replace_block(seq, start, width, proposal)
        new_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
        _replace_block(seq, start, width, old_block)
        old_f = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
        Δ = β * (new_f - old_f)
        if Δ >= 0 or np.log(rng.random()) < Δ:
            _replace_block(seq, start, width, proposal)
            self.accept_tally[move_kind] += 1
            accepted = True
        return move_kind, accepted

    def _sample_move_kind(self, rng: np.random.Generator, move_probs: np.ndarray) -> str:
        return str(rng.choice(MOVE_KINDS, p=move_probs))

    def _maybe_log_progress(
        self,
        phase: str,
        step: int,
        total: int,
        *,
        beta_softmin: float | None = None,
        current_score: float | None = None,
        score_mean: float | None = None,
        score_std: float | None = None,
    ) -> None:
        if not self.progress_every:
            return
        if step % self.progress_every != 0 and step != total:
            return
        totals = dict(self.move_tally)
        accepted = dict(self.accept_tally)
        acceptance_rate = {k: (accepted.get(k, 0) / totals[k]) if totals.get(k, 0) else 0.0 for k in totals}
        acc_label = ", ".join(f"{k}={acceptance_rate[k]:.2f}" for k in sorted(acceptance_rate))
        pct = (step / total) * 100 if total else 100.0
        swap_rate = self.swap_accepts / self.swap_attempts if self.swap_attempts else 0.0
        score_blob = ""
        if current_score is not None:
            score_blob = f" score={current_score:.3f}"
        if score_mean is not None and score_std is not None:
            score_blob += f" mean={score_mean:.3f}±{score_std:.3f}"
        if self.best_score is not None:
            score_blob += f" best={self.best_score:.3f}"
        logger.info(
            "Progress: %s %d/%d (%.1f%%) accept={%s} swap_acc=%.2f%s",
            phase,
            step,
            total,
            pct,
            acc_label,
            swap_rate,
            score_blob,
        )
        if self.status_writer is not None:
            self.status_writer.update(
                phase=phase,
                step=step,
                total=total,
                progress_pct=round(pct, 2),
                acceptance_rate=acceptance_rate,
                swap_accepts=self.swap_accepts,
                swap_attempts=self.swap_attempts,
                swap_rate=swap_rate,
                current_score=current_score,
                score_mean=score_mean,
                score_std=score_std,
                beta_softmin=beta_softmin,
                beta_min=min(self.beta_ladder) if self.beta_ladder else None,
                beta_max=max(self.beta_ladder) if self.beta_ladder else None,
                best_score=self.best_score,
                best_chain=(self.best_meta[0] + 1) if self.best_meta else None,
                best_draw=(self.best_meta[1]) if self.best_meta else None,
            )

    def stats(self) -> Dict[str, object]:
        totals = dict(self.move_tally)
        accepted = dict(self.accept_tally)
        acceptance_rate = {k: (accepted.get(k, 0) / totals[k]) if totals.get(k, 0) else 0.0 for k in totals}
        swap_rate = self.swap_accepts / self.swap_attempts if self.swap_attempts else 0.0
        return {
            "moves": totals,
            "accepted": accepted,
            "acceptance_rate": acceptance_rate,
            "swap_attempts": self.swap_attempts,
            "swap_accepts": self.swap_accepts,
            "swap_acceptance_rate": swap_rate,
        }
