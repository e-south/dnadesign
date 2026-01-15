"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/optimizers/gibbs.py

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
from dnadesign.cruncher.core.optimizers.cooling import make_beta_scheduler
from dnadesign.cruncher.core.optimizers.helpers import _replace_block, slide_window, swap_block
from dnadesign.cruncher.core.optimizers.policies import (
    MOVE_KINDS,
    AdaptiveBetaController,
    MoveSchedule,
    TargetingPolicy,
    move_probs_array,
    targeted_start,
)
from dnadesign.cruncher.core.state import SequenceState, make_seed
from dnadesign.cruncher.utils.run_status import RunStatusWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GibbsOptimizer(Optimizer):
    """
    Cooling Gibbs/Metropolis sampler producing diverse motif‐rich sequences.
    Now creates an independent seed for each chain according to init_cfg,
    and records *all* states with a non‐negative “draw” index.
    """

    def __init__(
        self,
        evaluator: Any,
        cfg: Dict[str, Any],
        rng: np.random.Generator,
        *,
        init_cfg: Any,
        pwms: Dict[str, Any],
        status_writer: RunStatusWriter | None = None,
    ):
        """
        evaluator: SequenceEvaluator
        cfg: flattened config dict (keys include 'draws','tune','chains', etc.)
        rng: numpy.random.Generator

        init_cfg: Original InitConfig (with kind, length, pad_with, regulator)
        pwms: full dict of {tf_name: PWM} for seeding
        """
        super().__init__(evaluator, cfg, rng)
        logger.debug("Initializing GibbsOptimizer with config: %s", cfg)

        # Basic MCMC dimensions
        self.draws = int(cfg["draws"])
        self.tune = int(cfg["tune"])
        self.chains = int(cfg["chains"])
        self.min_dist = int(cfg["min_dist"])
        self.top_k = int(cfg["top_k"])
        self.record_tune = bool(cfg.get("record_tune", False))
        self.progress_bar = bool(cfg.get("progress_bar", True))
        self.progress_every = int(cfg.get("progress_every", 0))
        self.status_writer = status_writer

        # Unpack move‐ranges
        self.move_cfg = {
            "block_len_range": tuple(cfg["block_len_range"]),
            "multi_k_range": tuple(cfg["multi_k_range"]),
            "slide_max_shift": int(cfg["slide_max_shift"]),
            "swap_len_range": tuple(cfg["swap_len_range"]),
        }
        logger.debug("  Move config: %s", self.move_cfg)

        # Move probabilities (S,B,M,L,W,I)
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
        logger.debug(
            "  Move probabilities: %s",
            dict(zip(MOVE_KINDS, self.move_probs_start.tolist())),
        )

        # Build cooling schedule (per-chain)
        self.apply_during = str(cfg.get("apply_during", "tune"))
        self.schedule_scope = str(cfg.get("schedule_scope", "per_chain"))
        if self.schedule_scope not in {"per_chain", "global"}:
            raise ValueError(f"Unknown schedule_scope '{self.schedule_scope}' (expected per_chain|global).")
        if self.schedule_scope == "global" and self.apply_during == "tune":
            raise ValueError("schedule_scope='global' requires apply_during='all'.")
        sweeps_per_chain = self.tune + self.draws
        if self.schedule_scope == "global":
            schedule_sweeps = sweeps_per_chain * self.chains
        else:
            schedule_sweeps = self.tune if self.apply_during == "tune" else sweeps_per_chain
        schedule_sweeps = max(1, int(schedule_sweeps))
        cooling_cfg = {k: v for k, v in cfg.items() if k in ("kind", "beta", "stages")}
        self.beta_of = make_beta_scheduler(cooling_cfg, schedule_sweeps)
        logger.debug(
            "  Cooling config (beta_of scheduler) built with sweeps=%d (scope=%s)",
            schedule_sweeps,
            self.schedule_scope,
        )
        self.sweeps_per_chain = sweeps_per_chain
        self.schedule_sweeps = schedule_sweeps
        self.total_sweeps = schedule_sweeps if self.schedule_scope == "global" else sweeps_per_chain

        # Soft-min schedule (independent of MCMC temperature)
        softmin_cfg = cfg.get("softmin") or {}
        self.softmin_enabled = bool(softmin_cfg.get("enabled", False))
        if self.softmin_enabled:
            softmin_sched = {k: v for k, v in softmin_cfg.items() if k in ("kind", "beta", "stages")}
            self.softmin_of = make_beta_scheduler(softmin_sched, self.total_sweeps)
        else:
            self.softmin_of = None

        # Adaptive beta controller config (per-chain)
        self.adaptive_beta_cfg = cfg.get("adaptive_beta") or {}
        self.adaptive_beta_enabled = bool(self.adaptive_beta_cfg.get("enabled", False))

        # Keep references for seeding each chain
        self.init_cfg = init_cfg
        self.pwms = pwms

        # Tally of which move‐types happened
        self.move_tally: Counter = Counter()
        self.accept_tally: Counter = Counter()

        # STORAGE: sequences, metadata (chain, draw), and scores
        self.all_samples: List[np.ndarray] = []
        self.all_meta: List[Tuple[int, int]] = []
        self.all_scores: List[Dict[str, float]] = []

        # After ranking, store (chain, draw) for elites
        self.elites_meta: List[Tuple[int, int]] = []

        # ArviZ object for combined scalar
        self.trace_idata = None
        self.best_score: float | None = None
        self.best_meta: Tuple[int, int] | None = None

    def optimise(self) -> List[SequenceState]:
        """
        Run MCMC in two phases for each chain, using an independently drawn seed for each chain:
          1) Burn‐in: tune sweeps (record each state & per‐TF scores, draw indices = 0..tune−1)
          2) Sampling: draws sweeps (record each state & per‐TF scores, draw indices = tune..tune+draws−1)

        Returns a list of elite SequenceState objects (max K, Hamming‐distinct).
        """
        rng = self.rng
        evaluator: Any = self.scorer  # Actually SequenceEvaluator
        tune = self.tune
        draws = self.draws
        chains = self.chains
        min_dist = self.min_dist
        top_k = self.top_k
        move_cfg = self.move_cfg

        logger.debug("Beginning optimise: chains=%d, tune=%d, draws=%d", chains, tune, draws)

        # Will store combined‐fitness for each chain's sampled draws
        chain_scores: List[List[float]] = []

        # Clear old storage
        self.all_samples.clear()
        self.all_meta.clear()
        self.all_scores.clear()

        global_iter = 0
        for c in range(chains):
            # ─── 1) Create an independent seed for this chain ───
            seed_state = make_seed(self.init_cfg, self.pwms, rng)
            seq = seed_state.seq.copy()  # numpy array (L,)
            chain_trace: List[float] = []
            chain_iter = 0

            beta_controller = None
            if self.adaptive_beta_enabled:
                beta_controller = AdaptiveBetaController(
                    target=float(self.adaptive_beta_cfg.get("target_acceptance", 0.4)),
                    window=int(self.adaptive_beta_cfg.get("window", 100)),
                    k=float(self.adaptive_beta_cfg.get("k", 0.5)),
                    min_beta=float(self.adaptive_beta_cfg.get("min_beta", 1.0e-3)),
                    max_beta=float(self.adaptive_beta_cfg.get("max_beta", 10.0)),
                    moves=tuple(self.adaptive_beta_cfg.get("moves", ("B", "M"))),
                )
            stop_after_tune = bool(self.adaptive_beta_cfg.get("stop_after_tune", True))

            logger.debug("Chain %d: starting burn‐in", c + 1)
            beta_draw = self.beta_of(max(self.tune - 1, 0)) if self.apply_during == "tune" else None

            # ─── Phase 1: Burn‐in (record each state; draw_i = 0..tune−1) ───
            for b in tqdm(
                range(tune),
                desc=f"chain{c + 1} burn‐in",
                leave=False,
                disable=not self.progress_bar,
            ):
                schedule_iter = chain_iter if self.schedule_scope == "per_chain" else global_iter
                beta_mcmc = self.beta_of(schedule_iter)
                beta_softmin = self.softmin_of(schedule_iter) if self.softmin_of else None
                if beta_controller is not None:
                    beta_mcmc = beta_controller.beta(beta_mcmc)
                move_probs = self.move_schedule.probs(chain_iter / max(self.sweeps_per_chain - 1, 1))
                move_kind, accepted = self._perform_single_move(
                    seq, beta_mcmc, beta_softmin, evaluator, move_cfg, rng, move_probs
                )
                if beta_controller is not None:
                    beta_controller.record(move_kind, accepted)
                    beta_controller.update_scale()

                if self.record_tune:
                    # Record current state and per‐TF scores with draw index = b
                    current_state = SequenceState(seq.copy())
                    per_tf_map = evaluator(current_state)
                    draw_i = b  # 0 … tune−1
                    self.all_samples.append(seq.copy())
                    self.all_meta.append((c, draw_i))
                    self.all_scores.append(per_tf_map)
                    logger.debug(
                        "Chain %d burn‐in %d: recorded state draw_index=%d",
                        c + 1,
                        b,
                        draw_i,
                    )

                chain_iter += 1
                global_iter += 1
                self._maybe_log_progress("burn-in", c, b + 1, tune, beta=beta_mcmc, beta_softmin=beta_softmin)

            logger.debug("Chain %d: burn‐in complete. Starting sampling", c + 1)

            # ─── Phase 2: Sampling (record each state; draw_i = tune..tune+draws−1) ───
            for d in tqdm(
                range(draws),
                desc=f"chain{c + 1} sampling",
                leave=False,
                disable=not self.progress_bar,
            ):
                schedule_iter = chain_iter if self.schedule_scope == "per_chain" else global_iter
                if self.apply_during == "tune" and beta_draw is not None:
                    beta_mcmc = beta_draw
                else:
                    beta_mcmc = self.beta_of(schedule_iter)
                beta_softmin = self.softmin_of(schedule_iter) if self.softmin_of else None
                if beta_controller is not None:
                    beta_mcmc = beta_controller.beta(beta_mcmc)
                move_probs = self.move_schedule.probs(chain_iter / max(self.sweeps_per_chain - 1, 1))
                move_kind, accepted = self._perform_single_move(
                    seq, beta_mcmc, beta_softmin, evaluator, move_cfg, rng, move_probs
                )
                if beta_controller is not None and not stop_after_tune:
                    beta_controller.record(move_kind, accepted)
                    beta_controller.update_scale()

                current_state = SequenceState(seq.copy())
                per_tf_map = evaluator(current_state)
                combined_scalar = evaluator.combined(current_state, beta=beta_softmin)

                draw_i = tune + d  # tune … tune+draws−1
                self._track_best_score(combined_scalar, chain=c, draw=draw_i)
                self.all_samples.append(seq.copy())
                self.all_meta.append((c, draw_i))
                self.all_scores.append(per_tf_map)
                chain_trace.append(combined_scalar)

                logger.debug(
                    "Chain %d draw %d: combined_scalar=%.6f, per_tf=%s",
                    c + 1,
                    d,
                    combined_scalar,
                    per_tf_map,
                )
                chain_iter += 1
                global_iter += 1
                score_mean = None
                score_std = None
                if chain_trace:
                    window = min(len(chain_trace), 100)
                    window_scores = chain_trace[-window:]
                    score_mean = float(np.mean(window_scores))
                    score_std = float(np.std(window_scores))
                self._maybe_log_progress(
                    "sampling",
                    c,
                    d + 1,
                    draws,
                    beta=beta_mcmc,
                    beta_softmin=beta_softmin,
                    current_score=combined_scalar,
                    score_mean=score_mean,
                    score_std=score_std,
                )

            chain_scores.append(chain_trace)
            logger.debug("Chain %d: sampling complete with %d recorded draws", c + 1, draws)

        logger.debug("All chains complete. Move utilization: %s", dict(self.move_tally))

        # Build ArviZ InferenceData for combined fitness (chains × draws)
        scores_arr = np.asarray(chain_scores)
        if scores_arr.size == 0:
            raise RuntimeError(
                f"GibbsOptimizer: cannot build trace_inference_data with empty scores_arr (shape={scores_arr.shape})"
            )
        self.trace_idata = az.from_dict(posterior={"score": scores_arr})
        logger.debug("Built ArviZ InferenceData with shape %s", scores_arr.shape)

        # ─── Rank all recorded sequences by combined fitness at final β ───
        beta_final = self.beta_of(self.total_sweeps - 1)
        beta_softmin_final = self.softmin_of(self.total_sweeps - 1) if self.softmin_of else None
        if beta_softmin_final is None:
            logger.debug("Ranking sequences at final β=%.3f", beta_final)
        else:
            logger.debug("Ranking sequences at final β=%.3f (softmin=%.3f)", beta_final, beta_softmin_final)

        scored_list: List[Tuple[float, np.ndarray, int]] = []
        for idx, seq_arr in enumerate(self.all_samples):
            state = SequenceState(seq_arr.copy())
            combined_val = evaluator.combined(state, beta=beta_softmin_final)
            scored_list.append((combined_val, seq_arr.copy(), idx))

        # Sort descending by combined_val
        scored_list.sort(key=lambda x: x[0], reverse=True)
        logger.debug("Top combined scores: %s", [x[0] for x in scored_list[:10]])

        # Pick top_k, enforcing Hamming‐distance ≥ min_dist
        elites: List[SequenceState] = []
        used_indices: List[int] = []
        for combined_val, seq_arr, idx in scored_list:
            if len(elites) >= top_k:
                break
            if any(np.sum(seq_arr != e.seq) < min_dist for e in elites):
                continue
            elites.append(SequenceState(seq_arr.copy()))
            used_indices.append(idx)
            logger.debug(
                "Selected elite #%d (idx=%d, combined=%.6f)",
                len(elites),
                idx,
                combined_val,
            )

        self.elites_meta = [self.all_meta[i] for i in used_indices]
        logger.debug("Selected %d elites", len(elites))
        return elites

    def _perform_single_move(
        self,
        seq: np.ndarray,
        beta: float,
        beta_softmin: float | None,
        evaluator: Any,
        move_cfg: Dict[str, Any],
        rng: np.random.Generator,
        move_probs: np.ndarray,
    ) -> tuple[str, bool]:
        """
        Choose one move based on move_probs, then apply it.
        Uses evaluator.combined(state, beta=beta_softmin) with Metropolis acceptance scaled by beta.
        """
        L = seq.size
        move_kind = self._sample_move_kind(rng, move_probs)
        self.move_tally[move_kind] += 1
        accepted = False
        target = self.targeting.maybe_target(seq_len=L, state=SequenceState(seq.copy()), evaluator=evaluator, rng=rng)
        target_tf = target[0] if target else None
        target_window = (target[1], target[2]) if target else None

        if move_kind == "S":
            # Single‐nucleotide flip
            if target_window is not None:
                i = rng.integers(target_window[0], target_window[1])
            else:
                i = rng.integers(L)
            old_base = seq[i]
            lods = np.empty(4, float)
            logger.debug("Performing 'S' move at position %d", i)

            for b in range(4):
                seq[i] = b
                lods[b] = beta * evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            seq[i] = old_base

            lods -= lods.max()
            probs = np.exp(lods)
            new_base = rng.choice(4, p=probs / probs.sum())
            logger.debug(
                "    old_base=%d, lods=%s, probs=%s, chosen new_base=%d",
                old_base,
                lods,
                probs,
                new_base,
            )
            seq[i] = new_base
            self.accept_tally[move_kind] += 1
            accepted = True

        elif move_kind == "B":
            # Block replacement
            min_len, max_len = move_cfg["block_len_range"]
            length = rng.integers(min_len, max_len + 1)
            if length > L:
                return move_kind, False
            start = targeted_start(seq_len=L, block_len=length, target=target_window, rng=rng)
            proposal = rng.integers(0, 4, size=length)

            old_block = seq[start : start + length].copy()
            logger.debug(
                "Performing 'B' move: replace block [%d:%d] (length=%d)",
                start,
                start + length,
                length,
            )
            _replace_block(seq, start, length, proposal)
            new_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            _replace_block(seq, start, length, old_block)
            old_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)

            delta = beta * (new_comb - old_comb)
            logger.debug(
                "    old_comb=%.6f, new_comb=%.6f, delta=%.6f",
                old_comb,
                new_comb,
                delta,
            )
            if delta >= 0 or np.log(rng.random()) < delta:
                logger.debug("    Accepting 'B' move")
                _replace_block(seq, start, length, proposal)
                self.accept_tally[move_kind] += 1
                accepted = True
            else:
                logger.debug("    Rejecting 'B' move, reverting")

        elif move_kind == "M":
            # Multi‐site flips
            kmin, kmax = move_cfg["multi_k_range"]
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
            new_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            seq[idxs] = old_bases
            old_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)

            delta = beta * (new_comb - old_comb)
            logger.debug(
                "Performing 'M' move at idxs=%s: old_comb=%.6f, new_comb=%.6f, delta=%.6f",
                idxs.tolist(),
                old_comb,
                new_comb,
                delta,
            )
            if delta >= 0 or np.log(rng.random()) < delta:
                logger.debug("    Accepting 'M' move")
                seq[idxs] = proposal
                self.accept_tally[move_kind] += 1
                accepted = True
            else:
                logger.debug("    Rejecting 'M' move, reverting to old_bases")

        elif move_kind == "L":
            # Slide window
            min_len, max_len = move_cfg["swap_len_range"]
            length = rng.integers(min_len, max_len + 1)
            max_shift = int(move_cfg["slide_max_shift"])
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
            new_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            slide_window(seq, start + int(shift), length, int(-shift))
            old_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            delta = beta * (new_comb - old_comb)
            if delta >= 0 or np.log(rng.random()) < delta:
                slide_window(seq, start, length, int(shift))
                self.accept_tally[move_kind] += 1
                accepted = True

        elif move_kind == "W":
            # Swap two blocks
            min_len, max_len = move_cfg["swap_len_range"]
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
            new_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            swap_block(seq, start_a, start_b, length)
            old_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            delta = beta * (new_comb - old_comb)
            if delta >= 0 or np.log(rng.random()) < delta:
                swap_block(seq, start_a, start_b, length)
                self.accept_tally[move_kind] += 1
                accepted = True

        else:  # move_kind == "I"
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
            new_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            _replace_block(seq, start, width, old_block)
            old_comb = evaluator.combined(SequenceState(seq.copy()), beta=beta_softmin)
            delta = beta * (new_comb - old_comb)
            if delta >= 0 or np.log(rng.random()) < delta:
                _replace_block(seq, start, width, proposal)
                self.accept_tally[move_kind] += 1
                accepted = True

        return move_kind, accepted

    def _sample_move_kind(self, rng: np.random.Generator, move_probs: np.ndarray) -> str:
        """
        Sample a move‐kind among MOVE_KINDS according to move_probs.
        """
        idx = rng.choice(len(MOVE_KINDS), p=move_probs)
        choice = MOVE_KINDS[idx]
        logger.debug(
            "    Sampled move_kind=%s (probs=%s)",
            choice,
            dict(zip(MOVE_KINDS, move_probs.tolist())),
        )
        return choice

    def _track_best_score(self, score: float, *, chain: int, draw: int) -> None:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_meta = (chain, draw)

    def _maybe_log_progress(
        self,
        phase: str,
        chain_idx: int,
        step: int,
        total: int,
        *,
        beta: float | None = None,
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
        score_blob = ""
        if current_score is not None:
            score_blob = f" score={current_score:.3f}"
        if score_mean is not None and score_std is not None:
            score_blob += f" mean={score_mean:.3f}±{score_std:.3f}"
        if self.best_score is not None:
            score_blob += f" best={self.best_score:.3f}"
        logger.info(
            "Progress: chain %d %s %d/%d (%.1f%%) accept={%s}%s",
            chain_idx + 1,
            phase,
            step,
            total,
            pct,
            acc_label,
            score_blob,
        )
        if self.status_writer is not None:
            self.status_writer.update(
                phase=phase,
                chain=chain_idx + 1,
                step=step,
                total=total,
                progress_pct=round(pct, 2),
                acceptance_rate=acceptance_rate,
                beta=beta,
                beta_softmin=beta_softmin,
                current_score=current_score,
                score_mean=score_mean,
                score_std=score_std,
                best_score=self.best_score,
                best_chain=(self.best_meta[0] + 1) if self.best_meta else None,
                best_draw=(self.best_meta[1]) if self.best_meta else None,
            )

    def stats(self) -> Dict[str, object]:
        totals = dict(self.move_tally)
        accepted = dict(self.accept_tally)
        acceptance_rate = {k: (accepted.get(k, 0) / totals[k]) if totals.get(k, 0) else 0.0 for k in totals}
        return {
            "moves": totals,
            "accepted": accepted,
            "acceptance_rate": acceptance_rate,
        }
