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
from dnadesign.cruncher.core.optimizers.helpers import _replace_block
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
        logger.info("Initializing GibbsOptimizer with config: %s", cfg)

        # Basic MCMC dimensions
        self.draws = int(cfg["draws"])
        self.tune = int(cfg["tune"])
        self.chains = int(cfg["chains"])
        self.min_dist = int(cfg["min_dist"])
        self.top_k = int(cfg["top_k"])
        self.swap_prob = float(cfg["swap_prob"])
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

        # Unpack move_probs into a NumPy array (S,B,M)
        self.move_probs = np.array(
            [cfg["move_probs"]["S"], cfg["move_probs"]["B"], cfg["move_probs"]["M"]],
            dtype=float,
        )
        logger.debug(
            "  Move probabilities: %s",
            dict(S=self.move_probs[0], B=self.move_probs[1], M=self.move_probs[2]),
        )

        # Build cooling schedule
        total = self.tune + self.draws
        cooling_cfg = {k: v for k, v in cfg.items() if k in ("kind", "beta", "stages")}
        self.beta_of = make_beta_scheduler(cooling_cfg, total)
        logger.debug("  Cooling config (beta_of scheduler) built with total=%d sweeps", total)

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

        logger.info("Beginning optimise: chains=%d, tune=%d, draws=%d", chains, tune, draws)

        # Will store combined‐fitness for each chain's sampled draws
        chain_scores: List[List[float]] = []

        # Clear old storage
        self.all_samples.clear()
        self.all_meta.clear()
        self.all_scores.clear()

        global_iter = 0  # global sweep index across all chains

        for c in range(chains):
            # ─── 1) Create an independent seed for this chain ───
            seed_state = make_seed(self.init_cfg, self.pwms, rng)
            seq = seed_state.seq.copy()  # numpy array (L,)
            chain_trace: List[float] = []

            logger.info("Chain %d: starting burn‐in", c + 1)

            # ─── Phase 1: Burn‐in (record each state; draw_i = 0..tune−1) ───
            for b in tqdm(
                range(tune),
                desc=f"chain{c + 1} burn‐in",
                leave=False,
                disable=not self.progress_bar,
            ):
                beta_mcmc = self.beta_of(global_iter)
                self._perform_single_move(seq, beta_mcmc, evaluator, move_cfg, rng)

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

                global_iter += 1
                self._maybe_log_progress("burn-in", c, b + 1, tune, beta=beta_mcmc)

            logger.info("Chain %d: burn‐in complete. Starting sampling", c + 1)

            # ─── Phase 2: Sampling (record each state; draw_i = tune..tune+draws−1) ───
            for d in tqdm(
                range(draws),
                desc=f"chain{c + 1} sampling",
                leave=False,
                disable=not self.progress_bar,
            ):
                beta_mcmc = self.beta_of(global_iter)
                self._perform_single_move(seq, beta_mcmc, evaluator, move_cfg, rng)

                current_state = SequenceState(seq.copy())
                per_tf_map = evaluator(current_state)
                combined_scalar = evaluator.combined(current_state)

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
                    current_score=combined_scalar,
                    score_mean=score_mean,
                    score_std=score_std,
                )

            chain_scores.append(chain_trace)
            logger.info("Chain %d: sampling complete with %d recorded draws", c + 1, draws)

        logger.info("All chains complete. Move utilization: %s", dict(self.move_tally))

        # Build ArviZ InferenceData for combined fitness (chains × draws)
        scores_arr = np.asarray(chain_scores)
        if scores_arr.size == 0:
            raise RuntimeError(
                f"GibbsOptimizer: cannot build trace_inference_data with empty scores_arr (shape={scores_arr.shape})"
            )
        self.trace_idata = az.from_dict(posterior={"score": scores_arr})
        logger.info("Built ArviZ InferenceData with shape %s", scores_arr.shape)

        # ─── Rank all recorded sequences by combined fitness at final β ───
        total_sweeps = self.tune + self.draws
        beta_final = self.beta_of(total_sweeps - 1)
        logger.info("Ranking sequences at final β=%.3f", beta_final)

        scored_list: List[Tuple[float, np.ndarray, int]] = []
        for idx, seq_arr in enumerate(self.all_samples):
            state = SequenceState(seq_arr.copy())
            combined_val = evaluator.combined(state)
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
        logger.info("Selected %d elites", len(elites))
        return elites

    def _perform_single_move(
        self,
        seq: np.ndarray,
        beta: float,
        evaluator: Any,
        move_cfg: Dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """
        Choose one move (S, B, or M) based on self.move_probs, then apply it.
        Uses evaluator.combined(state) with Metropolis acceptance scaled by beta.
        """
        L = seq.size
        move_kind = self._sample_move_kind(rng)
        self.move_tally[move_kind] += 1

        if move_kind == "S":
            # Single‐nucleotide flip
            i = rng.integers(L)
            old_base = seq[i]
            lods = np.empty(4, float)
            logger.debug("Performing 'S' move at position %d", i)

            for b in range(4):
                seq[i] = b
                lods[b] = beta * evaluator.combined(SequenceState(seq.copy()))
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

        elif move_kind == "B":
            # Block replacement
            min_len, max_len = move_cfg["block_len_range"]
            length = rng.integers(min_len, max_len + 1)
            start = rng.integers(0, L - length + 1)
            proposal = rng.integers(0, 4, size=length)

            old_block = seq[start : start + length].copy()
            logger.debug(
                "Performing 'B' move: replace block [%d:%d] (length=%d)",
                start,
                start + length,
                length,
            )
            _replace_block(seq, start, length, proposal)
            new_comb = evaluator.combined(SequenceState(seq.copy()))
            _replace_block(seq, start, length, old_block)
            old_comb = evaluator.combined(SequenceState(seq.copy()))

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
            else:
                logger.debug("    Rejecting 'B' move, reverting")

        else:  # move_kind == "M"
            # Multi‐site flips
            kmin, kmax = move_cfg["multi_k_range"]
            k = rng.integers(kmin, kmax + 1)
            idxs = rng.choice(L, size=k, replace=False)
            old_bases = seq[idxs].copy()
            proposal = rng.integers(0, 4, size=k)

            seq[idxs] = proposal
            new_comb = evaluator.combined(SequenceState(seq.copy()))
            seq[idxs] = old_bases
            old_comb = evaluator.combined(SequenceState(seq.copy()))

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
            else:
                logger.debug("    Rejecting 'M' move, reverting to old_bases")

    def _sample_move_kind(self, rng: np.random.Generator) -> str:
        """
        Sample a move‐kind among {"S","B","M"} according to self.move_probs.
        """
        kinds = ["S", "B", "M"]
        idx = rng.choice(3, p=self.move_probs)
        choice = kinds[idx]
        logger.debug(
            "    Sampled move_kind=%s (probs=%s)",
            choice,
            dict(S=self.move_probs[0], B=self.move_probs[1], M=self.move_probs[2]),
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
