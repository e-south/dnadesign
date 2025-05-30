"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/optimizer/cgm.py

Gibbs/Metropolis sampler with cooling and move mixture.

Move types
----------
S   : single-nucleotide flip (Gibbs)                 - high acceptance, low step
B   : contiguous block replacement (Gibbs)           - escapes shallow basins
M   : k disjoint flips  (Gibbs)                      - mixes mid-exploration
SL  : window slide  (MH)                             - optional, TODO
SW  : substring swap/reverse (MH)                    - optional, TODO

The move probabilities evolve with β (annealing) as described in the spec.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict

import arviz as az
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from dnadesign.cruncher.sample.state import SequenceState

from .base import Optimizer

logger = logging.getLogger(__name__)


# Helper: contiguous block replacement (compiled, inner-loop speed critical)
@njit
def _replace_block(seq: np.ndarray, start: int, length: int, proposal: np.ndarray):
    seq[start : start + length] = proposal


class GibbsOptimizer(Optimizer):
    """Cooling Gibbs/Metropolis sampler producing diverse motif-rich sequences."""

    #  PUBLIC ENTRY POINT
    def optimise(self, initial: SequenceState) -> list[SequenceState]:
        cfg = self.cfg  # shorthand
        rng = self.rng

        # pre-extract config for speed
        draws, tune, chains = cfg["draws"], cfg["tune"], cfg["chains"]
        min_dist, top_k = cfg["min_dist"], cfg["top_k"]
        cooling_cfg = cfg["cooling"]
        move_cfg = cfg["moves"]

        #  Pre-compute cooling schedule helper
        if cooling_cfg["kind"] == "fixed":

            def beta_of(iter_idx: int) -> float:  # noqa: D401
                return cooling_cfg["beta"]

        else:  # piece-wise
            stages = sorted(cooling_cfg["stages"], key=lambda s: s["sweeps"])
            sweep_pts = [s["sweeps"] for s in stages]
            betas = [s["beta"] for s in stages]

            def beta_of(iter_idx: int) -> float:  # noqa: D401
                # rightmost stage
                if iter_idx >= sweep_pts[-1]:
                    return betas[-1]
                j = np.searchsorted(sweep_pts, iter_idx, side="right") - 1
                t0, t1 = sweep_pts[j], sweep_pts[j + 1]
                b0, b1 = betas[j], betas[j + 1]
                # linear interp
                return b0 + (b1 - b0) * ((iter_idx - t0) / (t1 - t0))

        # shorthand to motif scorer
        score_fn = self.scorer.score
        per_pwm_fn = self.scorer.score_per_pwm

        # diagnostics
        move_tally = Counter()

        # storage
        chain_scores: list[list[float]] = []
        all_samples: list[np.ndarray] = []
        records: list[Dict] = []
        global_iter = 0

        for c in range(chains):
            seq = initial.seq.copy()
            L = seq.size
            # burn-in + draws per chain
            total_sweeps = tune + draws
            chain_trace: list[float] = []

            for sweep in tqdm(range(total_sweeps), desc=f"chain{c+1}", leave=False):
                beta = beta_of(global_iter)
                # choose move kind
                move_kind = self._sample_move_kind(beta, move_cfg, rng)
                move_tally[move_kind] += 1

                if move_kind == "S":  # single flip
                    i = rng.integers(L)
                    lods = np.empty(4, float)
                    old_base = seq[i]
                    for b in range(4):
                        seq[i] = b
                        lods[b] = beta * score_fn(seq)
                    seq[i] = old_base  # restore
                    # numerically stable soft-max
                    lods -= lods.max()
                    probs = np.exp(lods)
                    seq[i] = rng.choice(4, p=probs / probs.sum())

                elif move_kind == "B":  # contiguous block
                    min_len, max_len = move_cfg["block_len_range"]
                    length = rng.integers(min_len, max_len + 1)
                    start = rng.integers(0, L - length + 1)
                    proposal = rng.integers(0, 4, size=length)
                    # Gibbs update over the *entire proposal* jointly:
                    #   P(new_block) ∝ exp(β·score)
                    old_block = seq[start : start + length].copy()
                    _replace_block(seq, start, length, proposal)
                    new_score = score_fn(seq)
                    _replace_block(seq, start, length, old_block)
                    old_score = score_fn(seq)
                    logp_new = beta * new_score
                    logp_old = beta * old_score
                    if rng.random() < np.exp(logp_new - logp_old):
                        _replace_block(seq, start, length, proposal)

                elif move_kind == "M":  # k disjoint flips
                    kmin, kmax = move_cfg["multi_k_range"]
                    k = rng.integers(kmin, kmax + 1)
                    idx = rng.choice(L, size=k, replace=False)
                    old = seq[idx].copy()
                    proposal = rng.integers(0, 4, size=k)
                    seq[idx] = proposal
                    new_score = score_fn(seq)
                    seq[idx] = old
                    old_score = score_fn(seq)
                    logp_new = beta * new_score
                    logp_old = beta * old_score
                    if rng.random() < np.exp(logp_new - logp_old):
                        seq[idx] = proposal

                #  Record diagnostics
                if sweep >= tune:  # draw phase
                    all_samples.append(seq.copy())
                    chain_trace.append(score_fn(seq))

                if sweep % 10 == 0:  # cheap – store every 10th sweep
                    per_pwm = per_pwm_fn(seq)
                    records.append(
                        dict(
                            chain=c,
                            iter=global_iter,
                            beta=beta,
                            **{f"score_{tf}": float(v) for tf, v in zip(self.scorer.logodds, per_pwm)},
                        )
                    )
                global_iter += 1

            chain_scores.append(chain_trace)

        #  Final selection & diagnostics
        logger.info("Move utilisation: %s", dict(move_tally))

        # diverse elite filter
        scored = sorted(((self.scorer.score(s), s) for s in all_samples), key=lambda t: t[0], reverse=True)
        elites: list[SequenceState] = []
        for score_val, seq in scored:
            if len(elites) == top_k:
                break
            if any(np.sum(seq != e.seq) < min_dist for e in elites):
                continue
            elites.append(SequenceState(seq.copy()))

        self.samples_df = pd.DataFrame(records)
        self.trace_idata = az.from_dict(posterior={"score": np.asarray(chain_scores)})

        return elites

    #  helpers
    @staticmethod
    def _sample_move_kind(beta: float, cfg: Dict, rng) -> str:
        """Return one of 'S','B','M' according to β-dependent probabilities."""
        beta_target = 1.0
        r = min(beta / beta_target, 1.0)
        probs = {
            "S": 0.4 + 0.4 * r,
            "B": 0.4 * (1 - r),
            "M": 0.2,
        }
        kinds = list(probs)
        weights = np.fromiter(probs.values(), dtype=float)
        weights /= weights.sum()
        return rng.choice(kinds, p=weights)
