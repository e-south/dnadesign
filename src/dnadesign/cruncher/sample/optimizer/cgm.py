"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/optimizer/cgm.py

Gibbs/Metropolis sampler with linear or piecewise cooling.

Supported move types:
  - S: single-nucleotide flip (Gibbs)
  - B: contiguous block replacement (Gibbs)
  - M: k disjoint flips (Gibbs)
Reserved moves: SL (slide), SW (swap).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

import arviz as az
import numpy as np
import pandas as pd
from tqdm import tqdm

from dnadesign.cruncher.sample.optimizer.base import Optimizer
from dnadesign.cruncher.sample.optimizer.cooling import make_beta_scheduler
from dnadesign.cruncher.sample.optimizer.helpers import _replace_block
from dnadesign.cruncher.sample.state import SequenceState

logger = logging.getLogger(__name__)


class GibbsOptimizer(Optimizer):
    """
    Cooling Gibbs/Metropolis sampler producing diverse motif-rich sequences.

    Expected cfg keys:
      - draws: int
      - tune: int
      - chains: int
      - min_dist: int
      - top_k: int
      - block_len_range: (int,int)
      - multi_k_range: (int,int)
      - slide_max_shift: int
      - swap_len_range: (int,int)
      - kind, beta, stages (from cooling)
      - swap_prob: float
    """

    def __init__(self, scorer: Any, cfg: Dict[str, Any], rng):
        super().__init__(scorer, cfg, rng)

        self.draws = int(cfg["draws"])
        self.tune = int(cfg["tune"])
        self.chains = int(cfg["chains"])
        self.min_dist = int(cfg["min_dist"])
        self.top_k = int(cfg["top_k"])
        self.swap_prob = float(cfg["swap_prob"])

        self.move_cfg = {
            "block_len_range": tuple(cfg["block_len_range"]),
            "multi_k_range": tuple(cfg["multi_k_range"]),
            "slide_max_shift": int(cfg["slide_max_shift"]),
            "swap_len_range": tuple(cfg["swap_len_range"]),
        }

        total = self.tune + self.draws
        cooling_cfg = {k: v for k, v in cfg.items() if k in ("kind", "beta", "stages")}
        self.beta_of = make_beta_scheduler(cooling_cfg, total)

        self.move_tally: Counter = Counter()
        self.samples_df: pd.DataFrame | None = None
        self.trace_idata = None

    def optimise(self, initial: SequenceState) -> List[SequenceState]:
        """
        Run MCMC in two phases:
          1) Burn-in: tune sweeps (no recording)
          2) Sampling: draws sweeps (record each state)

        Returns a list of elite SequenceState objects (max K, Hamming-distinct).
        """
        rng = self.rng
        scorer = self.scorer

        tune = self.tune
        draws = self.draws
        chains = self.chains
        min_dist = self.min_dist
        top_k = self.top_k
        move_cfg = self.move_cfg

        all_samples: List[np.ndarray] = []
        chain_scores: List[List[float]] = []
        records: List[Dict[str, Any]] = []

        for c in range(chains):
            seq = initial.seq.copy()
            L = seq.size
            chain_trace: List[float] = []
            global_iter = 0

            # Phase 1: Burn-in
            for _ in tqdm(range(tune), desc=f"chain{c+1} burn-in", leave=False):
                beta = self.beta_of(global_iter)
                self._perform_single_move(seq, beta, scorer, move_cfg, rng)
                global_iter += 1

            # Phase 2: Sampling
            for _ in tqdm(range(draws), desc=f"chain{c+1} sampling", leave=False):
                beta = self.beta_of(global_iter)
                self._perform_single_move(seq, beta, scorer, move_cfg, rng)

                # Record sampled state and score
                all_samples.append(seq.copy())
                chain_trace.append(scorer.score(seq))

                global_iter += 1

                # Every 10 iterations, record per-PWM scaled scores
                if global_iter % 10 == 0:
                    scaled_scores: List[float] = []
                    L = seq.size
                    for tfname, info in scorer._cache.items():
                        raw_llr, _extras, _, _ = scorer._best_llr_and_extra_hits_and_location(seq, info)
                        val = scorer._scale_llr(raw_llr, info, seq_length=L)
                        scaled_scores.append(val)
                    records.append(
                        {
                            "chain": c,
                            "iter": global_iter,
                            "beta": beta,
                            **{f"score_{tf}": scaled for tf, scaled in zip(scorer._cache.keys(), scaled_scores)},
                        }
                    )

            chain_scores.append(chain_trace)

        logger.info("Move utilization: %s", dict(self.move_tally))

        # Rank all sampled sequences by raw fitness
        scored = sorted(
            ((scorer.score(arr), arr) for arr in all_samples),
            key=lambda pair: pair[0],
            reverse=True,
        )

        elites: List[SequenceState] = []
        for score_val, seq_arr in scored:
            if len(elites) >= top_k:
                break
            if any(np.sum(seq_arr != e.seq) < min_dist for e in elites):
                continue
            elites.append(SequenceState(seq_arr.copy()))

        self.samples_df = pd.DataFrame(records)
        self.trace_idata = az.from_dict(posterior={"score": np.asarray(chain_scores)})

        return elites

    def _perform_single_move(
        self,
        seq: np.ndarray,
        beta: float,
        scorer: Any,
        move_cfg: Dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        """
        Choose one move (S, B, or M) based on β-dependent probabilities, and apply to `seq`.
        """
        L = seq.size
        move_kind = self._sample_move_kind(beta, move_cfg, rng)
        self.move_tally[move_kind] += 1

        if move_kind == "S":
            # Single-nucleotide flip
            i = rng.integers(L)
            old_base = seq[i]
            lods = np.empty(4, float)
            for b in range(4):
                seq[i] = b
                lods[b] = beta * scorer.score(seq)
            seq[i] = old_base
            lods -= lods.max()
            probs = np.exp(lods)
            seq[i] = rng.choice(4, p=probs / probs.sum())

        elif move_kind == "B":
            # Block replacement
            min_len, max_len = move_cfg["block_len_range"]
            length = rng.integers(min_len, max_len + 1)
            start = rng.integers(0, L - length + 1)
            proposal = rng.integers(0, 4, size=length)

            old_block = seq[start : start + length].copy()
            _replace_block(seq, start, length, proposal)
            new_score = scorer.score(seq)
            _replace_block(seq, start, length, old_block)
            old_score = scorer.score(seq)

            logp_new = beta * new_score
            logp_old = beta * old_score
            delta = logp_new - logp_old
            if delta >= 0 or np.log(rng.random()) < delta:
                _replace_block(seq, start, length, proposal)

        else:  # move_kind == "M"
            # Multi-site flips
            kmin, kmax = move_cfg["multi_k_range"]
            k = rng.integers(kmin, kmax + 1)
            idx = rng.choice(L, size=k, replace=False)
            old_bases = seq[idx].copy()
            proposal = rng.integers(0, 4, size=k)
            seq[idx] = proposal
            new_score = scorer.score(seq)
            seq[idx] = old_bases
            old_score = scorer.score(seq)

            logp_new = beta * new_score
            logp_old = beta * old_score
            delta = logp_new - logp_old
            if delta >= 0 or np.log(rng.random()) < delta:
                seq[idx] = proposal

    @staticmethod
    def _sample_move_kind(beta: float, move_cfg: Dict[str, Any], rng: np.random.Generator) -> str:
        """
        Return one of "S","B","M" with β-dependent weights:
          - β=0 → exploratory: P(S)=P(B)=0.4, P(M)=0.2
          - β→1+ → fine-tuning: P(S)→0.8, P(B)→0, P(M)=0.2
        """
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
