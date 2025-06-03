"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/optimizer/pt.py

Parallel-Tempered Gibbs/Metropolis optimiser.

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
from tqdm import trange

from dnadesign.cruncher.sample.optimizer.base import Optimizer
from dnadesign.cruncher.sample.optimizer.cooling import make_beta_ladder
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.state import SequenceState, make_seed

logger = logging.getLogger(__name__)


class PTGibbsOptimizer(Optimizer):
    """
    Parallel-Tempering Gibbs sampler with the same `optimise()` API as GibbsOptimizer.

    Expected cfg keys:
      • draws       : int
      • tune        : int
      • chains      : int         (must equal len(beta_ladder))
      • min_dist    : int
      • top_k       : int
      • block_len_range  : (int,int)
      • multi_k_range    : (int,int)
      • slide_max_shift  : int
      • swap_len_range   : (int,int)
      • kind        : “geometric” or “fixed”  (from cooling block)
      • beta        : List[float] or float
      • swap_prob   : float
      • softmax_beta: float   (required if kind == "pt")
    """

    def __init__(self, scorer: Scorer, cfg: Dict[str, Any], rng, *, pwms: Dict[str, "PWM"], init_cfg):
        """
        scorer:   instance of Scorer
        cfg:      flattened dict containing keys above
        rng:      numpy.random.Generator
        pwms:     dictionary of name→PWM (for random baseline)
        init_cfg: InitConfig (for consensus seeding in other chains)
        """
        super().__init__(scorer, cfg, rng)
        self._pwms = pwms
        self._init_cfg = init_cfg

        self.draws = int(cfg["draws"])
        self.tune = int(cfg["tune"])
        self.chains = int(cfg["chains"])
        self.min_dist = int(cfg["min_dist"])
        self.top_k = int(cfg["top_k"])
        self.swap_prob = float(cfg["swap_prob"])
        self.softmax_beta = float(cfg["softmax_beta"])

        # Move kernel parameters (shared with Gibbs)
        self.block_len_range = tuple(cfg["block_len_range"])
        self.multi_k_range = tuple(cfg["multi_k_range"])
        self.slide_max_shift = int(cfg["slide_max_shift"])
        self.swap_len_range = tuple(cfg["swap_len_range"])

        self.samples_df: pd.DataFrame | None = None
        self.trace_idata = None
        self.random_df: pd.DataFrame | None = None

    def optimise(self, initial: SequenceState) -> List[SequenceState]:
        """
        Main PT loop. Runs `chains` parallel chains, each at a different β from the ladder,
        performing intra-chain moves & inter-chain swaps, then returns “elites.”
        """
        scorer = self.scorer
        rng = self.rng

        # Build β‐ladder (list of floats)
        betas = make_beta_ladder(self.cfg["cooling"])
        if len(betas) != self.chains:
            raise ValueError("In PT: chains must equal len(beta_ladder)")

        # Seed each chain: first chain gets `initial`; others get make_seed(...)
        states = [initial] + [make_seed(self._init_cfg, self._pwms, rng) for _ in range(self.chains - 1)]
        energies = np.array([scorer.score(s.seq) for s in states])

        draws, tune = self.draws, self.tune
        total = draws + tune
        records: List[Dict[str, Any]] = []
        move_ct: Counter = Counter()

        # ─── Parallel‐tempering sweeps ───────────────────────────────────────────
        for sweep in trange(total, desc="pt-gibbs"):
            # ── Within‐chain moves ────────────────────────────────────────────────
            for t_idx, beta in enumerate(betas):
                seq = states[t_idx].seq.copy()
                L = seq.size

                # Randomly choose either single flip (50%) or block replacement (50%)
                if rng.random() < 0.5:
                    pos = rng.integers(L)
                    # single flip: pick a random new base different from current
                    seq[pos] = (seq[pos] + 1 + rng.integers(3)) % 4
                else:
                    # block replacement
                    blk_len = rng.integers(*self.block_len_range)
                    start = rng.integers(0, L - blk_len + 1)
                    seq[start : start + blk_len] = rng.integers(0, 4, size=blk_len)

                # MH acceptance: if beta<1 use soft‐min; if beta=1 use direct score
                new_E = scorer.soft_min(seq, self.softmax_beta) if beta < 1.0 else scorer.score(seq)
                delta = beta * (new_E - energies[t_idx])
                if delta >= 0 or np.log(rng.random()) < delta:
                    states[t_idx] = SequenceState(seq)
                    energies[t_idx] = new_E

            # ── Between‐chain swaps ──────────────────────────────────────────────
            for i in range(self.chains - 1):
                if rng.random() > self.swap_prob:
                    continue
                d = (betas[i + 1] - betas[i]) * (energies[i + 1] - energies[i])
                if d >= 0 or np.log(rng.random()) < d:
                    states[i], states[i + 1] = states[i + 1], states[i]
                    energies[i], energies[i + 1] = energies[i + 1], energies[i]

            # Record only after burn‐in
            if sweep >= tune:
                for t_idx, state in enumerate(states):
                    records.append({"iter": sweep, "chain": t_idx, "beta": betas[t_idx], "fitness": energies[t_idx]})

        # ─── Elite selection & diagnostics ─────────────────────────────────────
        ranked = sorted(zip(energies, states), key=lambda t: t[0], reverse=True)
        elites = [s for _, s in ranked[: self.top_k]]

        self.samples_df = pd.DataFrame(records)
        self.trace_idata = az.from_dict(posterior={"fitness": np.array(energies)[None, :]})

        # Build a random baseline (for scatter plot) of equal length to samples_df
        rand_scores = []
        rng2 = np.random.default_rng(0)
        for _ in range(len(self.samples_df)):
            x = SequenceState.random(initial.seq.size, rng2)
            pwm_scores = scorer.score_per_pwm(x.seq)
            rand_scores.append(
                {"iter": len(rand_scores), **{f"score_{tf}": float(s) for tf, s in zip(self._pwms, pwm_scores)}}
            )
        self.random_df = pd.DataFrame(rand_scores)

        return elites
