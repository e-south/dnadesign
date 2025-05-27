"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/cgm.py

Numba-accelerated Gibbs sampler (one-site updates) behind an Optimizer interface.

Hyperparams:
 - β: inverse temperature
 - tune: burn-in sweeps
 - draws: recorded sweeps
 - chains: independent chains
 - min_dist: Hamming diversity
 - top_k: number of elites to return

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import arviz as az
from .base import Optimizer
from dnadesign.cruncher.sample.state import SequenceState

logger = logging.getLogger(__name__)

class GibbsOptimizer(Optimizer):
    """
    Gibbs sampler that records every sweep (initial state + burn-in + sampling)
    in self.samples_df.

    Config keys in self.cfg:
      - beta, tune, draws, chains, top_k, min_dist
    """
    def __init__(self, scorer, cfg, rng):
        super().__init__(scorer, cfg, rng)
        # Keep the TF names in the same order as the scorer’s logodds
        self.tf_names = list(self.scorer.logodds.keys())

    def optimise(self, initial: SequenceState) -> list[SequenceState]:
        β        = self.cfg["beta"]
        tune     = self.cfg["tune"]
        draws    = self.cfg["draws"]
        chains   = self.cfg["chains"]
        top_k    = self.cfg["top_k"]
        min_dist = self.cfg.get("min_dist", 1)
        rng      = self.rng

        L = initial.seq.size
        all_samples  = []
        chain_scores = []
        proposals = accepts = 0

        records = []       # one dict per iteration (including pre–burn-in)
        global_iter = 0    # absolute iteration counter across burn-in+sampling

        for c in range(chains):
            # ——— fresh random start each chain ———
            x = SequenceState.random(L, rng).seq.copy()

            # record the *true* random seed before any burn-in
            init_scores = self.scorer.score_per_pwm(x)
            records.append({
                "chain": c,
                "iter": global_iter,
                **{f"score_{tf}": float(val)
                   for tf, val in zip(self.tf_names, init_scores)}
            })
            global_iter += 1

            this_chain = []
            logger.info(f"Chain {c+1}/{chains}: tuning {tune} sweeps…")

            # ——— Burn-in sweeps (and record each sweep) ———
            for _ in tqdm(range(tune), desc=f"Tune {c+1}", leave=False):
                # (insert your block-Gibbs or swap logic here;
                #  for brevity we show one-site updates)
                for i in range(L):
                    logps = np.empty(4, dtype=float)
                    for b in range(4):
                        x[i] = b
                        logps[b] = β * self.scorer.score(x)
                    a    = logps.max()
                    raw  = np.exp(logps - a)
                    tot  = raw.sum()
                    probs = raw/tot if (np.isfinite(tot) and tot>0) else np.ones(4)/4

                    prev = x[i]
                    new  = rng.choice(4, p=probs)
                    proposals += 1
                    accepts   += (new != prev)
                    x[i] = new

                # record per-PWM at end of this burn-in sweep
                burn_scores = self.scorer.score_per_pwm(x)
                records.append({
                    "chain": c,
                    "iter": global_iter,
                    **{f"score_{tf}": float(val)
                       for tf, val in zip(self.tf_names, burn_scores)}
                })
                global_iter += 1

            # ——— Sampling sweeps (and record each sweep) ———
            logger.info(f"Chain {c+1}/{chains}: sampling {draws} sweeps…")
            for _ in tqdm(range(draws), desc=f"Sample {c+1}", leave=False):
                # again your Gibbs or block-swap moves...
                for i in range(L):
                    logps = np.empty(4, dtype=float)
                    for b in range(4):
                        x[i] = b
                        logps[b] = β * self.scorer.score(x)
                    a    = logps.max()
                    raw  = np.exp(logps - a)
                    tot  = raw.sum()
                    probs = raw/tot if (np.isfinite(tot) and tot>0) else np.ones(4)/4

                    prev = x[i]
                    new  = rng.choice(4, p=probs)
                    proposals += 1
                    accepts   += (new != prev)
                    x[i] = new

                # track overall score for ArviZ trace
                s = self.scorer.score(x)
                this_chain.append(s)
                all_samples.append(x.copy())

                # record per-PWM at end of this sampling sweep
                samp_scores = self.scorer.score_per_pwm(x)
                records.append({
                    "chain": c,
                    "iter": global_iter,
                    **{f"score_{tf}": float(val)
                       for tf, val in zip(self.tf_names, samp_scores)}
                })
                global_iter += 1

            chain_scores.append(this_chain)

        logger.info(f"Gibbs acceptance rate: {accepts}/{proposals} = {accepts/proposals:.1%}")

        # build a DataFrame of every iteration (init + burn-in + samples)
        self.samples_df = pd.DataFrame(records)

        # ——— select elites exactly as before ———
        scored = sorted(
            ((self.scorer.score(s), s) for s in all_samples),
            key=lambda t: t[0], reverse=True
        )
        elites = []
        for score_val, seq in scored:
            if len(elites) >= top_k:
                break
            if any(np.sum(seq != e.seq) < min_dist for e in elites):
                continue
            elites.append(SequenceState(seq=seq.copy()))
        if len(elites) < top_k:
            logger.warning(f"Requested top_k={top_k} but only found {len(elites)} diverse sequences.")

        # package trace for ArviZ
        scores_arr = np.array(chain_scores)  # shape = (chains, draws)
        self.trace_idata = az.from_dict(posterior={"score": scores_arr})

        return elites