"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/cgm.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
from tqdm import tqdm
import numpy as np
from numba import njit

from .base import Optimizer
from dnadesign.cruncher.sample.state import SequenceState

logger = logging.getLogger(__name__)

# Numba-compiled helper: slide one PWM log-odds matrix over seq → best score
@njit
def best_score_pwm(seq: np.ndarray, lom: np.ndarray) -> float:
    L = seq.shape[0]
    m = lom.shape[0]
    best = -1e12
    for offset in range(L - m + 1):
        s = 0.0
        for j in range(m):
            s += lom[j, seq[offset + j]]
        if s > best:
            best = s
    return best

class GibbsOptimizer(Optimizer):
    """
    Pure-Python, one-site-at-a-time Gibbs sampler over {0,1,2,3}^L,
    with Numba–accelerated PWM scoring, and a minimum-Hamming-distance
    constraint on the final top-k.
    """

    def optimise(self, initial: SequenceState) -> list[SequenceState]:
        # 1) Extract each PWM’s log-odds matrix into a Python list
        mats = list(self.scorer.logodds.values())

        # 2) Unpack parameters
        seq0   = initial.seq.copy()
        L      = seq0.size
        β      = self.cfg["beta"]
        draws  = self.cfg["draws"]
        tune   = self.cfg["tune"]
        chains = self.cfg["chains"]
        top_k  = self.cfg["top_k"]
        min_dist = self.cfg.get("min_dist", 1)
        rng    = self.rng

        # helper: compute overall sequence score = min over all PWMs
        def seq_score(x: np.ndarray) -> float:
            bests = np.empty(len(mats), dtype=np.float64)
            for i, lom in enumerate(mats):
                bests[i] = best_score_pwm(x, lom)
            return float(np.min(bests))

        all_samples = []

        # 3) Gibbs chains
        for c in range(chains):
            logger.info(f"Chain {c+1}/{chains}: tuning {tune} sweeps…")
            x = seq0.copy()
            # tuning sweeps (not recorded)
            for _ in tqdm(range(tune),
                          desc=f"Chain {c+1} tune",
                          unit="sweep",
                          leave=False):
                for i in range(L):
                    # compute log-posterior for each base
                    logps = np.empty(4, dtype=np.float64)
                    for b in range(4):
                        x[i] = b
                        logps[b] = β * seq_score(x)
                    a = logps.max()
                    probs = np.exp(logps - a)
                    probs /= probs.sum()
                    x[i] = rng.choice(4, p=probs)

            logger.info(f"Chain {c+1}/{chains}: sampling {draws} sweeps…")
            samples = []
            pbar = tqdm(range(draws),
                        desc=f"Chain {c+1} sample",
                        unit="sweep",
                        leave=False)
            for j in pbar:
                for i in range(L):
                    logps = np.empty(4, dtype=np.float64)
                    for b in range(4):
                        x[i] = b
                        logps[b] = β * seq_score(x)
                    a = logps.max()
                    probs = np.exp(logps - a)
                    probs /= probs.sum()
                    x[i] = rng.choice(4, p=probs)
                samples.append(x.copy())
                if j % max(1, draws // 10) == 0:
                    pbar.set_postfix(score=f"{seq_score(x):.2f}")

            all_samples.extend(samples)

        # 4) Rank + enforce diversity
        #    sort by descending score
        scored = sorted(
            ((seq_score(s), s) for s in all_samples),
            key=lambda t: t[0],
            reverse=True,
        )

        elites: list[SequenceState] = []
        for score_val, seq in scored:
            if len(elites) >= top_k:
                break
            # enforce min Hamming distance
            too_close = False
            for e in elites:
                if np.sum(seq != e.seq) < min_dist:
                    too_close = True
                    break
            if too_close:
                logger.debug(f"Skipping seq {seq} (score={score_val:.2f})—too close")
                continue

            elites.append(SequenceState(seq=seq.copy()))

        if len(elites) < top_k:
            logger.warning(
                f"Requested top_k={top_k} but only found {len(elites)} diverse sequences."
            )

        return elites