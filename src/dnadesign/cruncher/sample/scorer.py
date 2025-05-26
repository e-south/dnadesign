"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/scorer.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from dnadesign.cruncher.motif.model import PWM

class Scorer:
    def __init__(self, pwms: Dict[str, PWM]):
        self.pwms = pwms
        self.logodds: Dict[str, np.ndarray] = {}
        for name, pwm in pwms.items():
            if pwm.log_odds_matrix is not None:
                # use the parsed block verbatim
                self.logodds[name] = pwm.log_odds_matrix
            else:
                # fallback compute
                self.logodds[name] = pwm.log_odds()

    def score_per_pwm(self, seq: np.ndarray) -> np.ndarray:
        """For each PWM, slide it along seq and take the best-matching log-odds sum."""
        scores: List[float] = []
        L = seq.size
        for name, lom in self.logodds.items():
            m = lom.shape[0]
            if m > L:
                scores.append(-np.inf)
                continue
            # slide‐window dot-product
            window_scores = np.array([
                lom[:, seq[i : i + m]].sum()
                for i in range(L - m + 1)
            ], dtype=float)
            scores.append(window_scores.max())
        return np.array(scores, dtype=float)

    def score(self, seq: np.ndarray) -> float:
        """Overall objective = minimum of individual PWM best-scores."""
        per = self.score_per_pwm(seq)
        return float(np.min(per))