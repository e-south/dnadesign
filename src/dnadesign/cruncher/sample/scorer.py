"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/scorer.py

FIMO-like Scorer: from raw PWM log-odds to calibrated p-value fitness.

1) Precompute:
   - log-odds matrix under uniform background
   - exact null distribution and tail p-values via DP lookup
2) For each sequence:
   - Slide PWMs (and reverse complement) with Numba → best log-odds per motif
   - Map best score → p-value by binary search + array indexing
   - Return -log10(min p-value) as the fitness objective

Inspired by Grant et al. 2011 (DOI: 10.1093/bioinformatics/btr064).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.pvalue import logodds_to_p_lookup
from dnadesign.cruncher.sample.numba_helpers import best_score_pwm


class Scorer:
    """
    Scores DNA sequences against multiple PWMs using exact p-values.

    Attributes:
      logodds: dict of motif_name→(Lx4) log-odds arrays
      lookups: dict of motif_name→(scores, tail_p) arrays for p-value mapping
      bidirectional: whether to scan both strands
    """

    def __init__(
        self,
        pwms: Dict[str, PWM],
        background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        bidirectional: bool = True,
    ):
        self.bidirectional = bidirectional
        bg = np.array(background, float)
        # Precompute log-odds and DP-based p-value lookups
        self.logodds = {name: pwm.log_odds(background=bg) for name, pwm in pwms.items()}
        self.lookups = {name: logodds_to_p_lookup(lom, bg) for name, lom in self.logodds.items()}

    def score_per_pwm(self, seq: np.ndarray) -> np.ndarray:
        """
        Get the maximum log-odds score per PWM over the sequence.
        """
        scores = []
        rc = None
        if self.bidirectional:
            rc = (3 - seq)[::-1]
        for lom in self.logodds.values():
            L, w = seq.size, lom.shape[0]
            if w > L:
                scores.append(-np.inf)
                continue
            fwd = best_score_pwm(seq, lom)
            if rc is not None:
                rev = best_score_pwm(rc, lom)
                scores.append(max(fwd, rev))
            else:
                scores.append(fwd)
        return np.array(scores)

    def _interp_p(self, score: float, scores: np.ndarray, tail: np.ndarray) -> float:
        """
        Map a raw score to a tail p-value via binary search + indexing.
        """
        idx = np.searchsorted(scores, score, "right") - 1
        idx = np.clip(idx, 0, len(tail) - 1)
        return float(tail[idx])

    def score(self, seq: np.ndarray) -> float:
        """
        Compute fitness = -log10(min p-value across all PWMs).
        """
        lods = self.score_per_pwm(seq)
        pvals = [self._interp_p(lod, *self.lookups[name]) for name, lod in zip(self.logodds, lods)]
        return -np.log10(min(pvals))

    def score_pvals(self, seq: np.ndarray) -> Dict[str, float]:
        """
        Return per-PWM p-values for reporting or thresholding.
        """
        lods = self.score_per_pwm(seq)
        return {name: self._interp_p(lod, *self.lookups[name]) for name, lod in zip(self.logodds, lods)}
