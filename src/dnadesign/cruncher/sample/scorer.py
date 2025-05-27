"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/sample/scorer.py

Scorer uses Numba helper for per-PWM sliding-window scoring.
Supports optional bidirectional (reverse-complement) mode.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict
import numpy as np
from dnadesign.cruncher.sample.numba_helpers import best_score_pwm
from dnadesign.cruncher.motif.model import PWM

class Scorer:
    """
    Given a set of PWMs, compute how well any candidate DNA sequence
    matches each PWM.  The overall objective is the minimum (worst) of
    per-PWM best-match scores, but you can also retrieve each PWM's score.

    If bidirectional=True, we also scan the reverse-complement of the
    sequence and take the higher of forward- or reverse-match.
    """

    def __init__(self, pwms: Dict[str, PWM], bidirectional: bool = True):
        # Whether to also consider the reverse-complement of each sequence
        self.bidirectional = bidirectional

        # Precompute each PWM’s log‐odds matrix so we never recompute it on the fly.
        # If the PWM file itself provided a log‐odds block, we use that directly;
        # otherwise we compute log2(prob/background) from the probability matrix.
        self.logodds: Dict[str, np.ndarray] = {
            name: (
                pwm.log_odds_matrix
                if pwm.log_odds_matrix is not None
                else pwm.log_odds()
            )
            for name, pwm in pwms.items()
        }

    def score_per_pwm(self, seq: np.ndarray) -> np.ndarray:
        """
        For each PWM, slide its log‐odds matrix across the given sequence
        (and optionally its reverse‐complement), returning an array of
        best‐match scores—one per PWM.

        Args:
            seq: integer‐encoded DNA (0=A,1=C,2=G,3=T)

        Returns:
            1D numpy array of length = # of PWMs, with each entry the
            maximum log‐odds sum over all valid alignments.
        """
        scores = []
        # If bidirectional, compute the reverse‐complement once:
        if self.bidirectional:
            # rc: flip each base 0<->3, 1<->2, then reverse order
            rc = (3 - seq)[::-1]

        for lom in self.logodds.values():
            motif_length = lom.shape[0]

            # If the motif is longer than the sequence, it can never match:
            if motif_length > seq.size:
                scores.append(-np.inf)
                continue

            # Compute forward‐strand best match via Numba helper:
            forward_score = best_score_pwm(seq, lom)

            if self.bidirectional:
                # And reverse‐strand best match:
                reverse_score = best_score_pwm(rc, lom)
                # Take the better of the two
                scores.append(max(forward_score, reverse_score))
            else:
                scores.append(forward_score)

        return np.array(scores, dtype=float)

    def score(self, seq: np.ndarray) -> float:
        """
        Overall scoring function: take the worst‐match across all PWMs.
        This makes the sampler maximize the minimum PWM score, effectively
        forcing every PWM to bind at least moderately well.

        Args:
            seq: integer‐encoded DNA (0=A,1=C,2=G,3=T)

        Returns:
            Single float = min(score_per_pwm(seq))
        """
        per_pwm = self.score_per_pwm(seq)
        return float(per_pwm.min())