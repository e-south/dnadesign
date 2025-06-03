"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/sample/transforms.py

Post-hoc transforms to convert raw LLRs into derived metrics (logp_norm, z-score, etc.).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.pvalue import logodds_to_p_lookup


class Transform:
    """
    Base interface for any “post-hoc” transform.
    Given raw LLRs per TF and sequence length, return derived values.
    """

    def __call__(self, raw_llr_map: Dict[str, float], seq_length: int) -> Dict[str, float]: ...


class LogPNorm(Transform):
    """
    Compute logp_norm = −log10(p_seq) / −log10(p_consensus) for each TF.
    """

    def __init__(
        self,
        pwms: Dict[str, PWM],
        bidirectional: bool,
        background=(0.25, 0.25, 0.25, 0.25),
    ):
        self.pwms = pwms
        self.bidirectional = bidirectional
        self.bg = np.asarray(background, dtype=float)

    def __call__(self, raw_llr_map: Dict[str, float], seq_length: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for tf, raw_llr in raw_llr_map.items():
            pwm = self.pwms[tf]
            lom = pwm.log_odds()
            null_scores, tail_p = logodds_to_p_lookup(lom, self.bg)
            w = pwm.matrix.shape[0]

            idx = np.searchsorted(null_scores, raw_llr, side="right") - 1
            idx = np.clip(idx, 0, len(tail_p) - 1)
            p_win = float(tail_p[idx])
            n_win = max(1, seq_length - w + 1)
            p_seq = 1.0 - (1.0 - p_win) ** n_win
            p_seq = max(p_seq, 1e-300)
            neglogp_seq = -np.log10(p_seq)

            cons_vec = np.argmax(lom, axis=1).astype(np.int8)
            cons_llr = float(lom[np.arange(w), cons_vec].sum())
            idx2 = np.searchsorted(null_scores, cons_llr, side="right") - 1
            idx2 = np.clip(idx2, 0, len(tail_p) - 1)
            p_win_cons = float(tail_p[idx2])
            p_cons = 1.0 - (1.0 - p_win_cons) ** n_win
            p_cons = max(p_cons, 1e-300)
            neglogp_cons = -np.log10(p_cons)

            if np.isclose(raw_llr, cons_llr, atol=1e-6, rtol=0.0):
                out[tf] = 1.0
            else:
                out[tf] = float(neglogp_seq / neglogp_cons)
        return out


class ZTransform(Transform):
    """
    Compute z-score = (LLR - μ_null) / σ_null for each TF.
    """

    def __init__(
        self,
        pwms: Dict[str, PWM],
        bidirectional: bool,
        background=(0.25, 0.25, 0.25, 0.25),
    ):
        self.pwms = pwms
        self.bidirectional = bidirectional
        self.bg = np.asarray(background, dtype=float)
        self._stats: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for tf, pwm in pwms.items():
            lom = pwm.log_odds()
            null_scores, tail_p = logodds_to_p_lookup(lom, self.bg)
            pmf = np.empty_like(tail_p)
            pmf[:-1] = tail_p[:-1] - tail_p[1:]
            pmf[-1] = tail_p[-1]
            μ = float(np.sum(null_scores * pmf))
            var = float(np.sum(((null_scores - μ) ** 2) * pmf))
            σ = float(np.sqrt(var)) if var > 0 else 1.0
            self._stats[tf] = (μ, σ)

    def __call__(self, raw_llr_map: Dict[str, float], seq_length: int) -> Dict[str, float]:
        return {tf: (raw_llr - μ) / σ for tf, raw_llr in raw_llr_map.items() for (μ, σ) in [self._stats[tf]]}


class PlainP(Transform):
    """
    Compute raw −log10(p_seq) (no normalization by consensus) for each TF.
    """

    def __init__(
        self,
        pwms: Dict[str, PWM],
        bidirectional: bool,
        background=(0.25, 0.25, 0.25, 0.25),
    ):
        self.pwms = pwms
        self.bidirectional = bidirectional
        self.bg = np.asarray(background, dtype=float)

    def __call__(self, raw_llr_map: Dict[str, float], seq_length: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for tf, raw_llr in raw_llr_map.items():
            pwm = self.pwms[tf]
            lom = pwm.log_odds()
            null_scores, tail_p = logodds_to_p_lookup(lom, self.bg)
            w = lom.shape[0]

            idx = np.searchsorted(null_scores, raw_llr, side="right") - 1
            idx = np.clip(idx, 0, len(tail_p) - 1)
            p_win = float(tail_p[idx])
            n_win = max(1, seq_length - w + 1)
            p_seq = min(1.0, p_win * n_win)
            p_seq = max(p_seq, 1e-300)
            out[tf] = float(-np.log10(p_seq))
        return out
