"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/scoring.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from dnadesign.cruncher.core.pvalue import logodds_to_p_lookup
from dnadesign.cruncher.core.pwm import PWM

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PWMInfo:
    """
    Holds all precomputed data for one PWM:
      • lom               : log-odds matrix (w x 4)
      • null_scores, tail_p: DP table → P(X ≥ LLR) for each possible LLR
      • width             : motif length
      • consensus_llr     : sum of column-max LLRs (for normalization)
      • consensus_neglogp : -log₁₀(p_seq) of that consensus LLR (once seq_length is known)
      • null_mean         : mean of the null distribution of single-window LLRs
      • null_std          : standard deviation of that null distribution
    """

    lom: np.ndarray
    null_scores: np.ndarray
    tail_p: np.ndarray
    width: int

    consensus_llr: float = 0.0
    consensus_neglogp: float = 0.0

    null_mean: float = 0.0
    null_std: float = 1.0


class Scorer:
    """
    Multi‐PWM scorer with exactly four supported scales:
      • "llr"                 → raw max LLR per PWM
      • "z"                   → z-score of raw LLR vs the PWM-specific null distribution
      • "logp"                → -log10(p_seq) per PWM
      • "consensus-neglop-sum"→ normalized (-log10(p_seq) / -log10(p_consensus)) per PWM

    Usage:
        scorer = Scorer(pwms, background=(0.25,0.25,0.25,0.25), bidirectional=True, scale="z")
        per_tf = scorer.compute_all_per_pwm(seq_array, seq_length)
    """

    SUPPORTED_SCALES = {"llr", "z", "logp", "consensus-neglop-sum"}

    def __init__(
        self,
        pwms: Dict[str, PWM],
        *,
        background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        bidirectional: bool = True,
        scale: str = "logp",
    ) -> None:
        self.scale = scale.lower()
        if self.scale not in self.SUPPORTED_SCALES:
            raise ValueError(f"Unsupported scale '{scale}'; choose from {self.SUPPORTED_SCALES}.")

        self.bg = np.asarray(background, dtype=float)
        if self.bg.shape != (4,) or not np.isclose(self.bg.sum(), 1.0):
            raise ValueError("background must be a length-4 probability vector summing to 1.0.")

        self.bidirectional = bool(bidirectional)

        logger.info("Building Scorer (scale=%r, bidirectional=%s)", self.scale, self.bidirectional)

        # Build per‐PWM info (lom, null_scores, tail_p, width, plus compute null mean/std).
        self._cache: Dict[str, _PWMInfo] = {}
        for name, pwm in pwms.items():
            logger.debug("  Precomputing PWM info for %s", name)
            info = self._build_one_pwm_info(pwm, self.bg)

            # Precompute consensus_llr for each PWM (sum of column‐maxes).
            info.consensus_llr = float(np.max(info.lom, axis=1).sum())

            # Compute null distribution's PMF, mean, and variance for single-window LLRs.
            # null_scores[i] is sorted list of unique LLR values.
            # tail_p[i] = P(LLR >= null_scores[i]).
            # Then P(LLR = null_scores[i]) = tail_p[i] - tail_p[i+1], with tail_p[last+1] = 0.
            scores = info.null_scores
            tails = info.tail_p
            # Append a zero at the end for tail_p[last+1]
            tails_extended = np.concatenate([tails, [0.0]])
            pmf = tails_extended[:-1] - tails_extended[1:]
            pmf = np.clip(pmf, 0.0, 1.0)  # numerical safeguards

            mean_null = float((scores * pmf).sum())
            var_null = float(((scores - mean_null) ** 2 * pmf).sum())
            std_null = float(np.sqrt(var_null)) if var_null > 0 else 1.0

            info.null_mean = mean_null
            info.null_std = std_null

            self._cache[name] = info

        logger.info("Finished building cache with %d PWMs", len(self._cache))

    @staticmethod
    def _build_one_pwm_info(pwm: PWM, background: np.ndarray) -> _PWMInfo:
        """
        Given a PWM, compute its log‐odds matrix and null distribution lookup table.
        """
        lom = pwm.log_odds()  # shape (w, 4)
        null_scores, tail_p = logodds_to_p_lookup(lom, background)
        return _PWMInfo(lom=lom, null_scores=null_scores, tail_p=tail_p, width=lom.shape[0])

    def _interp_tail_p(self, raw_llr: float, info: _PWMInfo) -> float:
        """
        Return P(X ≥ raw_llr) from the DP table.
        """
        idx = np.searchsorted(info.null_scores, raw_llr, side="right") - 1
        idx = np.clip(idx, 0, info.tail_p.size - 1)
        return float(info.tail_p[idx])

    def _per_pwm_neglogp(self, raw_llr: float, info: _PWMInfo, seq_length: int) -> float:
        """
        Compute −log10(p_seq) for a single PWM:
          p_win = P(X ≥ raw_llr) on one window,
          n_win = max(1, seq_length − width + 1),
          p_seq = 1 − (1 − p_win)^n_win,
          return −log10(p_seq).
        """
        w = info.width
        p_win = self._interp_tail_p(raw_llr, info)
        n_win = max(1, seq_length - w + 1)
        p_seq = 1.0 - (1.0 - p_win) ** n_win
        p_seq = max(p_seq, 1e-300)
        neglogp = -np.log10(p_seq)
        logger.debug("    PWMCALC: raw_llr=%.3f, width=%d, n_win=%d, neglogp=%.3f", raw_llr, w, n_win, neglogp)
        return neglogp

    def _best_llr_and_location(self, seq: np.ndarray, info: _PWMInfo) -> Tuple[float, int, str]:
        """
        Scan a numeric‐encoded sequence to find the best raw LLR (and its offset & strand).
        """
        L, w = seq.size, info.width
        if L < w:
            return float("-inf"), 0, "+"

        def _scan(arr: np.ndarray, strand_label: str) -> Tuple[float, int, str]:
            windows = np.lib.stride_tricks.sliding_window_view(arr, w)
            # windows shape: (L - w + 1, w)
            llrs = info.lom[np.arange(w)[:, None], windows.T].sum(axis=0)
            best_offset = int(np.argmax(llrs))
            best_llr = float(llrs[best_offset])
            return best_llr, best_offset, strand_label

        best_llr, best_offset, best_strand = _scan(seq, "+")
        if self.bidirectional:
            rev = (3 - seq)[::-1]
            rev_llr, rev_offset, rev_strand = _scan(rev, "-")
            if rev_llr > best_llr:
                best_llr, best_offset, best_strand = rev_llr, rev_offset, rev_strand

        logger.debug("    BEST_LLR: best_llr=%.3f at offset=%d, strand=%s", best_llr, best_offset, best_strand)
        return best_llr, best_offset, best_strand

    def _info(self, tf: str) -> _PWMInfo:
        info = self._cache.get(tf)
        if info is None:
            available = ", ".join(self._cache.keys())
            raise ValueError(f"Unknown TF '{tf}'. Available: {available}")
        return info

    @property
    def tf_names(self) -> list[str]:
        return list(self._cache.keys())

    @property
    def pwm_count(self) -> int:
        return len(self._cache)

    def consensus_llr(self, tf: str) -> float:
        return float(self._info(tf).consensus_llr)

    def pwm_width(self, tf: str) -> int:
        return int(self._info(tf).width)

    def consensus_sequence(self, tf: str) -> str:
        info = self._info(tf)
        idx_vec = np.argmax(info.lom, axis=1)
        return "".join("ACGT"[i] for i in idx_vec)

    def best_llr(self, seq: np.ndarray, tf: str) -> Tuple[float, int, str]:
        info = self._info(tf)
        return self._best_llr_and_location(seq, info)

    def normalized_llr_components(self, seq: np.ndarray) -> list[float]:
        components: list[float] = []
        for info in self._cache.values():
            raw_llr, *_ = self._best_llr_and_location(seq, info)
            num, denom = raw_llr - info.null_mean, info.consensus_llr - info.null_mean
            frac = 0.0 if denom <= 0 else max(0.0, num / denom)
            components.append(frac)
        return components

    def compute_all_per_pwm(self, seq: np.ndarray, seq_length: int) -> Dict[str, float]:
        """
        For each PWM (TF), compute exactly one “scaled” value, depending on self.scale:

          • "llr" : raw LLR
          • "z"   : z-score → (raw_llr − null_mean) / null_std
          • "logp": −log10(p_seq)
          • "consensus-neglop-sum":
              (-log10(p_seq) / precomputed_neglogp(consensus_llr))

        On the very first call for a given PWM, we fill in info.consensus_neglogp using
        info.consensus_llr and the same Bonferroni formula.
        """
        out: Dict[str, float] = {}
        logger.debug("compute_all_per_pwm: seq_length=%d, scale=%s", seq_length, self.scale)

        for tf, info in self._cache.items():
            raw_llr, offset, strand = self._best_llr_and_location(seq, info)

            if self.scale == "llr":
                out[tf] = float(raw_llr)
                continue

            # If z-score requested:
            if self.scale == "z":
                # z = (raw_llr - mean_null) / std_null
                z_val = (raw_llr - info.null_mean) / info.null_std
                out[tf] = float(z_val)
                continue

            # For any neglogp‐based result (logp or consensus‐neglop-sum):
            neglogp_seq = self._per_pwm_neglogp(raw_llr, info, seq_length)
            if self.scale == "logp":
                out[tf] = float(neglogp_seq)
                continue

            # Now scale must be "consensus-neglop-sum"
            if info.consensus_neglogp <= 0.0:
                cons_llr = info.consensus_llr
                neglogp_cons = self._per_pwm_neglogp(cons_llr, info, seq_length)
                info.consensus_neglogp = neglogp_cons
                logger.debug("    Set consensus_neglogp for %s = %.3f", tf, neglogp_cons)

            if info.consensus_neglogp > 0.0:
                normalized = float(neglogp_seq / info.consensus_neglogp)
                out[tf] = normalized
            else:
                out[tf] = 0.0

        logger.debug("  Per-TF scaled map: %s", out)
        return out
