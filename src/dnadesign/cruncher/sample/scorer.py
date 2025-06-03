"""
<dnadesign project>
dnadesign/cruncher/sample/scorer.py

FIMO-like Scorer (inspired by Grant et al. 2011, 10.1093/bioinformatics/btr064).

Utility class that turns raw PWM log-odds into a single fitness value. Supports four scales:
    • "llr"        - raw log-odds ratio (max over PWMs)
    • "z"          - z-score vs. PWM-specific null distribution
    • "p"          - Bonferroni-corrected p-value (min over PWMs)
    • "logp_norm"  - −log10(p) / −log10(p_consensus)

Module Author(s): Eric J. South
Dunlop Lab
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.parse.pvalue import logodds_to_p_lookup


@dataclass(slots=True)
class _PWMInfo:
    """
    Immutable container for per‐PWM metadata used in Scorer.
    """

    lom: np.ndarray  # log‐odds matrix (motif_length × 4)
    null_scores: np.ndarray  # ascending grid of possible log‐odds sums
    tail_p: np.ndarray  # P(X ≥ score) for each entry in null_scores
    width: int  # motif length
    mu: Optional[float] = None
    sigma: Optional[float] = None


class Scorer:
    """
    Multi‐PWM scorer that computes raw LLRs, p‐values, z‐scores, or normalized log‐p for a DNA sequence.

    After instantiation, call Scorer.score(seq_array) to obtain a single fitness value
    according to the chosen scale. It also provides utility methods for per‐PWM diagnostics:
      - score_per_pwm(seq) → numpy.ndarray of raw LLRs (one per PWM)
      - neglogp_vector(seq) → numpy.ndarray of −log10(p_seq) (one per PWM)
      - soft_min(seq, beta) → soft‐minimum over the −log10(p_seq) vector (for PT)
    """

    SUPPORTED_SCALES = {"llr", "z", "p", "logp", "logp_norm"}

    def __init__(
        self,
        pwms: Dict[str, PWM],
        *,
        background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        bidirectional: bool = True,
        scale: str = "logp_norm",
    ) -> None:
        """
        Args:
          pwms:         dict of {name: PWM}, each PWM provides .matrix → (w×4) freq/log‐odds.
          background:   length‐4 background frequencies (must sum to 1.0). Default=(0.25,0.25,0.25,0.25).
          bidirectional: if True, scan both forward and reverse‐complement strands.
          scale:        one of "llr", "z", "p", "logp", or "logp_norm".
        """
        self.scale = scale.lower()
        if self.scale not in self.SUPPORTED_SCALES:
            raise ValueError(f"Unsupported scale '{scale}'. Choose from {self.SUPPORTED_SCALES}.")

        self.bg = np.asarray(background, dtype=float)
        if self.bg.shape != (4,) or not np.isclose(self.bg.sum(), 1.0):
            raise ValueError("background must be a length‐4 probability vector summing to 1.0.")

        self.bidirectional = bool(bidirectional)

        # Build a _PWMInfo entry for each PWM: precompute log‐odds and null distributions
        self._cache: Dict[str, _PWMInfo] = {name: self._build_one_pwm_info(pwm) for name, pwm in pwms.items()}

        # If z‐score scale is requested, precompute μ and σ for each PWM’s null distribution
        if self.scale == "z":
            for info in self._cache.values():
                info.mu, info.sigma = self._compute_null_zstats(info)

    @staticmethod
    def _build_one_pwm_info(pwm: PWM) -> _PWMInfo:
        """
        Create a _PWMInfo by computing the log‐odds matrix and exact tail probabilities.
        """
        lom = pwm.log_odds()  # shape = (w, 4)
        null_scores, tail_p = logodds_to_p_lookup(lom, np.full(4, 0.25))
        return _PWMInfo(lom=lom, null_scores=null_scores, tail_p=tail_p, width=lom.shape[0])

    @staticmethod
    def _compute_null_zstats(info: _PWMInfo) -> Tuple[float, float]:
        """
        From tail_p and null_scores, compute μ and σ for the null distribution of LLRs.
        """
        tail = info.tail_p
        pmf = np.empty_like(tail)
        pmf[:-1] = tail[:-1] - tail[1:]
        pmf[-1] = tail[-1]
        mu = float(np.sum(info.null_scores * pmf))
        var = float(np.sum(((info.null_scores - mu) ** 2) * pmf))
        return mu, float(np.sqrt(var)) if var > 0 else 1.0

    def _interp_tail_p(self, score: float, info: _PWMInfo) -> float:
        """
        Given a raw LLR `score`, look up its tail probability P(X ≥ score) from the DP table.
        """
        idx = np.searchsorted(info.null_scores, score, side="right") - 1
        idx = np.clip(idx, 0, info.tail_p.size - 1)
        return float(info.tail_p[idx])

    def _best_llr_and_location(self, seq: np.ndarray, info: _PWMInfo) -> Tuple[float, int, str]:
        """
        Scan a numeric‐encoded sequence `seq` to find:
          1) best raw LLR (sum of log‐odds over a window)
          2) the offset (0‐based index) and strand label ("+" or "-") of that best window.

        Returns:
          (best_llr, best_offset, best_strand).

        Note: We no longer count or penalize “extra hits.” extras are always zero.
        """
        L, w = seq.size, info.width
        if L < w:
            # If sequence shorter than motif length, no hits possible.
            return float("-inf"), 0, "+"

        best_llr = float("-inf")
        best_offset = 0
        best_strand = "+"

        # Prepare strands: forward (“+”) and, if requested, reverse complement (“-”)
        strands: List[tuple[np.ndarray, str]] = [(seq, "+")]
        if self.bidirectional:
            rev = (3 - seq)[::-1]
            strands.append((rev, "-"))

        for strand_array, strand_label in strands:
            for off in range(L - w + 1):
                window = strand_array[off : off + w]
                llr_val = float(info.lom[np.arange(w), window].sum())
                if llr_val > best_llr:
                    best_llr = llr_val
                    best_offset = off
                    best_strand = strand_label

        return best_llr, best_offset, best_strand

    def _scale_llr(self, raw_llr: float, info: _PWMInfo, *, seq_length: int | None) -> float:
        """
        Convert a raw LLR to “logp_norm” = −log10(p_seq) / −log10(p_consensus).
        If scale != "logp_norm", dispatch to _scale_other.
        """
        if self.scale != "logp_norm":
            return self._scale_other(raw_llr, info, seq_length)

        if seq_length is None:
            raise ValueError("seq_length is required for logp_norm scale")

        # Compute p_seq for this raw LLR
        n_win = max(1, seq_length - info.width + 1)
        p_win = self._interp_tail_p(raw_llr, info)
        p_seq = 1.0 - (1.0 - p_win) ** n_win
        p_seq = max(p_seq, 1e-300)

        # Compute consensus LLR (argmax per column) and its p_seq
        cons_llr = float(info.lom.max(axis=1).sum())
        p_win_cons = self._interp_tail_p(cons_llr, info)
        p_cons = 1.0 - (1.0 - p_win_cons) ** n_win
        p_cons = max(p_cons, 1e-300)

        # If raw_llr ≈ cons_llr, clamp to 1.0
        if np.isclose(raw_llr, cons_llr, atol=1e-6, rtol=0.0):
            return 1.0

        return -np.log10(p_seq) / -np.log10(p_cons)

    def _scale_other(self, raw_llr: float, info: _PWMInfo, seq_length: int | None) -> float:
        """
        Handle scales "llr", "z", "p", "logp":
          - "llr"  → raw_llr
          - "z"    → (raw_llr − μ) / σ
          - "p"    → p_seq (Bonferroni correction)
          - "logp" → −log10(p_seq)
        """
        if self.scale == "llr":
            return raw_llr

        if self.scale == "z":
            assert info.mu is not None and info.sigma is not None
            return (raw_llr - info.mu) / info.sigma

        # For "p" and "logp", we need seq_length
        if seq_length is None:
            raise ValueError("seq_length is required for p/logp scales")

        p_win = self._interp_tail_p(raw_llr, info)
        p_seq = min(1.0, p_win * (seq_length - info.width + 1))

        if self.scale == "p":
            return p_seq

        # "logp"
        return -np.log10(p_seq if p_seq > 0 else 1e-300)

    def score_per_pwm(self, seq: np.ndarray) -> np.ndarray:
        """
        Return an array of raw LLRs (no scaling) for each PWM in the cache.
        """
        out = []
        for info in self._cache.values():
            best_llr, _, _ = self._best_llr_and_location(seq, info)
            out.append(best_llr)
        return np.array(out, float)

    def neglogp_vector(self, seq: np.ndarray) -> np.ndarray:
        """
        Compute −log10(p_seq) for each PWM (no extra‐hit penalty).
        Returns a numpy array of length = number of PWMs.
        """
        L = seq.size
        out: List[float] = []
        for info in self._cache.values():
            best_llr, _, _ = self._best_llr_and_location(seq, info)
            p_win = self._interp_tail_p(best_llr, info)
            p_seq = min(1.0, p_win * (L - info.width + 1))
            val = -np.log10(p_seq if p_seq > 0 else 1e-300)
            out.append(val)
        return np.asarray(out, float)

    def score(self, seq: np.ndarray) -> float:
        """
        Compute a single fitness value for the whole sequence `seq`:
          • If "llr": return max(raw LLR over all PWMs).
          • If "z":   return min(z‐score over all PWMs).
          • If "p" or "logp":
              – Build neglogp_vector; if "p", return min(p_seq); if "logp", return min(−log10(p_seq)).
          • If "logp_norm": return min(logp_norm over all PWMs).
        """
        L = seq.size

        if self.scale == "llr":
            return float(np.max(self.score_per_pwm(seq)))

        if self.scale == "z":
            zs = [
                self._scale_other(
                    self._best_llr_and_location(seq, info)[0],
                    info,
                    seq_length=L,
                )
                for info in self._cache.values()
            ]
            return float(np.min(zs))

        if self.scale in {"p", "logp"}:
            vec = self.neglogp_vector(seq)
            if self.scale == "p":
                # Convert neglogp back to p_seq via 10^(−neglogp), then take minimum
                return float(10 ** (-np.max(vec)))
            return float(np.min(vec))

        # "logp_norm"
        norm = [
            self._scale_llr(
                self._best_llr_and_location(seq, info)[0],
                info,
                seq_length=L,
            )
            for info in self._cache.values()
        ]
        return float(np.min(norm))

    def soft_min(self, seq: np.ndarray, beta: float) -> float:
        """
        Compute the soft‐minimum of the −log10(p_seq) vector at inverse temperature beta.
        Useful for Parallel Tempering.
        """
        return -logsumexp(-beta * self.neglogp_vector(seq)) / beta
