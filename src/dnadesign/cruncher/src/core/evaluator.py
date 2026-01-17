"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/evaluator.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import logging
from typing import Callable, Dict, Optional

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.state import SequenceState

logger = logging.getLogger(__name__)


class SequenceEvaluator:
    """
    Wraps a single Scorer instance. Based on the chosen scale, it:
      1) Returns a per-TF dict of scaled values, via Scorer.compute_all_per_pwm().
      2) Combines them into a single float for MCMC acceptance.
    """

    def __init__(
        self,
        pwms: Dict[str, PWM],
        *,
        scale: str,
        combiner: Optional[Callable[[list[float]], float]] = None,
        scorer: Scorer | None = None,
        bidirectional: bool = True,
        background: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        pseudocounts: float = 0.0,
        log_odds_clip: float | None = None,
        length_penalty_lambda: float = 0.0,
        length_penalty_ref: int | None = None,
    ) -> None:
        """
        Args:
          pwms:       dict of {tf_name: PWM}, passed into Scorer.
          scale:      one of {"llr","z","logp","normalized-llr","consensus-neglop-sum"}.
          combiner:   how to reduce the list of per‐TF values to a single float.
                      If None and scale=="consensus-neglop-sum", defaults to sum().
                      If None and scale in {"llr","z","logp","normalized-llr"}, defaults to min().
          scorer:     optional pre-built Scorer (must match scale/bidirectional/background).
          bidirectional: scan both strands if True.
          background: zero-order background frequencies (length-4).
          length_penalty_lambda: subtract lambda * (L - length_penalty_ref) from combined scores.
          length_penalty_ref: reference length used by the penalty when lambda > 0.
        """
        self._scale = scale.lower()
        logger.debug("Instantiating SequenceEvaluator (scale=%r)", self._scale)
        if scorer is None:
            self._scorer = Scorer(
                pwms,
                scale=self._scale,
                bidirectional=bidirectional,
                background=background,
                pseudocounts=pseudocounts,
                log_odds_clip=log_odds_clip,
            )
        else:
            if scorer.scale != self._scale:
                msg = f"SequenceEvaluator scale '{self._scale}' does not match scorer scale '{scorer.scale}'."
                raise ValueError(msg)
            if bool(scorer.bidirectional) != bool(bidirectional):
                raise ValueError("SequenceEvaluator bidirectional flag does not match scorer.")
            if not np.allclose(scorer.bg, np.asarray(background, dtype=float)):
                raise ValueError("SequenceEvaluator background does not match scorer background.")
            if float(scorer.pseudocounts) != float(pseudocounts):
                raise ValueError("SequenceEvaluator pseudocounts do not match scorer.")
            scorer_clip = scorer.log_odds_clip
            if (scorer_clip is None) != (log_odds_clip is None) or (
                scorer_clip is not None and log_odds_clip is not None and float(scorer_clip) != float(log_odds_clip)
            ):
                raise ValueError("SequenceEvaluator log_odds_clip does not match scorer.")
            if set(pwms.keys()) != set(scorer.tf_names):
                raise ValueError("SequenceEvaluator PWMs do not match scorer TF set.")
            self._scorer = scorer

        if self._scale == "consensus-neglop-sum":
            if combiner is None:
                self._combiner = lambda vs: sum(vs)
                self._use_softmin = False
            else:
                self._combiner = combiner
                self._use_softmin = combiner is min
        else:
            self._combiner = combiner if combiner is not None else min
            self._use_softmin = combiner is None or combiner is min

        self._length_penalty_lambda = float(length_penalty_lambda)
        if self._length_penalty_lambda < 0:
            raise ValueError("length_penalty_lambda must be >= 0")
        if self._length_penalty_lambda > 0 and length_penalty_ref is None:
            raise ValueError("length_penalty_ref must be set when length_penalty_lambda > 0")
        self._length_penalty_ref = int(length_penalty_ref) if length_penalty_ref is not None else None

        logger.debug("  Scorer attached with scale=%r", self._scorer.scale)
        logger.debug("  Combiner function = %r", self._combiner)
        if self._length_penalty_lambda > 0:
            logger.debug(
                "  Length penalty enabled: lambda=%.3f ref=%s",
                self._length_penalty_lambda,
                self._length_penalty_ref,
            )

    def __call__(self, state: SequenceState) -> Dict[str, float]:
        """
        Return a dict mapping each TF → “scaled” score (based on self._scale).
        """
        seq_arr = state.seq
        L = len(seq_arr)
        logger.debug("Evaluator __call__: computing per-TF for sequence length %d", L)
        per_tf = self._scorer.compute_all_per_pwm(seq_arr, L)
        return per_tf

    def combined(self, state: SequenceState, beta: Optional[float] = None) -> float:
        """
        Reduce the per‐TF values (returned by __call__) into a single float via self._combiner.
        If beta is provided and the default combiner is min(), we apply a soft-min
        (log-sum-exp) temperature to smooth the minimum across TFs.
        """
        per_tf_vals = list(self(state).values())
        if beta is not None and self._use_softmin:
            vals = np.asarray(per_tf_vals, dtype=float)
            scaled = -beta * vals
            max_scaled = float(np.max(scaled))
            logsum = max_scaled + float(np.log(np.exp(scaled - max_scaled).sum()))
            combined_val = -logsum / beta
        else:
            combined_val = float(self._combiner(per_tf_vals))
        combined_val = self._apply_length_penalty(combined_val, len(state))
        logger.debug(
            "Evaluator combined: per-TF values = %s → combined = %.6f",
            per_tf_vals,
            combined_val,
        )
        return combined_val

    def combined_from_scores(
        self,
        per_tf_scores: Dict[str, float],
        beta: Optional[float] = None,
        *,
        length: int | None = None,
    ) -> float:
        """
        Combine precomputed per-TF scores without rescanning the sequence.
        The optional length is required when length penalties are enabled.
        """
        per_tf_vals = list(per_tf_scores.values())
        if beta is not None and self._use_softmin:
            vals = np.asarray(per_tf_vals, dtype=float)
            scaled = -beta * vals
            max_scaled = float(np.max(scaled))
            logsum = max_scaled + float(np.log(np.exp(scaled - max_scaled).sum()))
            combined_val = -logsum / beta
        else:
            combined_val = float(self._combiner(per_tf_vals))
        combined_val = self._apply_length_penalty(combined_val, length)
        return combined_val

    def evaluate(
        self,
        state: SequenceState,
        beta: Optional[float] = None,
        *,
        length: int | None = None,
    ) -> tuple[Dict[str, float], float]:
        """
        Compute per-TF scores and the combined score in a single scan.

        Returns:
          (per_tf_scores, combined_score)
        """
        per_tf = self(state)
        seq_len = len(state) if length is None else length
        combined_val = self.combined_from_scores(per_tf, beta=beta, length=seq_len)
        return per_tf, combined_val

    @property
    def tf_names(self) -> list[str]:
        return self._scorer.tf_names

    @property
    def scorer(self) -> Scorer:
        return self._scorer

    def pwm_width(self, tf: str) -> int:
        return self._scorer.pwm_width(tf)

    def best_hits(self, state: SequenceState) -> Dict[str, tuple[float, int, str]]:
        seq_arr = state.seq
        return {tf: self._scorer.best_llr(seq_arr, tf) for tf in self._scorer.tf_names}

    def best_hit(self, state: SequenceState, tf: str) -> tuple[float, int, str]:
        return self._scorer.best_llr(state.seq, tf)

    def normalized_llr_map(self, state: SequenceState) -> Dict[str, float]:
        return self._scorer.normalized_llr_map(state.seq)

    def _apply_length_penalty(self, score: float, length: int | None) -> float:
        if self._length_penalty_lambda <= 0:
            return float(score)
        if length is None:
            raise ValueError("length must be provided when length_penalty_lambda > 0")
        if self._length_penalty_ref is None:
            raise ValueError("length_penalty_ref is required when length_penalty_lambda > 0")
        penalty = self._length_penalty_lambda * (float(length) - float(self._length_penalty_ref))
        return float(score) - penalty
