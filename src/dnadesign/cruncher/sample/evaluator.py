"""
<dnadesign project>
dnadesign/cruncher/sample/evaluator.py

Wrap a Scorer so that MCMC optimizers can call `evaluator(sequence_state) → float`.
Gathers each PWM's output from Scorer and then combines them into one number via some reducer.

SequenceEvaluator builds raw LLR maps for every TF, converts to the chosen scale
(llr, logp, or consensus_neglop_sum), and then reduces to a single fitness value.

Module Author(s): Eric J. South
Dunlop Lab
"""

import logging
from typing import Callable, Dict, Optional

from dnadesign.cruncher.parse.model import PWM
from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.state import SequenceState

logger = logging.getLogger(__name__)


class SequenceEvaluator:
    """
    Wraps a single Scorer instance. Based on the chosen scale, it:
      1) Returns a per‐TF dict of scaled values, via Scorer.compute_all_per_pwm().
      2) Combines them into a single float for MCMC acceptance.
    """

    def __init__(
        self,
        pwms: Dict[str, PWM],
        *,
        scale: str,
        combiner: Optional[Callable[[list[float]], float]] = None,
    ) -> None:
        """
        Args:
          pwms:       dict of {tf_name: PWM}, passed into Scorer.
          scale:      one of {"llr","logp","consensus-neglop-sum"}.
          combiner:   how to reduce the list of per‐TF values to a single float.
                      If None and scale=="consensus-neglop-sum", defaults to sum().
                      If None and scale in {"llr","logp"}, defaults to min().
        """
        self._scale = scale.lower()
        logger.info("Instantiating SequenceEvaluator (scale=%r)", self._scale)
        self._scorer = Scorer(pwms, scale=self._scale)

        if self._scale == "consensus-neglop-sum":
            self._combiner = combiner if combiner is not None else (lambda vs: sum(vs))
        else:
            self._combiner = combiner if combiner is not None else min

        logger.debug("  Scorer attached with scale=%r", self._scorer.scale)
        logger.debug("  Combiner function = %r", self._combiner)

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
        The `beta` argument is ignored (no soft‐min anymore).
        """
        per_tf_vals = list(self(state).values())
        combined_val = float(self._combiner(per_tf_vals))
        logger.debug("Evaluator combined: per-TF values = %s → combined = %.6f", per_tf_vals, combined_val)
        return combined_val
