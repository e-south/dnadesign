"""
<dnadesign project>
dnadesign/cruncher/sample/evaluator.py

Wrap a Scorer so that MCMC optimizers can call `evaluator(sequence_state) → float`.

SequenceEvaluator builds raw LLR maps for every TF, converts to the chosen scale
(llr, z, p, logp, or logp_norm), and then reduces to a single fitness value.

Module Author(s): Eric J. South
Dunlop Lab
"""

from typing import Dict

from dnadesign.cruncher.sample.scorer import Scorer
from dnadesign.cruncher.sample.state import SequenceState


class SequenceEvaluator:
    """
    Adapter that presents a uniform interface for optimizers: evaluator(state) → fitness.

    Internally, it uses Scorer to:
      1. Gather raw LLRs per TF.
      2. Convert each raw LLR into the chosen scale (llr, z, p, logp, or logp_norm).
      3. Reduce all per‐TF scores to a single float via `reducer` (default: min).
    """

    def __init__(self, scorer: Scorer, *, reducer=min):
        """
        Args:
          scorer:  an instance of Scorer (configured with PWMs, scale, etc.).
          reducer: a function to collapse multiple TF‐scores into one (e.g. built‐in min).
        """
        self._scorer = scorer
        self._reducer = reducer

    def __call__(self, state: SequenceState) -> float:
        """
        Evaluate the fitness of `state.seq` (a SequenceState wrapping a numpy array).

        Steps:
          1. For each TF in scaler._cache:
               (raw_llr, _, _) = _scorer._best_llr_and_location(state.seq, info)
               raw_map[tf] = raw_llr

          2. For each TF, convert `raw_llr` to the chosen scale:
             - If "llr":        scaled[tf] = raw_llr
             - If "z":          scaled[tf] = _scorer._scale_other(raw_llr, info, seq_length=L)
             - If "p"/"logp":
                   nonlog = _scorer._scale_other(raw_llr, info, seq_length=L)
                   if "p": scaled[tf] = 10**(-nonlog)
                   else:   scaled[tf] = nonlog
             - If "logp_norm":  scaled[tf] = _scorer._scale_llr(raw_llr, info, seq_length=L)

          3. Return self._reducer(scaled.values()) as a float.
        """
        raw_map: Dict[str, float] = {}
        for tf, info in self._scorer._cache.items():
            raw_llr, _, _ = self._scorer._best_llr_and_location(state.seq, info)
            raw_map[tf] = raw_llr

        # Convert raw_map → scaled_map
        scaled: Dict[str, float] = {}
        L = len(state)
        for tf, raw_val in raw_map.items():
            info = self._scorer._cache[tf]
            if self._scorer.scale == "llr":
                scaled[tf] = raw_val
            elif self._scorer.scale in ("p", "logp"):
                nonlog = self._scorer._scale_other(raw_val, info, seq_length=L)
                if self._scorer.scale == "p":
                    scaled[tf] = 10 ** (-nonlog)
                else:
                    scaled[tf] = nonlog
            elif self._scorer.scale == "z":
                scaled[tf] = self._scorer._scale_other(raw_val, info, seq_length=L)
            else:  # "logp_norm"
                scaled[tf] = self._scorer._scale_llr(raw_val, info, seq_length=L)

        return float(self._reducer(scaled.values()))
