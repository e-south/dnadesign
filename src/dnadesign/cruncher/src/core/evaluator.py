"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/core/evaluator.py

Core runtime primitives for evaluator Cruncher core.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import logging
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from dnadesign.cruncher.core.objectives.capabilities import BEST_HIT_CAPABILITIES
from dnadesign.cruncher.core.objectives.engine import ObjectiveEngine
from dnadesign.cruncher.core.objectives.models import (
    BestHitAggregationSpec,
    BestHitSelectorSpec,
    ObjectiveSpec,
)
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
        objective_specs: Sequence[ObjectiveSpec] | None = None,
        objective_engine: ObjectiveEngine | None = None,
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
        if objective_engine is not None:
            self._objective_engine = objective_engine
        else:
            specs = tuple(objective_specs or ())
            if not specs:
                specs = tuple(
                    ObjectiveSpec(
                        objective_id=tf_name,
                        tf=tf_name,
                        pwm_source_id=tf_name,
                        score_scale=self._scale,
                        bidirectional=bool(bidirectional),
                        selector=BestHitSelectorSpec(),
                        aggregator=BestHitAggregationSpec(),
                        capabilities=BEST_HIT_CAPABILITIES,
                        metadata={"objective_kind": "best_hit"},
                    )
                    for tf_name in sorted(pwms)
                )
            self._objective_engine = ObjectiveEngine(scorer=self._scorer, objectives=specs)

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
        return self._objective_engine.evaluate(seq_arr, seq_length=L).scalars

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

    def combined_from_raw_llr(
        self,
        raw_llr_by_tf: Dict[str, float],
        beta: Optional[float] = None,
        *,
        length: int | None = None,
    ) -> float:
        """
        Combine raw per-TF LLR values without allocating an intermediate per-TF scaled map.
        """
        if length is None:
            raise ValueError("length must be provided when combining raw LLR values")
        per_tf_vals = self._scorer.scaled_values_from_raw_llr(raw_llr_by_tf, length)
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
        return [objective.objective_id for objective in self._objective_engine.objectives]

    @property
    def scorer(self) -> Scorer:
        return self._scorer

    def pwm_width(self, tf: str) -> int:
        objective = next((item for item in self._objective_engine.objectives if item.objective_id == tf), None)
        if objective is None:
            raise ValueError(f"Unknown objective '{tf}'.")
        return self._scorer.pwm_width(objective.tf)

    def best_hits(self, state: SequenceState) -> Dict[str, tuple[float, int, str]]:
        results = self._objective_engine.evaluate(state.seq, seq_length=len(state)).results
        out: dict[str, tuple[float, int, str]] = {}
        for objective_id, result in results.items():
            rep = result.representative_hit
            if rep is None:
                out[objective_id] = (float("-inf"), 0, "+")
            else:
                out[objective_id] = (float(rep.raw_score), int(rep.start), str(rep.strand))
        return out

    def best_hit(self, state: SequenceState, tf: str) -> tuple[float, int, str]:
        result = self._objective_engine.evaluate_objective(tf, state.seq, seq_length=len(state))
        rep = result.representative_hit
        if rep is None:
            return float("-inf"), 0, "+"
        return float(rep.raw_score), int(rep.start), str(rep.strand)

    def normalized_llr_map(self, state: SequenceState) -> Dict[str, float]:
        return self._scorer.normalized_llr_map(state.seq)

    @property
    def objective_engine(self) -> ObjectiveEngine:
        return self._objective_engine

    def _apply_length_penalty(self, score: float, length: int | None) -> float:
        if self._length_penalty_lambda <= 0:
            return float(score)
        if length is None:
            raise ValueError("length must be provided when length_penalty_lambda > 0")
        if self._length_penalty_ref is None:
            raise ValueError("length_penalty_ref is required when length_penalty_lambda > 0")
        penalty = self._length_penalty_lambda * (float(length) - float(self._length_penalty_ref))
        return float(score) - penalty
