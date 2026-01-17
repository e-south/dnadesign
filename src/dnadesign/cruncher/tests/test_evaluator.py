"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_evaluator.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer


def _pwm(name: str) -> PWM:
    matrix = np.full((4, 4), 0.25, dtype=float)
    return PWM(name=name, matrix=matrix)


def test_sequence_evaluator_accepts_shared_scorer() -> None:
    pwms = {"tf1": _pwm("tf1"), "tf2": _pwm("tf2")}
    scorer = Scorer(pwms, scale="llr", bidirectional=True)
    evaluator = SequenceEvaluator(
        pwms=pwms,
        scale="llr",
        scorer=scorer,
        bidirectional=True,
        background=(0.25, 0.25, 0.25, 0.25),
    )
    assert evaluator.scorer is scorer


def test_sequence_evaluator_rejects_scale_mismatch() -> None:
    pwms = {"tf1": _pwm("tf1")}
    scorer = Scorer(pwms, scale="llr", bidirectional=True)
    with pytest.raises(ValueError, match="scale"):
        SequenceEvaluator(
            pwms=pwms,
            scale="logp",
            scorer=scorer,
            bidirectional=True,
            background=(0.25, 0.25, 0.25, 0.25),
        )


def test_sequence_evaluator_combiner_min_vs_sum() -> None:
    pwms = {"tf1": _pwm("tf1"), "tf2": _pwm("tf2")}
    per_tf = {"tf1": 1.0, "tf2": 2.0}

    evaluator_min = SequenceEvaluator(pwms=pwms, scale="llr", combiner=None)
    assert evaluator_min.combined_from_scores(per_tf) == 1.0

    evaluator_sum = SequenceEvaluator(pwms=pwms, scale="llr", combiner=lambda vs: sum(vs))
    assert evaluator_sum.combined_from_scores(per_tf) == 3.0


def test_sequence_evaluator_consensus_neglop_defaults_to_sum() -> None:
    pwms = {"tf1": _pwm("tf1"), "tf2": _pwm("tf2")}
    per_tf = {"tf1": 1.5, "tf2": -0.5}
    evaluator = SequenceEvaluator(pwms=pwms, scale="consensus-neglop-sum", combiner=None)
    assert evaluator.combined_from_scores(per_tf) == 1.0
