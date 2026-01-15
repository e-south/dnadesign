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
