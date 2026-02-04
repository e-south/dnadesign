"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_polish.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.app.sample_workflow import _polish_sequence
from dnadesign.cruncher.core.evaluator import SequenceEvaluator
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer
from dnadesign.cruncher.core.state import SequenceState


def test_polish_improves_or_preserves_score() -> None:
    matrix = np.array([[0.9, 0.05, 0.03, 0.02]] * 3, dtype=float)
    pwm = PWM(name="tfA", matrix=matrix)
    pwms = {"tfA": pwm}
    scorer = Scorer(pwms, bidirectional=False, scale="llr")
    evaluator = SequenceEvaluator(
        pwms=pwms,
        scale="llr",
        scorer=scorer,
        bidirectional=False,
        background=(0.25, 0.25, 0.25, 0.25),
        pseudocounts=0.0,
        log_odds_clip=None,
    )
    seq = np.array([1, 1, 1], dtype=np.int8)  # CCC, suboptimal for A-rich PWM
    before = evaluator.combined(SequenceState(seq), beta=None)
    polished = _polish_sequence(
        seq,
        evaluator=evaluator,
        beta_softmin_final=None,
        max_rounds=2,
        improvement_tol=0.0,
        max_evals=200,
    )
    after = evaluator.combined(SequenceState(polished), beta=None)
    assert after >= before
