"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_scorer_contract.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
import pytest

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer


def test_scorer_helpers_surface_consensus_and_llr() -> None:
    matrix = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.9, 0.05, 0.03, 0.02],
            [0.9, 0.05, 0.03, 0.02],
        ],
        dtype=float,
    )
    pwm = PWM(name="tfA", matrix=matrix)
    scorer = Scorer({"tfA": pwm}, bidirectional=False, scale="llr")

    assert scorer.consensus_sequence("tfA") == "AAA"

    seq = np.zeros(pwm.length, dtype=int)
    fracs = scorer.normalized_llr_components(seq)
    assert len(fracs) == 1
    assert fracs[0] == pytest.approx(1.0)

    best_llr, offset, strand = scorer.best_llr(seq, "tfA")
    assert offset == 0
    assert strand == "+"
    assert best_llr == pytest.approx(scorer.consensus_llr("tfA"))
