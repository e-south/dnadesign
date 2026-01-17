"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_move_schedule.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.core.optimizers.policies import MoveSchedule, move_probs_array


def test_move_probs_normalizes_and_clips() -> None:
    probs = move_probs_array({"S": -1.0, "B": 2.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0})
    assert np.isclose(probs.sum(), 1.0)
    assert probs.min() >= 0
    assert probs[1] == 1.0  # B move should dominate after clipping


def test_move_schedule_normalizes_start() -> None:
    start = move_probs_array({"S": 2.0, "B": 1.0, "M": 0.0, "L": 0.0, "W": 0.0, "I": 0.0})
    schedule = MoveSchedule(start=start, end=None)
    probs = schedule.probs(0.5)
    assert np.isclose(probs.sum(), 1.0)
