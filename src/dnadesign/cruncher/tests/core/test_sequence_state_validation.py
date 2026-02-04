"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_sequence_state_validation.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
import pytest

from dnadesign.cruncher.core.state import SequenceState


def test_sequence_state_rejects_non_1d() -> None:
    with pytest.raises(ValueError):
        SequenceState(np.array([[0, 1], [2, 3]]))


def test_sequence_state_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        SequenceState(np.array([0, 4, 1], dtype=np.int8))


def test_sequence_state_accepts_valid() -> None:
    state = SequenceState(np.array([0, 1, 2, 3], dtype=np.int8))
    assert len(state) == 4
