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


def test_sequence_state_accepts_particle_id() -> None:
    state = SequenceState(np.array([0, 1, 2, 3], dtype=np.int8), particle_id=2)
    assert state.particle_id == 2


def test_sequence_state_rejects_negative_particle_id() -> None:
    with pytest.raises(ValueError, match="particle_id"):
        SequenceState(np.array([0, 1, 2, 3], dtype=np.int8), particle_id=-1)


def test_sequence_state_rejects_non_integer_particle_id() -> None:
    with pytest.raises(ValueError, match="particle_id"):
        SequenceState(np.array([0, 1, 2, 3], dtype=np.int8), particle_id=1.5)  # type: ignore[arg-type]
