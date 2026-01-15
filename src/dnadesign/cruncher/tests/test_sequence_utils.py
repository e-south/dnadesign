"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_sequence_utils.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.cruncher.core.sequence import hamming_distance


def test_hamming_distance_equal_length() -> None:
    a = np.array([0, 1, 2], dtype=np.int8)
    b = np.array([0, 2, 2], dtype=np.int8)
    assert hamming_distance(a, b) == 1


def test_hamming_distance_variable_length() -> None:
    a = np.array([0, 1, 2], dtype=np.int8)
    b = np.array([1, 1, 2, 3], dtype=np.int8)
    assert hamming_distance(a, b) == 2


def test_hamming_distance_requires_1d() -> None:
    a = np.array([[0, 1], [2, 3]], dtype=np.int8)
    b = np.array([0, 1], dtype=np.int8)
    with pytest.raises(ValueError):
        hamming_distance(a, b)
