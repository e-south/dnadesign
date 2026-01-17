"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_optimizer_helpers.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.core.optimizers.helpers import slide_window, swap_block


def test_slide_window_right() -> None:
    seq = np.array([0, 1, 2, 3, 4, 5], dtype=np.int8)
    slide_window(seq, start=1, length=2, shift=2)
    assert seq.tolist() == [0, 3, 4, 1, 2, 5]


def test_slide_window_left() -> None:
    seq = np.array([0, 1, 2, 3, 4, 5], dtype=np.int8)
    slide_window(seq, start=3, length=2, shift=-2)
    assert seq.tolist() == [0, 3, 4, 1, 2, 5]


def test_swap_block() -> None:
    seq = np.array([0, 1, 2, 3, 4, 5], dtype=np.int8)
    swap_block(seq, a=1, b=4, length=1)
    assert seq.tolist() == [0, 4, 2, 3, 1, 5]
