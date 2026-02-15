"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_scoring_logp.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.core.scoring import Scorer


@pytest.mark.parametrize(
    ("p_win", "n_offsets"),
    [
        (1.0e-6, 1),
        (1.0e-6, 10),
        (1.0e-6, 1000),
        (1.0e-3, 10),
        (1.0e-1, 10),
    ],
)
def test_logp_bidirectional_counts_both_strands(p_win: float, n_offsets: int) -> None:
    p_fwd = Scorer._p_seq_from_p_win(p_win, n_offsets, bidirectional=False)
    p_bi = Scorer._p_seq_from_p_win(p_win, n_offsets, bidirectional=True)
    expected_fwd = 1.0 - (1.0 - p_win) ** n_offsets
    expected_bi = 1.0 - (1.0 - p_win) ** (2 * n_offsets)
    assert p_fwd == pytest.approx(expected_fwd)
    assert p_bi == pytest.approx(expected_bi)
    assert p_bi >= p_fwd
