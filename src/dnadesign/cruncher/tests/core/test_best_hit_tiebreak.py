"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_best_hit_tiebreak.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.core.scoring import _best_hit_from_llrs


def test_best_hit_prefers_smaller_start_across_strands() -> None:
    llrs_fwd = np.array([1.0, 2.0, 5.0, 2.0, 1.0], dtype=float)
    llrs_rev = np.array([1.0, 2.0, 2.0, 5.0, 1.0], dtype=float)
    best_llr, offset, strand, _rule = _best_hit_from_llrs(
        llrs_fwd,
        llrs_rev,
        seq_len=6,
        width=2,
        prefer_strand="+",
    )
    assert best_llr == 5.0
    assert strand == "-"
    assert offset == 3


def test_best_hit_prefers_plus_when_start_ties() -> None:
    llrs_fwd = np.array([5.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    llrs_rev = np.array([1.0, 1.0, 1.0, 1.0, 5.0], dtype=float)
    best_llr, offset, strand, _rule = _best_hit_from_llrs(
        llrs_fwd,
        llrs_rev,
        seq_len=6,
        width=2,
        prefer_strand="+",
    )
    assert best_llr == 5.0
    assert strand == "+"
    assert offset == 0
