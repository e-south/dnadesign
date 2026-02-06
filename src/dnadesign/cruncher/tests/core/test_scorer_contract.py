"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_scorer_contract.py

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


def test_consensus_neglogp_cached_by_length() -> None:
    matrix = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.9, 0.05, 0.03, 0.02],
        ],
        dtype=float,
    )
    pwm = PWM(name="tfA", matrix=matrix)
    scorer = Scorer({"tfA": pwm}, bidirectional=False, scale="consensus-neglop-sum")
    seq1 = np.zeros(5, dtype=int)
    seq2 = np.zeros(8, dtype=int)

    scorer.compute_all_per_pwm(seq1, seq1.size)
    scorer.compute_all_per_pwm(seq2, seq2.size)

    info = scorer._cache["tfA"]
    assert seq1.size in info.consensus_neglogp_by_len
    assert seq2.size in info.consensus_neglogp_by_len
    assert info.consensus_neglogp_by_len[seq1.size] != info.consensus_neglogp_by_len[seq2.size]


def test_best_llr_tie_breaks_leftmost_offset() -> None:
    matrix = np.full((2, 4), 0.25, dtype=float)
    pwm = PWM(name="tfA", matrix=matrix)
    scorer = Scorer({"tfA": pwm}, bidirectional=False, scale="llr")
    seq = np.array([0, 1, 2, 3, 0], dtype=np.int8)
    best_llr, offset, strand = scorer.best_llr(seq, "tfA")
    assert best_llr == pytest.approx(0.0, abs=1.0e-6)
    assert offset == 0
    assert strand == "+"


def test_best_llr_tie_prefers_smaller_start_then_plus() -> None:
    matrix = np.full((2, 4), 0.25, dtype=float)
    pwm = PWM(name="tfA", matrix=matrix)
    scorer = Scorer({"tfA": pwm}, bidirectional=True, scale="llr")
    seq = np.array([0, 1, 2, 3, 0, 1], dtype=np.int8)
    best_llr, offset, strand = scorer.best_llr(seq, "tfA")
    assert best_llr == pytest.approx(0.0, abs=1.0e-6)
    assert offset == 0
    assert strand == "+"


def test_best_llr_handles_sequence_shorter_than_pwm_width() -> None:
    matrix = np.array(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.9, 0.05, 0.03, 0.02],
            [0.9, 0.05, 0.03, 0.02],
        ],
        dtype=float,
    )
    pwm = PWM(name="tfA", matrix=matrix)
    scorer = Scorer({"tfA": pwm}, bidirectional=True, scale="llr")
    seq = np.array([0, 1], dtype=np.int8)
    best_llr, offset, strand = scorer.best_llr(seq, "tfA")
    assert best_llr == float("-inf")
    assert offset == 0
    assert strand == "+"
