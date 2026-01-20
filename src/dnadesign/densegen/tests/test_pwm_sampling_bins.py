from __future__ import annotations

import numpy as np

from dnadesign.densegen.src.adapters.sources.pwm_sampling import (
    FimoCandidate,
    _assign_pvalue_bin,
    _stratified_sample,
)


def test_assign_pvalue_bin_edges() -> None:
    edges = [1e-4, 1e-2, 1.0]
    assert _assign_pvalue_bin(1e-4, edges) == (0, 0.0, 1e-4)
    assert _assign_pvalue_bin(5e-4, edges) == (1, 1e-4, 1e-2)
    assert _assign_pvalue_bin(0.5, edges) == (2, 1e-2, 1.0)


def test_stratified_sample_balances_bins() -> None:
    rng = np.random.default_rng(0)
    candidates = [
        FimoCandidate(
            seq="AAAA",
            pvalue=1e-6,
            score=10.0,
            bin_id=0,
            bin_low=0.0,
            bin_high=1e-4,
            start=0,
            stop=3,
            strand="+",
            matched_sequence=None,
        ),
        FimoCandidate(
            seq="AAAT",
            pvalue=5e-6,
            score=9.0,
            bin_id=0,
            bin_low=0.0,
            bin_high=1e-4,
            start=0,
            stop=3,
            strand="+",
            matched_sequence=None,
        ),
        FimoCandidate(
            seq="TTTT",
            pvalue=5e-3,
            score=6.0,
            bin_id=1,
            bin_low=1e-4,
            bin_high=1e-2,
            start=0,
            stop=3,
            strand="+",
            matched_sequence=None,
        ),
        FimoCandidate(
            seq="TTTA",
            pvalue=8e-3,
            score=5.0,
            bin_id=1,
            bin_low=1e-4,
            bin_high=1e-2,
            start=0,
            stop=3,
            strand="+",
            matched_sequence=None,
        ),
    ]

    picked = _stratified_sample(candidates, n_sites=3, rng=rng, n_bins=2)
    assert len(picked) == 3
    assert {int(c.bin_id) for c in picked} == {0, 1}
