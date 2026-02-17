"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_expected_improvement_edge_cases.py

Edge-case regression tests for expected-improvement acquisition behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.opal.src.selection.expected_improvement import ei


def test_expected_improvement_handles_mixed_zero_sigma_no_nan() -> None:
    out = ei(
        ids=np.array(["a", "b"], dtype=str),
        scores=np.array([0.4, 0.6], dtype=float),
        scalar_uncertainty=np.array([0.1, 0.0], dtype=float),
        top_k=1,
        objective="maximize",
        tie_handling="competition_rank",
        alpha=1.0,
        beta=1.0,
    )
    acq = np.asarray(out["score"], dtype=float).reshape(-1)
    assert np.all(np.isfinite(acq))
    assert acq.shape == (2,)
    assert abs(float(acq[1])) <= 1e-12


def test_expected_improvement_rejects_all_zero_uncertainty() -> None:
    with pytest.raises(ValueError, match="all zeros"):
        ei(
            ids=np.array(["a", "b"], dtype=str),
            scores=np.array([0.4, 0.6], dtype=float),
            scalar_uncertainty=np.array([0.0, 0.0], dtype=float),
            top_k=1,
            objective="maximize",
            tie_handling="competition_rank",
            alpha=1.0,
            beta=1.0,
        )


def test_expected_improvement_allows_finite_negative_weighted_acquisition() -> None:
    out = ei(
        ids=np.array(["a", "b", "c"], dtype=str),
        scores=np.array([0.2, 0.3, 0.1], dtype=float),
        scalar_uncertainty=np.array([0.05, 0.10, 0.05], dtype=float),
        top_k=1,
        objective="maximize",
        tie_handling="competition_rank",
        alpha=2.0,
        beta=0.1,
    )
    acq = np.asarray(out["score"], dtype=float).reshape(-1)
    assert np.all(np.isfinite(acq))
    assert np.any(acq < 0.0)
