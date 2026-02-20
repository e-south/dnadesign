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
from scipy.stats import norm

from dnadesign.opal.src.selection.expected_improvement import ei


def test_expected_improvement_rejects_any_zero_sigma() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        ei(
            ids=np.array(["a", "b"], dtype=str),
            scores=np.array([0.4, 0.6], dtype=float),
            scalar_uncertainty=np.array([0.1, 0.0], dtype=float),
            top_k=1,
            objective="maximize",
            tie_handling="competition_rank",
            alpha=1.0,
            beta=1.0,
        )


def test_expected_improvement_rejects_all_zero_uncertainty() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
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


def test_expected_improvement_scores_are_normalized_to_unit_interval() -> None:
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
    assert np.all((acq >= 0.0) & (acq <= 1.0))
    assert np.any(np.isclose(acq, 0.0, atol=1e-12))
    assert np.any(np.isclose(acq, 1.0, atol=1e-12))


def test_expected_improvement_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        ei(
            ids=np.array(["a", "b"], dtype=str),
            scores=np.array([0.4, 0.6], dtype=float),
            scalar_uncertainty=np.array([-0.1, 0.2], dtype=float),
            top_k=1,
            objective="maximize",
            tie_handling="competition_rank",
            alpha=1.0,
            beta=1.0,
        )


@pytest.mark.parametrize(
    "bad_unc",
    [
        np.array([0.1, np.nan], dtype=float),
        np.array([0.1, np.inf], dtype=float),
    ],
)
def test_expected_improvement_rejects_non_finite_sigma(bad_unc: np.ndarray) -> None:
    with pytest.raises(ValueError, match="uncertainty must be finite"):
        ei(
            ids=np.array(["a", "b"], dtype=str),
            scores=np.array([0.4, 0.6], dtype=float),
            scalar_uncertainty=bad_unc,
            top_k=1,
            objective="maximize",
            tie_handling="competition_rank",
            alpha=1.0,
            beta=1.0,
        )


def test_expected_improvement_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="scores must be finite"):
        ei(
            ids=np.array(["a", "b"], dtype=str),
            scores=np.array([0.4, np.nan], dtype=float),
            scalar_uncertainty=np.array([0.1, 0.2], dtype=float),
            top_k=1,
            objective="maximize",
            tie_handling="competition_rank",
            alpha=1.0,
            beta=1.0,
        )


def test_expected_improvement_degenerate_normalization_orders_by_id() -> None:
    out = ei(
        ids=np.array(["b", "a", "c"], dtype=str),
        scores=np.array([1.0, 1.0, 1.0], dtype=float),
        scalar_uncertainty=np.array([0.2, 0.2, 0.2], dtype=float),
        top_k=2,
        objective="maximize",
        tie_handling="competition_rank",
        alpha=1.0,
        beta=1.0,
    )
    acq = np.asarray(out["score"], dtype=float).reshape(-1)
    assert np.allclose(acq, np.zeros(3, dtype=float))
    np.testing.assert_array_equal(np.asarray(out["order_idx"], dtype=int), np.array([1, 0, 2], dtype=int))


def test_expected_improvement_minimize_scores_normalize_and_rank_consistently() -> None:
    ids = np.array(["a", "b", "c"], dtype=str)
    scores = np.array([1.1, 0.9, 1.2], dtype=float)
    sigma = np.array([0.15, 0.10, 0.20], dtype=float)
    out = ei(
        ids=ids,
        scores=scores,
        scalar_uncertainty=sigma,
        top_k=1,
        objective="minimize",
        tie_handling="competition_rank",
        alpha=1.0,
        beta=1.0,
    )
    acq_norm = np.asarray(out["score"], dtype=float).reshape(-1)
    assert np.all((acq_norm >= 0.0) & (acq_norm <= 1.0))

    incumbent = float(np.min(scores))
    improvement = incumbent - scores
    z = improvement / sigma
    acq_raw = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    primary = -acq_raw
    expected_order = np.lexsort((ids, primary)).astype(int)
    np.testing.assert_array_equal(np.asarray(out["order_idx"], dtype=int), expected_order)


def test_expected_improvement_normalization_preserves_order_monotonicity() -> None:
    rng = np.random.default_rng(17)
    ids = np.asarray([f"id{i:03d}" for i in range(40)], dtype=str)
    scores = rng.uniform(low=-0.2, high=0.4, size=ids.size).astype(float)
    sigma = rng.uniform(low=0.2, high=0.5, size=ids.size).astype(float)
    out = ei(
        ids=ids,
        scores=scores,
        scalar_uncertainty=sigma,
        top_k=10,
        objective="maximize",
        tie_handling="competition_rank",
        alpha=1.7,
        beta=0.8,
    )
    incumbent = float(np.max(scores))
    improvement = scores - incumbent
    z = improvement / sigma
    acq_raw = 1.7 * (improvement * norm.cdf(z)) + 0.8 * (sigma * norm.pdf(z))
    sorted_raw = np.sort(acq_raw)
    if sorted_raw.size > 1 and np.min(np.diff(sorted_raw)) <= 1e-12:
        pytest.skip("raw acquisition ties make strict order comparison ambiguous under float rounding")
    expected_order = np.lexsort((ids, -acq_raw)).astype(int)
    np.testing.assert_array_equal(np.asarray(out["order_idx"], dtype=int), expected_order)


def test_expected_improvement_is_deterministic_for_repeated_calls() -> None:
    kwargs = dict(
        ids=np.array(["z", "a", "k", "b"], dtype=str),
        scores=np.array([0.2, 0.3, 0.1, 0.4], dtype=float),
        scalar_uncertainty=np.array([0.05, 0.07, 0.06, 0.08], dtype=float),
        top_k=2,
        objective="maximize",
        tie_handling="competition_rank",
        alpha=1.0,
        beta=1.0,
    )
    out_a = ei(**kwargs)
    out_b = ei(**kwargs)
    np.testing.assert_allclose(np.asarray(out_a["score"], dtype=float), np.asarray(out_b["score"], dtype=float))
    np.testing.assert_array_equal(np.asarray(out_a["order_idx"], dtype=int), np.asarray(out_b["order_idx"], dtype=int))
