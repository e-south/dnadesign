"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/selection/test_expected_improvement_edge_cases.py

Edge-case regression tests for expected-improvement acquisition behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from dnadesign.opal.src.selection.expected_improvement import ei


def _minmax_01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _expected_ei_scores(
    *,
    scores: np.ndarray,
    sigma: np.ndarray,
    objective: str,
    alpha: float,
    beta: float,
    normalize_sigma_for_explore: bool = True,
    normalize_sigma_for_z: bool = False,
) -> np.ndarray:
    preds = np.asarray(scores, dtype=float).reshape(-1)
    unc = np.asarray(sigma, dtype=float).reshape(-1)
    mode = str(objective).strip().lower()
    if mode == "maximize":
        incumbent = float(np.max(preds))
        improvement = preds - incumbent
    elif mode == "minimize":
        incumbent = float(np.min(preds))
        improvement = incumbent - preds
    else:
        raise ValueError(f"Unsupported objective mode: {objective}")

    sigma_for_z = _minmax_01(unc) if normalize_sigma_for_z else unc
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(
            improvement,
            sigma_for_z,
            out=np.zeros_like(improvement, dtype=float),
            where=sigma_for_z > 0.0,
        )
    sigma_for_explore = _minmax_01(unc) if normalize_sigma_for_explore else unc
    acquisition = alpha * (improvement * norm.cdf(z)) + beta * (sigma_for_explore * norm.pdf(z))
    return _minmax_01(acquisition)


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

    expected_scores = _expected_ei_scores(
        scores=scores,
        sigma=sigma,
        objective="minimize",
        alpha=1.0,
        beta=1.0,
        normalize_sigma_for_explore=True,
        normalize_sigma_for_z=False,
    )
    np.testing.assert_allclose(acq_norm, expected_scores)
    primary = -expected_scores
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
    expected_scores = _expected_ei_scores(
        scores=scores,
        sigma=sigma,
        objective="maximize",
        alpha=1.7,
        beta=0.8,
        normalize_sigma_for_explore=True,
        normalize_sigma_for_z=False,
    )
    np.testing.assert_allclose(np.asarray(out["score"], dtype=float), expected_scores)
    expected_order = np.lexsort((ids, -expected_scores)).astype(int)
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


def test_expected_improvement_uses_raw_sigma_for_z_and_normalized_sigma_for_explore() -> None:
    scores = np.array([0.5118, 0.9505, 0.1442, 0.9486, 0.3118, 0.4233, 0.8277, 0.4092], dtype=float)
    sigma = np.array([1.4794, 1.3984, 1.1306, 1.7115, 0.6131, 0.9701, 0.7515, 1.7844], dtype=float)
    ids = np.asarray([f"id{i:02d}" for i in range(scores.size)], dtype=str)
    out = ei(
        ids=ids,
        scores=scores,
        scalar_uncertainty=sigma,
        top_k=3,
        objective="maximize",
        tie_handling="competition_rank",
        alpha=1.0,
        beta=1.0,
    )
    actual = np.asarray(out["score"], dtype=float).reshape(-1)

    expected = _expected_ei_scores(
        scores=scores,
        sigma=sigma,
        objective="maximize",
        alpha=1.0,
        beta=1.0,
        normalize_sigma_for_explore=True,
        normalize_sigma_for_z=False,
    )
    np.testing.assert_allclose(actual, expected)

    wrong_z_norm = _expected_ei_scores(
        scores=scores,
        sigma=sigma,
        objective="maximize",
        alpha=1.0,
        beta=1.0,
        normalize_sigma_for_explore=True,
        normalize_sigma_for_z=True,
    )
    assert not np.allclose(expected, wrong_z_norm, rtol=1e-9, atol=1e-9)


def test_expected_improvement_weak_score_variance_correlation_limits_variance_coupling() -> None:
    rng = np.random.default_rng(18266)
    n = 64
    ids = np.asarray([f"id{i:03d}" for i in range(n)], dtype=str)
    scores = rng.uniform(low=0.0, high=1.0, size=n).astype(float)
    sigma = rng.lognormal(mean=0.2, sigma=0.8, size=n).astype(float)
    scalar_variance = np.square(sigma)

    score_var_corr = float(np.corrcoef(scores, scalar_variance)[0, 1])
    assert np.isfinite(score_var_corr)
    assert abs(score_var_corr) < 0.6
    assert float(np.min(scores)) >= 0.0
    assert float(np.max(scores)) <= 1.0

    out = ei(
        ids=ids,
        scores=scores,
        scalar_uncertainty=sigma,
        top_k=5,
        objective="maximize",
        tie_handling="competition_rank",
        alpha=1.0,
        beta=1.0,
    )
    acquisition = np.asarray(out["score"], dtype=float).reshape(-1)
    variance_ei_corr = abs(float(np.corrcoef(scalar_variance, acquisition)[0, 1]))
    assert variance_ei_corr < 0.8
