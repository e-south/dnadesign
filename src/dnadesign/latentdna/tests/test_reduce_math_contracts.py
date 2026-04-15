"""
Reducer math contract tests for latentdna.
"""

from __future__ import annotations

import numpy as np

from dnadesign.latentdna.src.views.reduce import _fit_pca, _fit_randomized_pca


def _low_rank_matrix(*, rows: int = 256, dims: int = 32) -> np.ndarray:
    rng = np.random.default_rng(17)
    scores = np.asarray(rng.normal(size=(rows, 3)), dtype=np.float32) * np.asarray([9.0, 4.0, 1.5], dtype=np.float32)
    basis_seed = np.asarray(rng.normal(size=(dims, 3)), dtype=np.float32)
    basis, _ = np.linalg.qr(basis_seed, mode="reduced")
    offset = np.asarray(rng.normal(size=(dims,)), dtype=np.float32)
    return np.asarray(scores @ basis.T + offset, dtype=np.float32)


def test_fit_pca_uses_total_variance_for_truncated_ratio() -> None:
    matrix = _low_rank_matrix(rows=128, dims=24)

    mean, components, explained_variance, explained_variance_ratio = _fit_pca(matrix, dims=2)

    total_variance = float(np.var(matrix, axis=0, ddof=1).sum())
    assert np.allclose(components @ components.T, np.eye(2, dtype=np.float32), atol=1e-5)
    assert np.allclose(((matrix - mean) @ components.T).mean(axis=0), 0.0, atol=1e-5)
    assert np.isclose(explained_variance_ratio.sum(), explained_variance.sum() / total_variance, rtol=1e-5)
    assert 0.0 < float(explained_variance_ratio.sum()) < 1.0


def test_randomized_pca_matches_exact_low_rank_subspace() -> None:
    matrix = _low_rank_matrix()

    _, exact_components, exact_variance, exact_ratio = _fit_pca(matrix, dims=3)
    _, randomized_components, randomized_variance, randomized_ratio = _fit_randomized_pca(matrix, dims=3, seed=17)

    overlaps = np.abs(randomized_components @ exact_components.T)
    assert np.allclose(randomized_components @ randomized_components.T, np.eye(3, dtype=np.float32), atol=1e-4)
    assert np.all(overlaps.max(axis=1) > 0.98)
    assert np.allclose(randomized_variance, exact_variance, rtol=5e-2, atol=1e-3)
    assert np.allclose(randomized_ratio, exact_ratio, rtol=5e-2, atol=1e-3)


def test_randomized_pca_ratio_is_scale_invariant_for_tiny_variance() -> None:
    matrix = _low_rank_matrix()

    _, _, _, baseline_ratio = _fit_randomized_pca(matrix, dims=3, seed=17)
    _, _, _, tiny_ratio = _fit_randomized_pca(matrix * np.float32(1e-12), dims=3, seed=17)

    assert np.allclose(tiny_ratio, baseline_ratio, rtol=1e-5, atol=1e-6)
