from __future__ import annotations

import math

import numpy as np

from dnadesign.latentdna.src.scalars.common import (
    _pairwise_cosine_distance_summary,
    _pearson_correlation,
    _spearman_correlation,
)


def test_pearson_correlation_returns_nan_for_degenerate_inputs() -> None:
    assert math.isnan(_pearson_correlation(np.asarray([1.0]), np.asarray([1.0])))
    assert math.isnan(_pearson_correlation(np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0])))


def test_spearman_correlation_returns_nan_for_degenerate_inputs() -> None:
    assert math.isnan(_spearman_correlation(np.asarray([1.0]), np.asarray([1.0])))
    assert math.isnan(_spearman_correlation(np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0])))


def test_pairwise_cosine_distance_summary_caps_rows_deterministically() -> None:
    rng = np.random.default_rng(17)
    matrix = np.asarray(rng.normal(size=(12, 4)), dtype=np.float32)

    first = _pairwise_cosine_distance_summary(matrix, max_rows=5, seed=11)
    second = _pairwise_cosine_distance_summary(matrix, max_rows=5, seed=11)

    assert first.method == "seeded_row_sample_all_pairs"
    assert first.source_rows == 12
    assert first.evaluated_rows == 5
    assert first.pair_count == 10
    assert first.median == second.median
    assert first.iqr == second.iqr


def test_pairwise_cosine_distance_summary_uses_exact_pairs_under_cap() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    summary = _pairwise_cosine_distance_summary(matrix, max_rows=5, seed=17)

    assert summary.method == "exact_all_pairs"
    assert summary.source_rows == 3
    assert summary.evaluated_rows == 3
    assert summary.pair_count == 3
    assert not math.isnan(summary.median)
