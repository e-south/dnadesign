from __future__ import annotations

import math

import numpy as np

from dnadesign.latentdna.src.scalars.common import _pearson_correlation, _spearman_correlation


def test_pearson_correlation_returns_nan_for_degenerate_inputs() -> None:
    assert math.isnan(_pearson_correlation(np.asarray([1.0]), np.asarray([1.0])))
    assert math.isnan(_pearson_correlation(np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0])))


def test_spearman_correlation_returns_nan_for_degenerate_inputs() -> None:
    assert math.isnan(_spearman_correlation(np.asarray([1.0]), np.asarray([1.0])))
    assert math.isnan(_spearman_correlation(np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0])))
