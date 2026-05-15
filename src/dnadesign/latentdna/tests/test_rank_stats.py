from __future__ import annotations

import math

from dnadesign.latentdna.src.stats.rank import (
    kendall_tau_b,
    linear_r2,
    pearson_correlation,
    rankdata_average,
    spearman_correlation,
)


def test_rankdata_average_uses_one_based_tie_adjusted_ranks() -> None:
    assert rankdata_average([3.0, 1.0, 1.0, 2.0]).tolist() == [4.0, 1.5, 1.5, 3.0]


def test_rank_statistics_return_nan_for_degenerate_inputs() -> None:
    assert math.isnan(spearman_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]))
    assert math.isnan(kendall_tau_b([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]))
    assert math.isnan(pearson_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]))
    assert math.isnan(linear_r2([1.0], [2.0]))


def test_kendall_tau_b_adjusts_for_ties() -> None:
    value = kendall_tau_b([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0])

    assert math.isclose(value, 0.8)


def test_spearman_and_linear_r2_report_positive_monotonic_alignment() -> None:
    assert math.isclose(spearman_correlation([1.0, 2.0, 3.0], [0.2, 0.5, 0.7]), 1.0)
    assert math.isclose(linear_r2([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
