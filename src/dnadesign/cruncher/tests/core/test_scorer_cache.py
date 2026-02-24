"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_scorer_cache.py

Validates Scorer cache reuse for precomputed PWM null statistics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.core.scoring import Scorer, clear_pwm_stats_cache, pwm_stats_cache_info


def test_scorer_reuses_cached_pwm_statistics() -> None:
    clear_pwm_stats_cache()
    matrix = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
        ],
        dtype=float,
    )
    pwm = PWM(name="tfA", matrix=matrix)

    info_before = pwm_stats_cache_info()
    Scorer({"tfA": pwm}, bidirectional=False, scale="llr")
    info_after_first = pwm_stats_cache_info()
    assert info_after_first.misses == info_before.misses + 1

    Scorer({"tfA": pwm}, bidirectional=False, scale="llr")
    info_after_second = pwm_stats_cache_info()
    assert info_after_second.hits == info_after_first.hits + 1
