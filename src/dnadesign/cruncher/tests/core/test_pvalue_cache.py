"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_pvalue_cache.py

Validates log-odds p-value lookup caching behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.core import pvalue


def test_logodds_cache_reuses_results() -> None:
    pvalue.clear_logodds_cache()
    lom = np.array(
        [
            [0.1, -0.2, 0.3, -0.1],
            [0.0, 0.2, -0.1, -0.3],
        ],
        dtype=float,
    )
    bg = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    info_before = pvalue.logodds_cache_info()
    pvalue.logodds_to_p_lookup(lom, bg)
    info_after_first = pvalue.logodds_cache_info()
    assert info_after_first.misses == info_before.misses + 1

    pvalue.logodds_to_p_lookup(lom, bg)
    info_after_second = pvalue.logodds_cache_info()
    assert info_after_second.hits == info_after_first.hits + 1
