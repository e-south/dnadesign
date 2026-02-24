"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_pvalue_cache.py

Validates log-odds p-value lookup caching behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import product

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


def test_logodds_lookup_matches_exact_integer_enumeration() -> None:
    pvalue.clear_logodds_cache()
    lom = np.array(
        [
            [-1.2, 0.4, 0.1, -0.3],
            [0.5, -0.8, 0.2, 0.1],
            [-0.6, 0.9, -0.2, 0.0],
        ],
        dtype=float,
    )
    bg = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    scores, tail_p = pvalue.logodds_to_p_lookup(lom, bg)

    lom_int = np.round(lom * pvalue._LOGODDS_SCALE).astype(np.int32)
    score_ints = np.round(scores * pvalue._LOGODDS_SCALE).astype(np.int64)
    all_int_scores: list[int] = []
    all_probs: list[float] = []
    width = int(lom_int.shape[0])
    for seq in product(range(4), repeat=width):
        score_int = 0
        prob = 1.0
        for pos, base in enumerate(seq):
            score_int += int(lom_int[pos, base])
            prob *= float(bg[base])
        all_int_scores.append(score_int)
        all_probs.append(prob)

    expected_tail = np.array(
        [sum(prob for score, prob in zip(all_int_scores, all_probs) if score >= int(th)) for th in score_ints],
        dtype=float,
    )
    np.testing.assert_allclose(tail_p, expected_tail, atol=1.0e-12, rtol=0.0)
