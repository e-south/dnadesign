"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_elite_selection.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from dnadesign.cruncher.app.sample_workflow import (
    _elite_filter_passes,
    _elite_rank_key,
    _EliteCandidate,
    _filter_elite_candidates,
)


def test_elite_rank_prefers_combined_and_min_norm() -> None:
    a = _elite_rank_key(5.0, 0.1, 1.5)
    b = _elite_rank_key(5.5, 0.4, 1.0)
    assert b > a


def test_elite_filter_backward_compat_pwm_sum_only() -> None:
    norm_map = {"tf1": 0.1, "tf2": 0.1}
    assert _elite_filter_passes(
        norm_map=norm_map,
        min_norm=0.1,
        sum_norm=0.2,
        min_per_tf_norm=None,
        require_all_tfs_over_min_norm=True,
        pwm_sum_min=0.2,
    )
    assert not _elite_filter_passes(
        norm_map=norm_map,
        min_norm=0.1,
        sum_norm=0.19,
        min_per_tf_norm=None,
        require_all_tfs_over_min_norm=True,
        pwm_sum_min=0.2,
    )


def test_elite_filter_min_per_tf_norm() -> None:
    norm_map = {"tf1": 0.2, "tf2": 0.05}
    assert not _elite_filter_passes(
        norm_map=norm_map,
        min_norm=0.05,
        sum_norm=0.25,
        min_per_tf_norm=0.1,
        require_all_tfs_over_min_norm=True,
        pwm_sum_min=0.0,
    )
    assert not _elite_filter_passes(
        norm_map=norm_map,
        min_norm=0.05,
        sum_norm=0.25,
        min_per_tf_norm=0.1,
        require_all_tfs_over_min_norm=False,
        pwm_sum_min=0.0,
    )


def test_post_polish_filter_reapplied() -> None:
    cand_ok = _EliteCandidate(
        seq_arr=np.array([0, 1], dtype=np.int8),
        chain_id=0,
        draw_idx=0,
        combined_score=1.0,
        min_norm=0.5,
        sum_norm=0.5,
        per_tf_map={"tf1": 1.0},
        norm_map={"tf1": 0.5},
    )
    cand_bad = _EliteCandidate(
        seq_arr=np.array([1, 1], dtype=np.int8),
        chain_id=0,
        draw_idx=1,
        combined_score=0.5,
        min_norm=0.1,
        sum_norm=0.1,
        per_tf_map={"tf1": 0.5},
        norm_map={"tf1": 0.1},
    )
    filtered = _filter_elite_candidates(
        [cand_ok, cand_bad],
        min_per_tf_norm=0.4,
        require_all_tfs_over_min_norm=True,
        pwm_sum_min=0.0,
    )
    assert filtered == [cand_ok]
