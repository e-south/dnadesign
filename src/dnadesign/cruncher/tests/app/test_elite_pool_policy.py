"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_elite_pool_policy.py

Tests elite pool-size resolution policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.app.sample.run_set import _resolve_elite_pool_size


def test_auto_pool_size_prefers_broad_rank_window() -> None:
    assert _resolve_elite_pool_size(pool_size_cfg="auto", elite_k=8, candidate_count=50_000) == 4_000
    assert _resolve_elite_pool_size(pool_size_cfg="auto", elite_k=20, candidate_count=50_000) == 10_000


def test_auto_pool_size_caps_to_available_candidates() -> None:
    assert _resolve_elite_pool_size(pool_size_cfg="auto", elite_k=8, candidate_count=900) == 900


def test_pool_size_all_uses_all_candidates() -> None:
    assert _resolve_elite_pool_size(pool_size_cfg="all", elite_k=8, candidate_count=1234) == 1234


def test_pool_size_int_is_clamped_to_available_candidates() -> None:
    assert _resolve_elite_pool_size(pool_size_cfg=5000, elite_k=8, candidate_count=3000) == 3000
