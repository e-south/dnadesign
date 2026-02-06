"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_a/test_stage_a_mining_audit.py

Stage-A mining audit helpers for tail unique slope.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.adapters.sources.stage_a.stage_a_metrics import _tail_unique_slope


def test_tail_unique_slope_windowed_ratio() -> None:
    generated = [100, 200, 350, 500]
    unique = [10, 20, 28, 30]
    audit = _tail_unique_slope(generated, unique, window=2)
    assert audit is not None
    assert audit["unique_slope_window"] == 2
    assert audit["unique_slope_generated"] == 150
    assert audit["unique_slope"] == pytest.approx(2 / 150)


def test_tail_unique_slope_requires_delta() -> None:
    audit = _tail_unique_slope([100], [10], window=2)
    assert audit is None
