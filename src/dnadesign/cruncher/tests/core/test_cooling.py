"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_cooling.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.cruncher.core.optimizers.cooling import make_beta_scheduler


def test_linear_schedule_hits_endpoint() -> None:
    cfg = {"kind": "linear", "beta": (0.1, 1.0)}
    beta_of = make_beta_scheduler(cfg, total_sweeps=5)
    assert beta_of(0) == pytest.approx(0.1)
    assert beta_of(4) == pytest.approx(1.0)


def test_piecewise_early_it_uses_first_beta() -> None:
    cfg = {
        "kind": "piecewise",
        "stages": [
            {"sweeps": 5, "beta": 0.2},
            {"sweeps": 10, "beta": 0.6},
        ],
    }
    beta_of = make_beta_scheduler(cfg, total_sweeps=10)
    assert beta_of(0) == pytest.approx(0.2)
    assert beta_of(3) == pytest.approx(0.2)


def test_piecewise_single_stage_constant() -> None:
    cfg = {
        "kind": "piecewise",
        "stages": [{"sweeps": 5, "beta": 0.7}],
    }
    beta_of = make_beta_scheduler(cfg, total_sweeps=5)
    assert beta_of(0) == pytest.approx(0.7)
    assert beta_of(4) == pytest.approx(0.7)
