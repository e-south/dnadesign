"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_beta_ladder.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pytest

from dnadesign.cruncher.config.schema_v3 import SampleOptimizerCoolingConfig
from dnadesign.cruncher.core.optimizers.cooling import make_beta_scheduler


def test_linear_mcmc_cooling_is_monotonic() -> None:
    cooling = SampleOptimizerCoolingConfig(
        kind="linear",
        beta=None,
        beta_start=0.25,
        beta_end=1.25,
    )
    beta_of = make_beta_scheduler(
        {
            "kind": "linear",
            "beta": (float(cooling.beta_start), float(cooling.beta_end)),
        },
        total_sweeps=7,
    )
    values = np.asarray([beta_of(i) for i in range(7)], dtype=float)

    assert values[0] == pytest.approx(0.25)
    assert values[-1] == pytest.approx(1.25)
    assert np.all(np.diff(values) >= 0.0)


def test_fixed_mcmc_beta_must_be_positive() -> None:
    with pytest.raises(ValueError, match="sample.optimizer.cooling.beta must be > 0"):
        SampleOptimizerCoolingConfig(kind="fixed", beta=0.0)
