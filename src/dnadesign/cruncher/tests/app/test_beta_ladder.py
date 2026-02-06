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

from dnadesign.cruncher.config.schema_v3 import SamplePtConfig
from dnadesign.cruncher.core.optimizers.cooling import make_beta_ladder


def test_geometric_beta_ladder_spacing() -> None:
    pt_cfg = SamplePtConfig(n_temps=4, temp_max=10.0)
    beta_min = 1.0 / float(pt_cfg.temp_max)
    geometric = list(np.geomspace(beta_min, 1.0, int(pt_cfg.n_temps)))
    ladders = make_beta_ladder({"kind": "geometric", "beta": geometric})
    ratios = [ladders[i + 1] / ladders[i] for i in range(len(ladders) - 1)]
    assert np.allclose(ratios, ratios[0])


def test_pt_temp_max_must_be_positive() -> None:
    with pytest.raises(ValueError, match="sample.pt.temp_max must be >= 1.0"):
        SamplePtConfig(n_temps=3, temp_max=0.0)
