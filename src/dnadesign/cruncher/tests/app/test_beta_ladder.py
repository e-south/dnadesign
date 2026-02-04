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

from dnadesign.cruncher.app.sample_workflow import _default_beta_ladder, _resolve_beta_ladder
from dnadesign.cruncher.config.schema_v2 import BetaLadderGeometric, OptimizersConfig, PTOptimizerConfig


def test_geometric_beta_ladder_spacing() -> None:
    pt_cfg = PTOptimizerConfig(beta_ladder=BetaLadderGeometric(beta_min=0.1, beta_max=1.6, n_temps=4))
    ladders, _ = _resolve_beta_ladder(OptimizersConfig(pt=pt_cfg).pt)
    ratios = [ladders[i + 1] / ladders[i] for i in range(len(ladders) - 1)]
    assert np.allclose(ratios, ratios[0])


def test_geometric_beta_ladder_requires_positive() -> None:
    with pytest.raises(ValueError, match="beta_min>0"):
        _default_beta_ladder(3, 0.0, 1.0)
