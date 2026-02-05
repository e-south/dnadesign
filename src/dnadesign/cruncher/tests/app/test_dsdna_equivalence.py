"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_dsdna_equivalence.py

Validate dsDNA equivalence resolution for bidirectional scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.app.sample.diagnostics import dsdna_equivalence_enabled
from dnadesign.cruncher.config.schema_v2 import InitConfig, SampleBudgetConfig, SampleConfig, SampleObjectiveConfig


def _sample_cfg(*, bidirectional: bool) -> SampleConfig:
    objective = SampleObjectiveConfig(bidirectional=bidirectional)
    return SampleConfig(
        budget=SampleBudgetConfig(tune=1, draws=1, restarts=1),
        init=InitConfig(kind="random", length=12),
        objective=objective,
    )


def test_dsdna_equivalence_enabled_for_bidirectional() -> None:
    sample_cfg = _sample_cfg(bidirectional=True)
    assert dsdna_equivalence_enabled(sample_cfg) is True


def test_dsdna_equivalence_disabled_for_unidirectional() -> None:
    sample_cfg = _sample_cfg(bidirectional=False)
    assert dsdna_equivalence_enabled(sample_cfg) is False
