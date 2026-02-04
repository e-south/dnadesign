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
from dnadesign.cruncher.config.schema_v2 import (
    InitConfig,
    SampleBudgetConfig,
    SampleConfig,
    SampleElitesConfig,
    SampleElitesSelectionConfig,
    SampleElitesSelectionDistanceConfig,
    SampleObjectiveConfig,
)


def _sample_cfg(*, policy: str, ds_mode: str) -> SampleConfig:
    selection = SampleElitesSelectionConfig(
        policy=policy,
        distance=SampleElitesSelectionDistanceConfig(dsDNA=ds_mode),
    )
    elites = SampleElitesConfig(selection=selection)
    objective = SampleObjectiveConfig(bidirectional=True)
    return SampleConfig(
        budget=SampleBudgetConfig(tune=1, draws=1, restarts=1),
        init=InitConfig(kind="random", length=12),
        elites=elites,
        objective=objective,
    )


def test_dsdna_equivalence_enabled_for_mmr_auto() -> None:
    sample_cfg = _sample_cfg(policy="mmr", ds_mode="auto")
    assert dsdna_equivalence_enabled(sample_cfg) is True


def test_dsdna_equivalence_disabled_for_top_score() -> None:
    selection = SampleElitesSelectionConfig(
        policy="top_score",
        distance=SampleElitesSelectionDistanceConfig(dsDNA="auto"),
    )
    elites = SampleElitesConfig(selection=selection, dsDNA_canonicalize=False, dsDNA_hamming=False)
    objective = SampleObjectiveConfig(bidirectional=True)
    sample_cfg = SampleConfig(
        budget=SampleBudgetConfig(tune=1, draws=1, restarts=1),
        init=InitConfig(kind="random", length=12),
        elites=elites,
        objective=objective,
    )
    assert dsdna_equivalence_enabled(sample_cfg) is False
