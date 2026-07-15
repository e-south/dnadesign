"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_response_magnitude.py

Tests for study-owned response and relative-fluorescence target margins.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    response_magnitude,
)


def test_response_separation_keeps_components_separate() -> None:
    summary = pd.DataFrame([_summary_row("primary", "candidate-1", (0.0, 2.0, 1.0, 3.0))])
    ethanol = StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))

    row = response_magnitude.build_response_separation_rows(summary, target_views=(ethanol,)).iloc[0]

    assert row["response_separation"] == 1.0
    assert row["on_magnitude_floor"] == 2.0
    assert row["off_magnitude_ceiling"] == 3.0
    assert row["response_semantics"] == "global_target_state_separation"
    assert not row["passes_all_zero_constraints"]


def test_zero_response_separation_matches_the_rmf_boundary() -> None:
    summary = pd.DataFrame([_summary_row("primary", "candidate-1", (0.0, 1.0, 1.0, 1.0))])
    summary.loc[:, ["b00", "b01"]] = -1.0
    summary.loc[:, ["b10", "b11"]] = 1.0
    ethanol = StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))

    row = response_magnitude.build_response_separation_rows(summary, target_views=(ethanol,)).iloc[0]

    assert row["response_separation"] == 0.0
    assert row["passes_response_zero"]
    assert row["passes_all_zero_constraints"]


def test_response_separation_stability_uses_declared_primary_reduction() -> None:
    summaries = pd.DataFrame(
        [
            _summary_row("primary", "a", (0.0, 2.0, 1.0, 3.0)),
            _summary_row("primary", "b", (0.0, 3.0, 1.0, 4.0)),
            _summary_row("challenger", "a", (0.0, 2.5, 1.0, 3.5)),
            _summary_row("challenger", "b", (0.0, 3.5, 1.0, 4.5)),
        ]
    )
    ethanol = StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))
    margins = response_magnitude.build_response_separation_rows(summaries, target_views=(ethanol,))

    stability = response_magnitude.summarize_response_separation_stability(margins, primary_reduction_id="primary")

    challenger = stability.loc[stability["reduction_id"].eq("challenger")].iloc[0]
    assert challenger["response_separation__spearman_to_primary"] == pytest.approx(1.0)


def _summary_row(reduction_id: str, candidate_id: str, response: tuple[float, ...]) -> dict[str, object]:
    return {
        "id": candidate_id,
        "design_id": candidate_id,
        "reader_experiment_id": "reader-exp",
        "reduction_id": reduction_id,
        "r00": response[0],
        "r10": response[1],
        "r01": response[2],
        "r11": response[3],
        "b00": 1.0,
        "b10": 2.0,
        "b01": 3.0,
        "b11": 4.0,
    }
