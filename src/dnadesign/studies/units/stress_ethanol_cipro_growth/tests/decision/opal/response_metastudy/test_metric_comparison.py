"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_metric_comparison.py

Tests for response metric comparison over aligned observed rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    metric_comparison,
)


def test_metric_comparison_joins_same_window_components_and_marks_examples() -> None:
    sfxi = pd.DataFrame(
        [
            {
                "id": "candidate",
                "design_id": "pDual-10-spyp",
                "assay_summary_id": "primary",
                "selection_view_id": "ethanol",
                "logic_fidelity": 0.7,
                "effect_scaled": 0.8,
                "sfxi": 0.56,
            }
        ]
    )
    response = pd.DataFrame(
        [
            {
                "id": "candidate",
                "design_id": "pDual-10-spyp",
                "reader_experiment_id": "experiment",
                "reduction_id": "primary",
                "selection_view_id": "ethanol",
                "response_separation": 2.0,
                "on_magnitude_floor": 1.5,
                "off_magnitude_ceiling": 0.25,
                "feasibility_margin": -0.5,
                "passes_all_zero_constraints": False,
                **{
                    f"{prefix}{state}": float(index)
                    for prefix in ("r", "b")
                    for index, state in enumerate(("00", "10", "01", "11"))
                },
            }
        ]
    )

    result = metric_comparison.build_metric_comparison_rows(
        sfxi,
        response,
        primary_reduction_id="primary",
        examples={"pDual-10-spyp": "SpyP measured ethanol-response example"},
    )

    assert len(result) == 1
    assert result.loc[0, "example_label"] == "SpyP measured ethanol-response example"
    assert result.loc[0, "off_suppression"] == -0.25
    assert result.loc[0, "sfxi"] == 0.56
