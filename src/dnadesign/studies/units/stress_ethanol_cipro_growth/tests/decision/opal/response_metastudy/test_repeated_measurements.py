"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_repeated_measurements.py

Tests for cross-experiment Reader measurement evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    repeated_measurements,
)


def test_repeated_measurements_preserve_selected_source_and_cross_experiment_range() -> None:
    rows = []
    for experiment, offset in (("exp-a", 0.0), ("exp-b", 0.5)):
        row = {"design_id": "D1", "reader_experiment_id": experiment}
        for prefix in ("r", "b"):
            for corner in ("00", "10", "01", "11"):
                row[f"{prefix}{corner}"] = float(int(corner, 2)) + offset
        rows.append(row)
    selected = pd.DataFrame({"id": ["candidate-1"], "design_id": ["D1"], "reader_experiment_id": ["exp-a"]})

    measurements, agreement = repeated_measurements.build_repeated_measurement_evidence(
        pd.DataFrame.from_records(rows),
        selected_labels=selected,
    )

    assert measurements["is_selected_label_source"].tolist() == [True, False]
    assert agreement.loc[0, "experiment_count"] == 2
    assert agreement.loc[0, "r00__range"] == 0.5
    assert agreement.loc[0, "maximum_selected_to_median_abs_difference"] == 0.25
