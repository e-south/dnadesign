"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_response_uncertainty.py

Tests for target uncertainty derived from Reader bootstrap records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    response_uncertainty,
)


def test_reader_joint_draws_produce_finite_constraint_scales() -> None:
    labels, draws = _reader_records(samples=100)
    ethanol = StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))

    result = response_uncertainty.estimate_response_calibration_from_reader_draws(
        labels,
        draws,
        target_views=(ethanol,),
        scale_quantile=0.9,
        expected_bootstrap_samples=100,
    )

    assert set(result.calibration["component"]) == {
        "response_separation",
        "on_magnitude_floor",
        "off_magnitude_ceiling",
    }
    assert (result.calibration["scale"] > 0.0).all()
    assert result.rows["feasibility_margin"].notna().all()
    assert set(result.calibration["scale_basis"]) == {"reader_joint_bootstrap_plus_conservative_event_bound"}


def test_reader_joint_draws_fail_on_incomplete_candidate_draws() -> None:
    labels, draws = _reader_records(samples=99)
    ethanol = StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="99 bootstrap draws; expected 100"):
        response_uncertainty.estimate_response_calibration_from_reader_draws(
            labels,
            draws,
            target_views=(ethanol,),
            scale_quantile=0.9,
            expected_bootstrap_samples=100,
        )


def test_reader_joint_draws_fail_when_a_candidate_is_missing() -> None:
    labels, draws = _reader_records(samples=100)
    draws = draws.loc[~draws["id"].eq("b")]
    ethanol = StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="lack one or more joint bootstrap summaries"):
        response_uncertainty.estimate_response_calibration_from_reader_draws(
            labels,
            draws,
            target_views=(ethanol,),
            scale_quantile=0.9,
            expected_bootstrap_samples=100,
        )


def _reader_records(*, samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.DataFrame([_state_row(candidate_id) for candidate_id in ("a", "b")])
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    for candidate_id in ("a", "b"):
        base = _state_row(candidate_id)
        for draw_index in range(samples):
            row = {"id": candidate_id, "draw_index": draw_index}
            for column in ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"):
                row[column] = float(base[column]) + float(rng.normal(0.0, 0.05))
            rows.append(row)
    return labels, pd.DataFrame(rows)


def _state_row(candidate_id: str) -> dict[str, object]:
    row: dict[str, object] = {
        "id": candidate_id,
        "design_id": candidate_id,
        "reader_experiment_id": "reader-exp",
        "reduction_id": "primary",
        "r00": 0.0,
        "r10": 2.0,
        "r01": 0.5,
        "r11": 2.5,
        "b00": -0.5,
        "b10": 0.5,
        "b01": -0.25,
        "b11": 0.75,
    }
    for prefix in ("r", "b"):
        for corner in ("00", "10", "01", "11"):
            row[f"{prefix}{corner}_event_half_range"] = 0.05
    return row
