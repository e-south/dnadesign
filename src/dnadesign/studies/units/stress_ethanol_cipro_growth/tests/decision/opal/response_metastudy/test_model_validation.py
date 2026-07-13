"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_model_validation.py

Repeated held-out validation tests for the response metric metastudy.

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
    model_validation,
    pressure_rows,
)

TARGET_VIEWS = (
    StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0)),
    StressTargetView("ciprofloxacin", "Ciprofloxacin", (0.0, 0.0, 1.0, 1.0)),
    StressTargetView("and", "AND", (0.0, 0.0, 0.0, 1.0)),
)


def test_repeated_cross_validation_reports_target_and_selection_view_metrics() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 6))
    base = x[:, :4]
    y = np.column_stack(
        [
            *(1.0 / (1.0 + np.exp(-base[:, idx])) for idx in range(4)),
            *(base[:, idx] for idx in range(4)),
        ]
    )
    target_views = (StressTargetView("ethanol", "Synthetic ethanol", (0.0, 1.0, 0.0, 1.0)),)

    result = model_validation.cross_validate_random_forest(
        x,
        y,
        target_views=target_views,
        target_view_denoms={"ethanol": 10.0},
        model_params={"n_estimators": 10, "criterion": "squared_error", "bootstrap": True},
        seeds=(3, 7),
        n_splits=4,
        yops_eps=1.0e-8,
        scaling_percentile=95,
        scaling_min_n=5,
        scaling_eps=1.0e-8,
        intensity_log2_offset_delta=0.0,
    )

    assert len(result) == 18
    assert set(result["scope"]) == {"target", "selection_view_objective"}
    assert set(result["seed"]) == {3, 7}
    assert result[["r2", "mae", "spearman"]].notna().all().all()


def test_repeated_cross_validation_is_byte_stable_for_fixed_seeds() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(20, 6))
    y = rng.normal(size=(20, 8))
    y[:, :4] = 1.0 / (1.0 + np.exp(-y[:, :4]))
    kwargs = {
        "target_views": (StressTargetView("ethanol", "Ethanol", (0.0, 1.0, 0.0, 1.0)),),
        "target_view_denoms": {"ethanol": 10.0},
        "model_params": {"n_estimators": 10, "n_jobs": -1},
        "seeds": (3,),
        "n_splits": 4,
        "yops_eps": 1.0e-8,
        "scaling_percentile": 95,
        "scaling_min_n": 5,
        "scaling_eps": 1.0e-8,
        "intensity_log2_offset_delta": 0.0,
    }

    first = model_validation.cross_validate_random_forest(x, y, **kwargs)
    second = model_validation.cross_validate_random_forest(x, y, **kwargs)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_grouped_model_validation_holds_out_complete_experiments() -> None:
    x, y = _synthetic_xy()
    groups = np.asarray(["e1", "e1", "e2", "e2", "e3", "e3"])

    result = model_validation.cross_validate_random_forest_by_group(
        x,
        y,
        groups=groups,
        target_views=TARGET_VIEWS,
        target_view_denoms={"ethanol": 1.0, "ciprofloxacin": 1.0, "and": 1.0},
        model_params={"n_estimators": 8, "random_state": 7},
        seeds=(3,),
        yops_eps=1.0e-8,
        scaling_percentile=95,
        scaling_min_n=2,
        scaling_eps=1.0e-8,
        intensity_log2_offset_delta=0.0,
    )

    assert set(result["split_strategy"]) == {"leave_one_experiment_out"}
    assert set(result["group_count"]) == {3}
    assert set(result["n"]) == {6}


def test_grouped_model_validation_rejects_one_experiment() -> None:
    x, y = _synthetic_xy()

    with pytest.raises(ValueError, match="at least two groups"):
        model_validation.cross_validate_random_forest_by_group(
            x,
            y,
            groups=np.asarray(["e1"] * len(x)),
            target_views=TARGET_VIEWS,
            target_view_denoms={"ethanol": 1.0, "ciprofloxacin": 1.0, "and": 1.0},
            model_params={"n_estimators": 8, "random_state": 7},
            seeds=(3,),
            yops_eps=1.0e-8,
            scaling_percentile=95,
            scaling_min_n=2,
            scaling_eps=1.0e-8,
            intensity_log2_offset_delta=0.0,
        )


def test_weak_held_out_target_view_ordering_blocks_policy_promotion() -> None:
    row = pressure_rows.model_support_row(
        {
            "weakest_target_view_median_score_spearman": 0.12,
            "target_view_median_score_spearman": {
                "ethanol": 0.12,
                "ciprofloxacin": 0.18,
                "and": 0.21,
            },
        },
        minimum=0.3,
    )

    assert row["status"] == "fail"
    assert row["severity"] == "blocker"


def _synthetic_xy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    x = rng.normal(size=(6, 4))
    y = rng.normal(size=(6, 8))
    y[:, :4] = 1.0 / (1.0 + np.exp(-y[:, :4]))
    return x, y
