"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_response_screen_manifest.py

Manifest tests for the response-model evidence screen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core import response_contracts
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    response_screen_publication,
)

_PRIMARY_REDUCTION = "event_logmean_4_8h_post"


def test_manifest_gates_on_exact_campaign_model_not_best_challenger() -> None:
    screen = _screen(
        [
            _model_row(
                model_id="campaign_random_forest",
                model_role="campaign_model",
                representation_id=_PRIMARY_REDUCTION,
                response_ordering=0.10,
                feasibility_ordering=0.40,
            ),
            _model_row(
                model_id="campaign_random_forest",
                model_role="fixed_challenger",
                representation_id=f"{_PRIMARY_REDUCTION}__factorial_contrast7",
                response_ordering=0.20,
                feasibility_ordering=0.20,
            ),
            _model_row(
                model_id="pls4",
                model_role="fixed_challenger",
                representation_id=_PRIMARY_REDUCTION,
                response_ordering=0.80,
                feasibility_ordering=0.90,
            ),
            _model_row(
                model_id="mean_baseline",
                model_role="baseline",
                representation_id=_PRIMARY_REDUCTION,
                response_ordering=0.0,
                feasibility_ordering=0.0,
            ),
        ]
    )

    manifest = response_screen_publication.response_screen_manifest(
        screen,
        primary_reduction_id=_PRIMARY_REDUCTION,
        campaign_to_screen_calibration={"status": "aligned"},
        campaign_model_params={"n_estimators": 100, "random_state": 7, "oob_score": True},
    )

    assert manifest["campaign_model_screen"]["model_id"] == "campaign_random_forest"
    assert manifest["campaign_model_screen"]["representation_id"] == _PRIMARY_REDUCTION
    assert manifest["campaign_model_screen"]["target_transform"] == "none"
    assert manifest["campaign_model_screen"]["posture"] == "configured_campaign_model_not_promoted"
    assert manifest["campaign_model_screen"]["median_channel_spearman"] == 0.1
    assert manifest["campaign_model_screen"]["minimum_channel_spearman"] == 0.0
    assert manifest["campaign_model_screen"]["response_magnitude_mae"] == 1.25
    assert manifest["campaign_model_screen"]["configured_model_params"] == {
        "n_estimators": 100,
        "oob_score": True,
        "random_state": 7,
    }
    assert manifest["best_fixed_model_screen"]["model_id"] == "pls4"
    assert manifest["best_fixed_model_screen"]["posture"] == "descriptive_challenger_not_promoted"
    assert manifest["baseline_model_screen"]["model_id"] == "mean_baseline"
    assert manifest["campaign_model_screen"]["target_view_ordering"] == {
        "ethanol": {
            "defined_group_count": 8,
            "feasibility_spearman": 0.4,
            "response_separation_spearman": 0.1,
        }
    }
    assert {row["model_id"] for row in manifest["prespecified_model_screens"]} == {
        "campaign_random_forest",
        "mean_baseline",
        "pls4",
    }
    assert any(row["model_id"] == "campaign_random_forest" for row in manifest["fixed_model_definitions"])
    assert manifest["model_support_ready"] is False
    assert manifest["model_support_basis"] == "configured_campaign_model"
    assert manifest["evidence_timing"] == "retrospective"
    assert manifest["response_semantics"] == "global_target_state_separation"
    assert manifest["window_comparison"] == {
        "reduction_count": 1,
        "window_selection_basis": "assay_evidence_not_model_performance",
        "model_evidence_use": "diagnostic_only",
        "trajectory_role": "diagnostic_only_not_label_reduction",
    }


def test_manifest_rejects_missing_exact_campaign_representation() -> None:
    screen = _screen(
        [
            _model_row(
                model_id="campaign_random_forest",
                model_role="fixed_challenger",
                representation_id=f"{_PRIMARY_REDUCTION}__factorial_contrast7",
                response_ordering=0.80,
                feasibility_ordering=0.80,
            ),
            _model_row(
                model_id="pls4",
                model_role="fixed_challenger",
                representation_id=_PRIMARY_REDUCTION,
                response_ordering=0.90,
                feasibility_ordering=0.90,
            ),
            _model_row(
                model_id="mean_baseline",
                model_role="baseline",
                representation_id=_PRIMARY_REDUCTION,
                response_ordering=0.0,
                feasibility_ordering=0.0,
            ),
        ]
    )

    with pytest.raises(ValueError, match="exactly one configured campaign-model row"):
        response_screen_publication.response_screen_manifest(
            screen,
            primary_reduction_id=_PRIMARY_REDUCTION,
            campaign_to_screen_calibration={"status": "aligned"},
            campaign_model_params={"n_estimators": 100, "random_state": 7},
        )


def test_manifest_records_undefined_campaign_ordering_as_no_model_support() -> None:
    screen = _screen(
        [
            _model_row(
                model_id="campaign_random_forest",
                model_role="campaign_model",
                representation_id=_PRIMARY_REDUCTION,
                response_ordering=float("nan"),
                feasibility_ordering=0.40,
            ),
            _model_row(
                model_id="pls4",
                model_role="fixed_challenger",
                representation_id=_PRIMARY_REDUCTION,
                response_ordering=0.80,
                feasibility_ordering=0.80,
            ),
            _model_row(
                model_id="mean_baseline",
                model_role="baseline",
                representation_id=_PRIMARY_REDUCTION,
                response_ordering=0.0,
                feasibility_ordering=0.0,
            ),
        ]
    )
    campaign_mask = screen.model_screen["model_role"].eq("campaign_model")
    screen.model_screen.loc[campaign_mask, "all_target_view_metrics_finite"] = False

    manifest = response_screen_publication.response_screen_manifest(
        screen,
        primary_reduction_id=_PRIMARY_REDUCTION,
        campaign_to_screen_calibration={"status": "aligned"},
        campaign_model_params={"n_estimators": 100, "random_state": 7},
    )

    assert manifest["campaign_model_screen"]["weakest_target_view_response_separation_spearman"] is None
    assert manifest["model_support_ready"] is False


def _model_row(
    *,
    model_id: str,
    model_role: str,
    representation_id: str,
    response_ordering: float,
    feasibility_ordering: float,
) -> dict[str, object]:
    return {
        "representation_id": representation_id,
        "promotion_eligible": True,
        "model_id": model_id,
        "model_role": model_role,
        "target_transform": "none",
        "validation": "leave_one_reader_experiment_out",
        "all_target_view_metrics_finite": True,
        "weakest_target_view_response_separation_spearman": response_ordering,
        "weakest_target_view_feasibility_spearman": feasibility_ordering,
        "weakest_required_ordering_spearman": min(response_ordering, feasibility_ordering),
        "median_channel_spearman": response_ordering,
        "minimum_channel_spearman": response_ordering - 0.1,
        "response_magnitude_mae": 1.25,
        "minimum_defined_group_count": 8,
        "metric_scope": "median_within_held_out_experiment",
        "ethanol__response_separation_spearman": response_ordering,
        "ethanol__feasibility_spearman": feasibility_ordering,
        "ethanol__defined_group_count": 8,
    }


def _screen(model_rows: list[dict[str, object]]) -> response_contracts.ResponseMetricScreen:
    labels = pd.DataFrame(
        {
            "id": ["candidate-a"],
            "reduction_id": [_PRIMARY_REDUCTION],
            "screen_role": ["primary"],
            "response_basis": ["event_relative"],
            "reduction_method": ["geometric_log_mean"],
            "window_start_event_h": [4.0],
            "window_end_event_h": [8.0],
        }
    )
    return response_contracts.ResponseMetricScreen(
        event_intervals=pd.DataFrame(
            {
                "event_interval_start_assay_h": [1.0],
                "event_interval_end_assay_h": [2.0],
            }
        ),
        labels=labels,
        margins=pd.DataFrame(),
        stability=pd.DataFrame(
            {
                "reduction_id": [_PRIMARY_REDUCTION],
                "selection_view_id": ["ethanol"],
                "zero_constraint_feasible_count": [1],
            }
        ),
        uncertainty=pd.DataFrame(),
        calibration=pd.DataFrame(
            {
                "selection_view_id": ["ethanol"],
                "component": ["response_separation"],
                "threshold": [0.0],
                "scale": [1.0],
                "bootstrap_samples": [100],
            }
        ),
        model_screen=pd.DataFrame.from_records(model_rows),
        model_group_metrics=pd.DataFrame(),
        retrospective_enrichment=pd.DataFrame(),
        enrichment_summary=pd.DataFrame(),
        campaign_greedy_support=pd.DataFrame(),
        best_fixed_challenger_greedy_support=pd.DataFrame(),
        repeated_measurements=pd.DataFrame(),
        repeated_agreement=pd.DataFrame({"maximum_selected_to_median_abs_difference": [0.0]}),
        window_evidence=pd.DataFrame(
            {
                "reduction_id": [_PRIMARY_REDUCTION],
                "response_semantics": ["global_target_state_separation"],
                "window_selection_basis": ["assay_evidence_not_model_performance"],
                "model_evidence_use": ["diagnostic_only"],
                "trajectory_role": ["diagnostic_only_not_label_reduction"],
            }
        ),
    )
