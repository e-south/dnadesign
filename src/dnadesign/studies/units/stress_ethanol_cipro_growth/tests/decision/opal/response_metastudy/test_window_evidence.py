"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_window_evidence.py

Equal-footing response-window evidence tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    window_evidence,
)


def test_reader_request_declares_comprehensive_geometric_windows() -> None:
    repo_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    request_path = (
        repo_root
        / "src/dnadesign/studies/units/stress_ethanol_cipro_growth"
        / "response_window_observations/config/reader_response_window.yaml"
    )
    payload = yaml.safe_load(request_path.read_text(encoding="utf-8"))
    reductions = {str(row["id"]): row for row in payload["reductions"]}

    assert {
        "event_logmean_0_6h_post": (0.0, 6.0),
        "event_logmean_4_8h_post": (4.0, 8.0),
        "event_logmean_6_12h_post": (6.0, 12.0),
        "event_logmean_8_12h_post": (8.0, 12.0),
        "event_logmean_0_12h_post": (0.0, 12.0),
    } == {
        reduction_id: (
            float(reductions[reduction_id]["window_start_event_h"]),
            float(reductions[reduction_id]["window_end_event_h"]),
        )
        for reduction_id in (
            "event_logmean_0_6h_post",
            "event_logmean_4_8h_post",
            "event_logmean_6_12h_post",
            "event_logmean_8_12h_post",
            "event_logmean_0_12h_post",
        )
    }
    assert [row["id"] for row in payload["reductions"] if row["role"] == "primary"] == ["event_logmean_4_8h_post"]


def test_window_evidence_compares_each_reduction_with_assay_and_model_diagnostics() -> None:
    result = window_evidence.build_response_window_evidence(
        labels=_labels(),
        margin_rows=_margin_rows(),
        reader_designs=_reader_designs(),
        reader_wells=_reader_wells(),
        reader_traces=_reader_traces(),
        model_screen=_model_screen(),
        reference_design_id="pDual-10",
        response_controls={
            "ethanol": "pDual-10-spyp",
            "ciprofloxacin": "pDual-10-sulAp",
        },
    )

    assert result["reduction_id"].tolist() == ["early", "primary"]
    assert result["response_semantics"].eq("global_target_state_separation").all()
    assert result["window_selection_basis"].eq("assay_evidence_not_model_performance").all()
    assert result["model_evidence_use"].eq("diagnostic_only").all()
    assert result["pdual_response_within_experiment_median_range"].gt(0.0).all()
    assert result["pdual_magnitude_within_experiment_median_range"].gt(0.0).all()
    assert result["pdual_response_cross_experiment_max_state_range"].gt(0.0).all()
    assert result["pdual_magnitude_cross_experiment_max_state_range"].gt(0.0).all()
    assert result["spyp_ethanol_experiment_count"].eq(2).all()
    assert result["spyp_ethanol_response_separation_median"].tolist() == [1.5, 2.0]
    assert result["sulap_ciprofloxacin_experiment_count"].eq(2).all()
    assert result["sulap_ciprofloxacin_response_separation_median"].tolist() == [2.5, 3.0]
    assert result["growth_endpoint_od600_q90"].notna().all()
    assert result["event_sensitivity_median_half_range"].gt(0.0).all()
    assert result["repeated_design_count"].eq(2).all()
    assert result["repeat_maximum_channel_range"].gt(0.0).all()
    assert result["campaign_random_forest_weakest_ordering_spearman"].tolist() == [0.1, 0.2]
    assert result["pls4_weakest_ordering_spearman"].tolist() == [0.3, 0.4]
    assert result["pls6_weakest_ordering_spearman"].tolist() == [0.2, 0.35]
    assert result["censoring_observability"].eq("reader_v5_midpoint_and_event_bounds").all()
    assert result["bounded_design_state_component_count"].eq(0).all()
    assert result["event_sensitivity_censored_design_state_component_count"].eq(0).all()


def test_window_evidence_rejects_a_reduction_missing_from_one_comparison_lane() -> None:
    missing = _model_screen().loc[lambda frame: ~frame["representation_id"].eq("early")].copy()

    with pytest.raises(ValueError, match="equal-footing model screen"):
        window_evidence.build_response_window_evidence(
            labels=_labels(),
            margin_rows=_margin_rows(),
            reader_designs=_reader_designs(),
            reader_wells=_reader_wells(),
            reader_traces=_reader_traces(),
            model_screen=missing,
            reference_design_id="pDual-10",
            response_controls={"ethanol": "pDual-10-spyp", "ciprofloxacin": "pDual-10-sulAp"},
        )


def _labels() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reduction_id, start, end in (("early", 0.0, 6.0), ("primary", 6.0, 12.0)):
        for candidate_id, design_id in (
            ("spyp", "pDual-10-spyp"),
            ("sulap", "pDual-10-sulAp"),
        ):
            rows.append(
                {
                    "id": candidate_id,
                    "design_id": design_id,
                    "reader_experiment_id": "exp-a",
                    "reduction_id": reduction_id,
                    "reduction_method": "geometric_time_mean",
                    "response_basis": "post_window",
                    "screen_role": "primary" if reduction_id == "primary" else "sensitivity",
                    "window_start_event_h": start,
                    "window_end_event_h": end,
                    **{
                        f"{prefix}{state}_event_half_range": 0.1
                        for prefix in ("r", "b")
                        for state in ("00", "10", "01", "11")
                    },
                }
            )
    return pd.DataFrame.from_records(rows)


def _margin_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    separations = {
        ("early", "pDual-10-spyp", "ethanol"): 1.5,
        ("primary", "pDual-10-spyp", "ethanol"): 2.0,
        ("early", "pDual-10-sulAp", "ciprofloxacin"): 2.5,
        ("primary", "pDual-10-sulAp", "ciprofloxacin"): 3.0,
    }
    for (reduction_id, design_id, view_id), value in separations.items():
        rows.append(
            {
                "reduction_id": reduction_id,
                "design_id": design_id,
                "selection_view_id": view_id,
                "response_separation": value,
            }
        )
    return pd.DataFrame.from_records(rows)


def _reader_designs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reduction_id in ("early", "primary"):
        for experiment_offset, experiment_id in enumerate(("exp-a", "exp-b")):
            for design_offset, design_id in enumerate(("pDual-10-spyp", "pDual-10-sulAp")):
                reduction_offset = 0.5 if reduction_id == "primary" else 0.0
                separation = float(experiment_offset + 1 + design_offset + reduction_offset)
                if design_id == "pDual-10-spyp":
                    response = {"r00": 0.0, "r10": separation, "r01": 0.0, "r11": separation}
                else:
                    response = {"r00": 0.0, "r10": 0.0, "r01": separation, "r11": separation}
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "design_id": design_id,
                        "reduction_id": reduction_id,
                        "is_reference": False,
                        **response,
                        "b00": 0.0,
                        "b10": 1.0,
                        "b01": 2.0,
                        "b11": 3.0,
                        **{
                            f"{prefix}{state}_{suffix}": False if suffix.startswith("has_") else "exact"
                            for prefix in ("r", "b")
                            for state in ("00", "10", "01", "11")
                            for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
                        },
                        **{
                            f"{prefix}{state}_event_sensitivity_has_{cause}": False
                            for prefix in ("r", "b")
                            for state in ("00", "10", "01", "11")
                            for cause in ("policy_clipping", "instrument_overflow")
                        },
                    }
                )
    return pd.DataFrame.from_records(rows)


def _reader_wells() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reduction_id in ("early", "primary"):
        for experiment_offset, experiment_id in enumerate(("exp-a", "exp-b")):
            for state_offset, state in enumerate(("00", "10", "01", "11")):
                for well_offset, position in enumerate(("A1", "A2")):
                    rows.append(
                        {
                            "experiment_id": experiment_id,
                            "design_id": "pDual-10",
                            "reduction_id": reduction_id,
                            "state": state,
                            "position": position,
                            "is_reference": True,
                            "response_well": experiment_offset + state_offset + well_offset * 0.2,
                            "magnitude_well": experiment_offset * 2 + state_offset + well_offset * 0.4,
                            "response_policy_clipped_point_count": 0,
                            "response_instrument_overflow_point_count": 0,
                            "response_bound_kind": "exact",
                            "magnitude_policy_clipped_point_count": 0,
                            "magnitude_instrument_overflow_point_count": 0,
                            "magnitude_bound_kind": "exact",
                        }
                    )
    return pd.DataFrame.from_records(rows)


def _reader_traces() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for design_id in ("pDual-10", "pDual-10-spyp", "pDual-10-sulAp"):
        for position in ("A1", "A2"):
            for time_from_event_h, value in ((0.0, 0.2), (6.0, 0.5), (12.0, 0.9)):
                rows.append(
                    {
                        "experiment_id": "exp-a",
                        "design_id": design_id,
                        "position": position,
                        "state": "00",
                        "time_from_event_h": time_from_event_h,
                        "value": value,
                        "value_policy_clipped": False,
                        "value_instrument_overflow": False,
                        "value_bound_kind": "exact",
                        "signal_kind": "growth",
                        "is_reference": design_id == "pDual-10",
                    }
                )
    return pd.DataFrame.from_records(rows)


def _model_screen() -> pd.DataFrame:
    values = {
        "early": {"campaign_random_forest": 0.1, "pls4": 0.3, "pls6": 0.2},
        "primary": {"campaign_random_forest": 0.2, "pls4": 0.4, "pls6": 0.35},
    }
    return pd.DataFrame.from_records(
        [
            {
                "representation_id": reduction_id,
                "model_id": model_id,
                "weakest_required_ordering_spearman": value,
            }
            for reduction_id, model_values in values.items()
            for model_id, value in model_values.items()
        ]
    )
