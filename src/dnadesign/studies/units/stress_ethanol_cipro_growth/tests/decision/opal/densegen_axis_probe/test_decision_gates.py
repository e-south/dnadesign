"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_decision_gates.py

Regression tests for decision gates studies units stress ethanol cipro growth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .helpers import (
    NULL_ORACLE_ID,
    ORACLE_ID,
    _claim_statuses,
    _decision_from_metrics,
    decision_reasons_from_metrics,
    enrich_metric_rows,
    gate_results_from_metrics,
    pytest,
    round_dynamics_summary,
)


def test_decision_rejects_missing_prediction_metrics() -> None:
    with pytest.raises(ValueError, match="missing_predictions"):
        _decision_from_metrics(
            [
                {
                    "run_key": "cipro_positive_random_id",
                    "campaign": "cipro",
                    "oracle_id": ORACLE_ID,
                    "split_id": "random_id",
                    "status": "missing_predictions",
                }
            ],
            {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
        )


def test_decision_stops_when_x_surface_contract_fails() -> None:
    decision = _decision_from_metrics(
        [
            {
                "run_key": "cipro_positive_random_id",
                "campaign": "cipro",
                "oracle_id": ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 2.0,
            }
        ],
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": False},
    )

    assert decision == "STOP"


def test_decision_debugs_when_only_null_metrics_exist() -> None:
    decision = _decision_from_metrics(
        [
            {
                "campaign": "cipro",
                "oracle_id": NULL_ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 1.5,
            }
        ],
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
    )

    assert decision == "DEBUG"


def test_decision_reasons_explain_null_and_pair_failures() -> None:
    metrics = [
        {
            "run_key": "ethanol_positive_leave_sigma35_variant",
            "campaign": "ethanol",
            "oracle_id": ORACLE_ID,
            "split_id": "leave_sigma35_variant",
            "target_lift_at_k_true": 1.51,
        },
        {
            "run_key": "ethanol_null_leave_sigma35_variant",
            "campaign": "ethanol",
            "oracle_id": NULL_ORACLE_ID,
            "split_id": "leave_sigma35_variant",
            "target_lift_at_k_true": 1.79,
        },
    ]

    reasons = decision_reasons_from_metrics(
        metrics,
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
        decision="STOP",
    )
    gate_results = gate_results_from_metrics(
        metrics,
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
    )

    assert any(reason["reason"] == "positive lift does not exceed null lift" for reason in reasons)
    pair_gate = [row for row in gate_results if row.get("campaign") == "ethanol"][0]
    assert pair_gate["status"] == "attention"
    assert pair_gate["reason"] == "null lift exceeds random-baseline lift; diagnostic only"
    assert pair_gate["positive_minus_null_lift"] == pytest.approx(-0.28)


def test_enrich_metric_rows_adds_count_aware_fields() -> None:
    row = enrich_metric_rows(
        [
            {
                "run_key": "ethanol_positive_random_id",
                "selection_k": 6,
                "selected_target_precision_at_k_true": 5 / 6,
                "target_class_prevalence_true": 0.22549019607843138,
            }
        ]
    )[0]

    assert row["selected_target_count_true"] == 5
    assert row["selected_target_count_label_true"] == "5/6"
    assert row["selected_target_binomial_tail_p_true"] == pytest.approx(0.0028, abs=0.0001)


def test_decision_debugs_incomplete_positive_null_pairs() -> None:
    decision = _decision_from_metrics(
        [
            {
                "campaign": "cipro",
                "oracle_id": ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 2.0,
            }
        ],
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
    )

    assert decision == "DEBUG"


def test_decision_pass_is_scoped_to_cipro_random_gate() -> None:
    decision = _decision_from_metrics(
        [
            {
                "campaign": "cipro",
                "oracle_id": ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 3.0,
            },
            {
                "campaign": "cipro",
                "oracle_id": NULL_ORACLE_ID,
                "split_id": "random_id",
                "target_lift_at_k_true": 0.7,
            },
        ],
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
    )

    assert decision == "PASS_CIPRO_RANDOM_GATE"


def test_claim_statuses_ignore_missing_prediction_rows() -> None:
    statuses = _claim_statuses(
        [
            {
                "campaign": "cipro",
                "oracle_id": ORACLE_ID,
                "split_id": "random_id",
                "status": "missing_predictions",
            }
        ],
        decision="DEBUG",
    )

    assert statuses["H-CIPRO"] == "not evaluated in this run"


def test_round_dynamics_marks_transient_null_spikes_as_debug_attention() -> None:
    run_metrics = [
        {
            "campaign": "cipro",
            "oracle_id": ORACLE_ID,
            "split_id": "random_id",
            "target_lift_at_k_true": 3.0,
        },
        {
            "campaign": "cipro",
            "oracle_id": NULL_ORACLE_ID,
            "split_id": "random_id",
            "target_lift_at_k_true": 0.7,
        },
    ]
    round_metrics = [
        {
            "run_key": "cipro_null_random_id",
            "campaign": "cipro",
            "oracle_id": NULL_ORACLE_ID,
            "split_id": "random_id",
            "as_of_round": 0,
            "target_lift_at_k_true": 0.7,
        },
        {
            "run_key": "cipro_null_random_id",
            "campaign": "cipro",
            "oracle_id": NULL_ORACLE_ID,
            "split_id": "random_id",
            "as_of_round": 1,
            "target_lift_at_k_true": 2.0,
        },
        {
            "run_key": "cipro_null_random_id",
            "campaign": "cipro",
            "oracle_id": NULL_ORACLE_ID,
            "split_id": "random_id",
            "as_of_round": 2,
            "target_lift_at_k_true": 0.0,
        },
    ]

    decision = _decision_from_metrics(
        run_metrics,
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
        round_metrics=round_metrics,
    )
    gate_results = gate_results_from_metrics(
        run_metrics,
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
        round_metrics=round_metrics,
    )
    reasons = decision_reasons_from_metrics(
        run_metrics,
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
        decision=decision,
        round_metrics=round_metrics,
    )
    dynamics = round_dynamics_summary(round_metrics)

    assert decision == "DEBUG"
    dynamics_gate = [row for row in gate_results if row["gate"] == "H-NULL-ROUND-DYNAMICS"][0]
    assert dynamics_gate["status"] == "attention"
    assert dynamics_gate["max_lift"] == 2.0
    assert dynamics_gate["final_lift"] == 0.0
    assert any(row["gate"] == "H-TRAJECTORY-SEPARATION" for row in reasons)
    assert dynamics[0]["null_transient_spike"] is True


def test_null_spike_diagnostics_do_not_block_positive_trajectory_separation() -> None:
    run_metrics = [
        {
            "campaign": "cipro",
            "oracle_id": ORACLE_ID,
            "split_id": "random_id",
            "target_lift_at_k_true": 3.0,
        },
        {
            "campaign": "cipro",
            "oracle_id": NULL_ORACLE_ID,
            "split_id": "random_id",
            "target_lift_at_k_true": 0.7,
        },
    ]
    round_metrics = [
        {
            "run_key": "cipro_positive_random_id",
            "campaign": "cipro",
            "oracle_id": ORACLE_ID,
            "split_id": "random_id",
            "as_of_round": 0,
            "target_lift_at_k_true": 2.0,
        },
        {
            "run_key": "cipro_positive_random_id",
            "campaign": "cipro",
            "oracle_id": ORACLE_ID,
            "split_id": "random_id",
            "as_of_round": 1,
            "target_lift_at_k_true": 3.0,
        },
        {
            "run_key": "cipro_null_random_id",
            "campaign": "cipro",
            "oracle_id": NULL_ORACLE_ID,
            "split_id": "random_id",
            "as_of_round": 0,
            "target_lift_at_k_true": 2.0,
        },
        {
            "run_key": "cipro_null_random_id",
            "campaign": "cipro",
            "oracle_id": NULL_ORACLE_ID,
            "split_id": "random_id",
            "as_of_round": 1,
            "target_lift_at_k_true": 0.7,
        },
    ]

    decision = _decision_from_metrics(
        run_metrics,
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
        round_metrics=round_metrics,
    )
    gate_results = gate_results_from_metrics(
        run_metrics,
        {"path_safety_pass": True, "forbidden_input_pass": True, "x_surface_pass": True},
        round_metrics=round_metrics,
    )

    assert decision == "PASS_CIPRO_RANDOM_GATE"
    assert [row for row in gate_results if row["gate"] == "H-NULL-ROUND-DYNAMICS"][0]["status"] == "attention"
    assert [row for row in gate_results if row["gate"] == "H-TRAJECTORY-SEPARATION"][0]["status"] == "pass"
