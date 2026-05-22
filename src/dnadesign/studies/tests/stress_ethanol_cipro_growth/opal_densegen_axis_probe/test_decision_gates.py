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


def test_decision_stops_when_null_enriches_true_target_class() -> None:
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

    assert decision == "STOP"


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

    assert any(reason["reason"] == "null lift exceeds 1.25" for reason in reasons)
    assert any(reason["reason"] == "positive lift does not exceed null lift" for reason in reasons)
    pair_gate = [row for row in gate_results if row.get("campaign") == "ethanol"][0]
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
