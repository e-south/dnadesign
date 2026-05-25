from __future__ import annotations

from .helpers import (
    NULL_ORACLE_ID,
    ORACLE_ID,
    pytest,
    trajectory_gate_results_from_metrics,
    trajectory_metric_payload,
)


def test_trajectory_metrics_pair_positive_and_null_auc_by_seed_campaign_split() -> None:
    round_metrics = [
        _round("cipro_positive_random_id", ORACLE_ID, 0, 1.0),
        _round("cipro_positive_random_id", ORACLE_ID, 1, 2.0),
        _round("cipro_positive_random_id", ORACLE_ID, 2, 3.0),
        _round("cipro_null_random_id", NULL_ORACLE_ID, 0, 1.0),
        _round("cipro_null_random_id", NULL_ORACLE_ID, 1, 1.0),
        _round("cipro_null_random_id", NULL_ORACLE_ID, 2, 1.0),
    ]

    payload = trajectory_metric_payload(run_metrics=[], round_metrics=round_metrics)
    pair = payload["pairs"][0]

    assert pair["positive_lift_auc"] == pytest.approx(2.0)
    assert pair["null_lift_auc"] == pytest.approx(1.0)
    assert pair["paired_auc_delta"] == pytest.approx(1.0)
    assert pair["final_positive_minus_null_lift"] == pytest.approx(2.0)
    assert pair["status"] == "pass"
    assert payload["seed_summaries"][0]["seed"] == 7
    assert payload["seed_summaries"][0]["paired_auc_delta_min"] == pytest.approx(1.0)


def test_trajectory_gate_debugs_when_null_auc_or_final_lift_wins() -> None:
    round_metrics = [
        _round("cipro_positive_random_id", ORACLE_ID, 0, 2.0),
        _round("cipro_positive_random_id", ORACLE_ID, 1, 1.0),
        _round("cipro_null_random_id", NULL_ORACLE_ID, 0, 0.5),
        _round("cipro_null_random_id", NULL_ORACLE_ID, 1, 1.5),
    ]

    gate = trajectory_gate_results_from_metrics(run_metrics=[], round_metrics=round_metrics)[0]

    assert gate["gate"] == "H-TRAJECTORY-SEPARATION"
    assert gate["status"] == "debug"
    assert "does not exceed paired null" in gate["reason"]


def _round(run_key: str, oracle_id: str, round_index: int, lift: float) -> dict[str, object]:
    return {
        "run_key": run_key,
        "campaign": "cipro",
        "split_id": "random_id",
        "seed": 7,
        "oracle_id": oracle_id,
        "as_of_round": round_index,
        "target_lift_at_k_true": lift,
    }
