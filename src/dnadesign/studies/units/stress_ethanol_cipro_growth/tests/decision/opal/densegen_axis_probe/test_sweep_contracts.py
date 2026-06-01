from __future__ import annotations

from .helpers import Path, build_plan, build_sweep_execution_contract, enforce_sweep_apply_contract, pytest


def test_many_campaign_scored_apply_requires_explicit_score_batch_size(tmp_path: Path) -> None:
    plan = build_plan(
        run_root=tmp_path / "probe",
        initial_label_count=12,
        selection_k=12,
        seed=7,
        gate="all",
        splits=("random_id", "leave_sigma35_variant"),
        rounds=12,
        stop_after="status",
        score_batch_size=None,
    )

    contract = build_sweep_execution_contract(plan)

    assert contract["status"] == "blocked"
    assert contract["planned_campaign_count"] == 24
    assert contract["planned_round_count"] == 288
    assert contract["expected_round_metric_rows"] == 288
    assert contract["expected_run_metric_rows"] == 24
    assert contract["expected_final_round"] == 11
    assert contract["expected_final_labeled_ids_per_campaign"] == 144
    assert contract["expected_selection_ids_per_campaign"] == 144
    assert contract["suite_campaign_count_if_repeated_for_all_suite_seeds"] == 72
    assert contract["blocking_problems"] == ["score_batch_size_required_for_many_campaign_scored_apply"]
    with pytest.raises(ValueError, match="score_batch_size_required"):
        enforce_sweep_apply_contract(plan)


def test_many_campaign_scored_apply_contract_accepts_explicit_score_batch_size(tmp_path: Path) -> None:
    plan = build_plan(
        run_root=tmp_path / "probe",
        initial_label_count=12,
        selection_k=12,
        seed=7,
        gate="all",
        splits=("random_id", "leave_sigma35_variant"),
        rounds=12,
        stop_after="status",
        score_batch_size=512,
    )

    contract = build_sweep_execution_contract(plan)

    assert contract["status"] == "ok"
    assert contract["score_batch_size"] == 512
    assert contract["opal_command_counts"]["run"] == 288
    enforce_sweep_apply_contract(plan)
