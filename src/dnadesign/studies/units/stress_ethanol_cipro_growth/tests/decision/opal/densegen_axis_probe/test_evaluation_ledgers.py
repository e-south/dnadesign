"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_evaluation_ledgers.py

Regression tests for evaluation ledgers studies units stress ethanol cipro growth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .helpers import (
    ORACLE_ID,
    Path,
    RunSpec,
    _evaluate_run,
    _evaluate_run_rounds,
    _write_probe_prediction_campaign,
    pd,
    pytest,
)


def test_evaluate_run_rejects_partial_prediction_ledgers(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-1"],
                "pred__y_hat_model": [[0.0, 0.0, 1.0, 1.0]],
                "pred__score_selected": [1.0],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-1", "eval-2"],
            "axis_class": ["background_only", "cipro_only", "cipro_only"],
            "quality_flag": ["ok", "ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
        selection_k=6,
    )

    with pytest.raises(RuntimeError, match="missing eval id"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"]},
        )


def test_evaluate_run_respects_split_eval_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    eval_ids = [f"eval-{idx}" for idx in range(1, 7)]
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": eval_ids,
                "pred__y_hat_model": [[0.0, 0.0, 1.0, 1.0]] * len(eval_ids),
                "pred__score_selected": [1.0 - (idx * 0.01) for idx in range(len(eval_ids))],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", *eval_ids, "eval-outside-cap"],
            "axis_class": ["background_only", *(["cipro_only"] * len(eval_ids)), "cipro_only"],
            "quality_flag": ["ok"] * (len(eval_ids) + 2),
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
        selection_k=6,
    )

    metrics = _evaluate_run(
        run=run,
        positive_labels=labels,
        run_labels=labels,
        split_metadata={
            "train_ids": ["train-1"],
            "eval_ids": eval_ids,
        },
    )

    assert metrics["eval_count"] == 6
    assert "candidate_cap_per_split" not in metrics
    assert "eval_full_count" not in metrics
    assert metrics["selected_count_in_eval"] == 6
    assert metrics["selected_ids"] == eval_ids


def test_evaluate_run_scores_tf_count_active_target_by_mean_lift(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-high", "eval-mid", "eval-zero"],
                "pred__y_hat_model": [[4.0, 0.0, 4.0], [2.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
                "pred__score_selected": [0.9, 0.8, 0.1],
                "sel__is_selected": [True, True, False],
                "sel__rank_competition": [1, 2, 3],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-high", "eval-mid", "eval-zero"],
            "quality_flag": ["ok"] * 4,
            "tf_family__lexA__count": [0, 4, 2, 0],
            "tf_family__cpxR__count": [0, 0, 0, 0],
            "tf_family__baeR__count": [0, 0, 0, 0],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="tf_count_cipro_positive_random_id",
        target_class="tf_count__lexA",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
        selection_k=2,
        label_family_id="tf_family_count",
        target_channel="tf_count__lexA",
    )

    metrics = _evaluate_run(
        run=run,
        positive_labels=labels,
        run_labels=labels,
        split_metadata={"train_ids": ["train-1"], "eval_ids": ["eval-high", "eval-mid", "eval-zero"]},
    )

    assert metrics["target_channel"] == "tf_count__lexA"
    assert metrics["selected_target_count_label_true"] == "2/2"
    assert metrics["target_mean_eval_true"] == 2.0
    assert metrics["selected_target_mean_true"] == 3.0
    assert metrics["target_lift_at_k_true"] == 1.5


def test_evaluate_run_rounds_tracks_retroactive_performance_by_round(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "run_id": ["run-r0", "run-r0", "run-r1"],
                "as_of_round": [0, 0, 1],
                "id": ["eval-a", "eval-b", "eval-b"],
                "pred__y_hat_model": [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                "pred__score_selected": [0.9, 0.1, 0.8],
                "sel__is_selected": [True, False, True],
                "sel__rank_competition": [1, 2, 1],
            }
        ),
        runs=[("run-r0", 0), ("run-r1", 1)],
    )
    sidecar_path = workdir / "observed_labels.parquet"
    pd.DataFrame(
        {
            "id": ["train-1", "eval-a"],
            "observed_round": [0, 1],
        }
    ).to_parquet(sidecar_path, index=False)
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-a", "eval-b"],
            "axis_class": ["background_only", "cipro_only", "background_only"],
            "quality_flag": ["ok", "ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=sidecar_path,
        selection_k=1,
    )

    rows = _evaluate_run_rounds(
        run=run,
        positive_labels=labels,
        run_labels=labels,
        split_metadata={"train_ids": ["train-1"], "eval_ids": ["eval-a", "eval-b"]},
    )

    assert [row["as_of_round"] for row in rows] == [0, 1]
    assert [row["train_count"] for row in rows] == [1, 2]
    assert [row["eval_count"] for row in rows] == [2, 1]
    assert [row["selected_target_count_label_true"] for row in rows] == ["1/1", "0/1"]


def test_evaluate_run_requires_prediction_schema(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame({"id": ["eval-1"], "pred__score_selected": [1.0]}),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-1"],
            "axis_class": ["background_only", "cipro_only"],
            "quality_flag": ["ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
        selection_k=6,
    )

    with pytest.raises(RuntimeError, match="missing column"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"]},
        )


def test_evaluate_run_rejects_duplicate_prediction_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-1", "eval-1"],
                "pred__y_hat_model": [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                ],
                "pred__score_selected": [1.0, 0.5],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-1"],
            "axis_class": ["background_only", "cipro_only"],
            "quality_flag": ["ok", "ok"],
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
        selection_k=6,
    )

    with pytest.raises(RuntimeError, match="duplicate prediction id"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"]},
        )


def test_evaluate_run_scores_actual_selected_rows_not_highest_unselected_score(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    selected_ids = [f"eval-selected-{idx}" for idx in range(6)]
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-high-unselected", *selected_ids],
                "pred__y_hat_model": [
                    [0.0, 0.0, 1.0, 1.0],
                    *([[0.0, 0.0, 1.0, 1.0]] * len(selected_ids)),
                ],
                "pred__score_selected": [0.99, *([0.5] * len(selected_ids))],
                "sel__is_selected": [False, *([True] * len(selected_ids))],
                "sel__rank_competition": [7, *range(1, 7)],
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", "eval-high-unselected", *selected_ids],
            "axis_class": ["background_only", "background_only", *(["cipro_only"] * len(selected_ids))],
            "quality_flag": ["ok"] * (len(selected_ids) + 2),
        }
    )
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=workdir,
        config_path=config_path,
        label_input_path=workdir / "inputs" / "r0" / "labels-b0.parquet",
        sidecar_path=workdir / "sidecar.parquet",
        selection_k=6,
    )

    metrics = _evaluate_run(
        run=run,
        positive_labels=labels,
        run_labels=labels,
        split_metadata={"train_ids": ["train-1"]},
    )

    assert metrics["selected_ids"] == selected_ids
    assert metrics["selected_target_precision_at_k_true"] == 1.0
