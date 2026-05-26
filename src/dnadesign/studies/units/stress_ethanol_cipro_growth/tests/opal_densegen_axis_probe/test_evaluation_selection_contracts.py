from __future__ import annotations

from .helpers import (
    ORACLE_ID,
    Path,
    RunSpec,
    _evaluate_run,
    _write_probe_prediction_campaign,
    pd,
    pytest,
)


def test_evaluate_run_rejects_less_than_six_evaluable_selected_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-1"],
                "pred__y_hat_model": [[0.0, 0.0, 1.0, 1.0]],
                "pred__score_selected": [1.0],
                "sel__is_selected": [True],
                "sel__rank_competition": [1],
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

    with pytest.raises(RuntimeError, match="expected 6 evaluable selected"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"], "eval_ids": ["eval-1"]},
        )


def test_evaluate_run_rejects_more_than_six_evaluable_selected_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    eval_ids = [f"eval-{idx}" for idx in range(7)]
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": eval_ids,
                "pred__y_hat_model": [[0.0, 0.0, 1.0, 1.0]] * len(eval_ids),
                "pred__score_selected": [1.0] * len(eval_ids),
                "sel__is_selected": [True] * len(eval_ids),
                "sel__rank_competition": [1] * len(eval_ids),
            }
        ),
    )
    labels = pd.DataFrame(
        {
            "id": ["train-1", *eval_ids],
            "axis_class": ["background_only", *(["cipro_only"] * len(eval_ids))],
            "quality_flag": ["ok"] * (len(eval_ids) + 1),
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

    with pytest.raises(RuntimeError, match="expected 6 evaluable selected"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"], "eval_ids": eval_ids},
        )


def test_evaluate_run_rejects_string_selection_flags(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    config_path = _write_probe_prediction_campaign(
        workdir,
        pd.DataFrame(
            {
                "id": ["eval-1"],
                "pred__y_hat_model": [[0.0, 0.0, 1.0, 1.0]],
                "pred__score_selected": [0.99],
                "sel__is_selected": ["False"],
                "sel__rank_competition": [1],
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
    )

    with pytest.raises(RuntimeError, match="sel__is_selected must be boolean"):
        _evaluate_run(
            run=run,
            positive_labels=labels,
            run_labels=labels,
            split_metadata={"train_ids": ["train-1"]},
        )
