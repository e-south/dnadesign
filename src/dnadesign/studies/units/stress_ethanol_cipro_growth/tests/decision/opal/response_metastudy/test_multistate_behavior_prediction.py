"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_multistate_behavior_prediction.py

Run-receipt tests for fixed behavior-shadow predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from dnadesign.opal import score_response_magnitude_feasibility
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    multistate_behavior_prediction as prediction_runtime,
)

RUN_ID = "run-fixed-1"
TARGET_MASKS = {
    "ethanol": (0.0, 1.0, 0.0, 1.0),
    "ciprofloxacin": (0.0, 0.0, 1.0, 1.0),
    "and": (0.0, 0.0, 0.0, 1.0),
}
STATE_IDS = ("00", "10", "01", "11")
CALIBRATION = {
    view_id: {
        "response_separation_min": 0.0,
        "on_magnitude_min": 0.0,
        "off_magnitude_max": 0.0,
        "response_separation_scale": 0.3 + index * 0.01,
        "on_magnitude_scale": 0.2 + index * 0.01,
        "off_magnitude_scale": 0.25 + index * 0.01,
    }
    for index, view_id in enumerate(TARGET_MASKS)
}
MODEL_PARAMS = {
    "n_estimators": 100,
    "criterion": "friedman_mse",
    "bootstrap": True,
    "oob_score": True,
    "random_state": 7,
    "n_jobs": -1,
    "emit_feature_importance": True,
}


def test_prediction_run_uses_the_receipted_scored_subset_not_the_full_candidate_table(
    tmp_path: Path,
) -> None:
    campaign_dir, candidate_records = _write_prediction_run(tmp_path, scored_count=6)

    result = prediction_runtime.load_verified_behavior_prediction_run(
        campaign_dir=campaign_dir,
        candidate_records_path=candidate_records,
        prediction_run_id=RUN_ID,
        state_ids=STATE_IDS,
        target_masks=TARGET_MASKS,
        comparator_calibration_by_view=CALIBRATION,
        comparator_objective_name="response_magnitude_feasibility_v1",
        comparator_channel="feasibility_margin",
        comparator_direction="maximize",
        model_name="random_forest",
        model_params=MODEL_PARAMS,
        raw_top_k=6,
    )

    assert len(result.predictions) == 6
    assert len(result.comparator_scores) == 18
    assert result.source["candidate_count"] == 6
    assert result.source["run_receipt_scored_count"] == 6
    assert result.source["ledger_root"] == "outputs/ledger"
    assert all(
        not Path(record["path"]).is_absolute() for records in result.source["files"].values() for record in records
    )


def test_prediction_run_rejects_scored_count_or_sequence_drift(tmp_path: Path) -> None:
    campaign_dir, candidate_records = _write_prediction_run(tmp_path, scored_count=7)

    with pytest.raises(ValueError, match="prediction rows disagree with the run receipt"):
        prediction_runtime.load_verified_behavior_prediction_run(
            campaign_dir=campaign_dir,
            candidate_records_path=candidate_records,
            prediction_run_id=RUN_ID,
            state_ids=STATE_IDS,
            target_masks=TARGET_MASKS,
            comparator_calibration_by_view=CALIBRATION,
            comparator_objective_name="response_magnitude_feasibility_v1",
            comparator_channel="feasibility_margin",
            comparator_direction="maximize",
            model_name="random_forest",
            model_params=MODEL_PARAMS,
            raw_top_k=6,
        )

    campaign_dir, candidate_records = _write_prediction_run(tmp_path / "sequence", scored_count=6)
    candidates = pd.read_parquet(candidate_records)
    candidates.loc[candidates["id"].eq("candidate-0"), "sequence"] = "T" * 60
    candidates.to_parquet(candidate_records, index=False)
    with pytest.raises(ValueError, match="sequence identity disagrees"):
        prediction_runtime.load_verified_behavior_prediction_run(
            campaign_dir=campaign_dir,
            candidate_records_path=candidate_records,
            prediction_run_id=RUN_ID,
            state_ids=STATE_IDS,
            target_masks=TARGET_MASKS,
            comparator_calibration_by_view=CALIBRATION,
            comparator_objective_name="response_magnitude_feasibility_v1",
            comparator_channel="feasibility_margin",
            comparator_direction="maximize",
            model_name="random_forest",
            model_params=MODEL_PARAMS,
            raw_top_k=6,
        )


def test_prediction_run_rejects_calibration_model_and_duplicate_channel_drift(tmp_path: Path) -> None:
    campaign_dir, candidate_records = _write_prediction_run(tmp_path, scored_count=6)
    run_path = campaign_dir / "outputs/ledger/runs.parquet/part-run.parquet"
    runs = pd.read_parquet(run_path)
    definitions = json.loads(runs.loc[0, "objective__defs_json"])
    definitions[0]["params"]["calibration"]["response_separation_scale"] = 999.0
    runs.loc[0, "objective__defs_json"] = json.dumps(definitions)
    runs.to_parquet(run_path, index=False)
    with pytest.raises(ValueError, match="objective parameters disagree"):
        _load_run(campaign_dir, candidate_records)

    campaign_dir, candidate_records = _write_prediction_run(tmp_path / "model", scored_count=6)
    run_path = campaign_dir / "outputs/ledger/runs.parquet/part-run.parquet"
    runs = pd.read_parquet(run_path)
    model_params = dict(runs.loc[0, "model__params"])
    model_params["n_estimators"] = 99
    runs.at[0, "model__params"] = model_params
    runs.to_parquet(run_path, index=False)
    with pytest.raises(ValueError, match="model parameters disagree"):
        _load_run(campaign_dir, candidate_records)

    campaign_dir, candidate_records = _write_prediction_run(tmp_path / "channels", scored_count=6)
    prediction_path = campaign_dir / "outputs/ledger/predictions/part-run.parquet"
    predictions = pd.read_parquet(prediction_path)
    channels = list(predictions.loc[0, "pred__score_channels"])
    channels.append(dict(channels[0]))
    predictions.at[0, "pred__score_channels"] = channels
    predictions.to_parquet(prediction_path, index=False)
    with pytest.raises(ValueError, match="unique per candidate"):
        _load_run(campaign_dir, candidate_records)


def test_prediction_run_rejects_comparator_channel_value_drift(tmp_path: Path) -> None:
    campaign_dir, candidate_records = _write_prediction_run(tmp_path, scored_count=6)
    prediction_path = campaign_dir / "outputs/ledger/predictions/part-run.parquet"
    predictions = pd.read_parquet(prediction_path)
    channels = [dict(item) for item in predictions.loc[0, "pred__score_channels"]]
    channels[0]["value"] = float(channels[0]["value"]) + 1.0
    predictions.at[0, "pred__score_channels"] = channels
    predictions.to_parquet(prediction_path, index=False)

    with pytest.raises(ValueError, match="does not replay from the prediction vector"):
        _load_run(campaign_dir, candidate_records)


def _load_run(campaign_dir: Path, candidate_records: Path):
    return prediction_runtime.load_verified_behavior_prediction_run(
        campaign_dir=campaign_dir,
        candidate_records_path=candidate_records,
        prediction_run_id=RUN_ID,
        state_ids=STATE_IDS,
        target_masks=TARGET_MASKS,
        comparator_calibration_by_view=CALIBRATION,
        comparator_objective_name="response_magnitude_feasibility_v1",
        comparator_channel="feasibility_margin",
        comparator_direction="maximize",
        model_name="random_forest",
        model_params=MODEL_PARAMS,
        raw_top_k=6,
    )


def _write_prediction_run(tmp_path: Path, *, scored_count: int) -> tuple[Path, Path]:
    campaign_dir = tmp_path / "campaign"
    prediction_dir = campaign_dir / "outputs/ledger/predictions"
    run_dir = campaign_dir / "outputs/ledger/runs.parquet"
    prediction_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    candidates = pd.DataFrame(
        {
            "id": [f"candidate-{index}" for index in range(7)],
            "sequence": ["ACGT" * 15 for _ in range(7)],
        }
    )
    candidate_records = tmp_path / "records.parquet"
    candidates.to_parquet(candidate_records, index=False)
    prediction_rows = []
    for index in range(6):
        values = [float(index + offset) for offset in range(8)]
        score_channels: list[dict[str, object]] = []
        for view_id, target_mask in TARGET_MASKS.items():
            score = score_response_magnitude_feasibility(
                pd.DataFrame([values]).to_numpy(dtype=float),
                target_mask=target_mask,
                calibration=CALIBRATION[view_id],
            )
            score_channels.extend(
                [
                    {"name": f"{view_id}/feasibility_margin", "value": float(score.feasibility_margin[0])},
                    {
                        "name": f"{view_id}/response_separation",
                        "value": float(score.components.response_separation[0]),
                    },
                    {
                        "name": f"{view_id}/on_magnitude_floor",
                        "value": float(score.components.on_magnitude_floor[0]),
                    },
                    {
                        "name": f"{view_id}/off_magnitude_ceiling",
                        "value": float(score.components.off_magnitude_ceiling[0]),
                    },
                ]
            )
        prediction_rows.append(
            {
                "event": "run_pred",
                "run_id": RUN_ID,
                "as_of_round": 0,
                "id": f"candidate-{index}",
                "sequence": "ACGT" * 15,
                "pred__y_dim": 8,
                "pred__y_hat_model": values,
                "pred__score_channels": score_channels,
            }
        )
    pd.DataFrame.from_records(prediction_rows).to_parquet(prediction_dir / "part-run.parquet", index=False)
    objective_defs = [
        {
            "selection_view_id": view_id,
            "objective_name": "response_magnitude_feasibility_v1",
            "params": {
                "state_ids": list(STATE_IDS),
                "target_mask": list(mask),
                "calibration": CALIBRATION[view_id],
            },
            "score_channels": [
                f"{view_id}/feasibility_margin",
                f"{view_id}/response_separation",
                f"{view_id}/on_magnitude_floor",
                f"{view_id}/off_magnitude_ceiling",
            ],
        }
        for view_id, mask in TARGET_MASKS.items()
    ]
    selection_defs = [
        {
            "selection_view_id": view_id,
            "objective_name": "response_magnitude_feasibility_v1",
            "objective_params": {
                "state_ids": list(STATE_IDS),
                "target_mask": list(TARGET_MASKS[view_id]),
                "calibration": CALIBRATION[view_id],
            },
            "selection_name": "top_n",
            "selection_params": {
                "top_k": 6,
                "score_ref": "feasibility_margin",
                "tie_handling": "ordinal",
                "objective_mode": "maximize",
                "exclude_already_labeled": True,
                "require_exact_top_k": True,
            },
            "score_ref": f"{view_id}/feasibility_margin",
            "objective_mode": "maximize",
            "tie_handling": "ordinal",
            "top_k": 6,
            "objective_summary_stats": {"candidate_count": 6},
        }
        for view_id in TARGET_MASKS
    ]
    pd.DataFrame.from_records(
        [
            {
                "event": "run_meta",
                "run_id": RUN_ID,
                "as_of_round": 0,
                "model__name": "random_forest",
                "model__params": {
                    **RandomForestRegressor(
                        **{key: value for key, value in MODEL_PARAMS.items() if key != "emit_feature_importance"}
                    ).get_params(deep=False),
                    "emit_feature_importance": True,
                },
                "training__y_ops": None,
                "x_transform__name": "identity",
                "x_transform__params": None,
                "y_ingest__name": "vector_from_table_v1",
                "y_ingest__params": {
                    "id_column": "id",
                    "sequence_column": "sequence",
                    "value_columns": ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"],
                },
                "stats__n_train": 27,
                "stats__n_scored": scored_count,
                "objective__defs_json": json.dumps(objective_defs),
                "selection_views__defs_json": json.dumps(selection_defs),
            }
        ]
    ).to_parquet(run_dir / "part-run.parquet", index=False)
    return campaign_dir, candidate_records
