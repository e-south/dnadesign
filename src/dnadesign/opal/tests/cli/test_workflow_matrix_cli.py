"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_workflow_matrix_cli.py

Regression tests for workflow matrix CLI OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.opal.src.analysis.ledger import read_selection_view_predictions
from dnadesign.opal.src.cli.commands.init import cmd_init
from dnadesign.opal.src.cli.commands.run import cmd_run
from dnadesign.opal.src.plots.response_magnitude_feasibility_data import parse_response_magnitude_feasibility_channels
from dnadesign.opal.src.storage.ledger import LedgerReader
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml


def _write_records_vec8(records_path: Path, *, slug: str = "demo") -> None:
    label_hist_col = f"opal__{slug}__label_hist"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["a", "b", "c"], type=pa.string()),
                "sequence": pa.array(["AAA", "AAT", "ATT"], type=pa.string()),
                "bio_type": pa.array(["dna", "dna", "dna"], type=pa.string()),
                "alphabet": pa.array(["dna_4", "dna_4", "dna_4"], type=pa.string()),
                "X": pa.FixedSizeListArray.from_arrays(
                    pa.array([0.1, 0.2, 0.2, 0.4, 0.3, 0.1], type=pa.float32()),
                    2,
                ),
                "Y": pa.nulls(3, type=pa.null()),
                label_hist_col: pa.array(
                    [
                        [
                            {
                                "kind": "label",
                                "observed_round": 0,
                                "ts": "2024-01-01T00:00:00Z",
                                "src": "ingest_y",
                                "y_obs": {
                                    "value": [0.8, 0.2, 0.2, 0.2, -1.0, 0.3, 0.1, 0.2],
                                    "dtype": "vector",
                                    "schema": {"length": 8},
                                },
                            }
                        ],
                        [
                            {
                                "kind": "label",
                                "observed_round": 0,
                                "ts": "2024-01-01T00:00:00Z",
                                "src": "ingest_y",
                                "y_obs": {
                                    "value": [0.6, 0.3, 0.2, 0.1, -0.5, 0.2, 0.1, 0.2],
                                    "dtype": "vector",
                                    "schema": {"length": 8},
                                },
                            }
                        ],
                        [],
                    ]
                ),
            }
        ),
        records_path,
    )


@pytest.mark.parametrize(
    "case_name,model_name,model_params,selection_name,selection_params",
    [
        (
            "rf_sfxi_topn",
            "random_forest",
            {"n_estimators": 8, "random_state": 7, "oob_score": False},
            "top_n",
            {"top_k": 1, "score_ref": "sfxi_v1/sfxi", "objective_mode": "maximize"},
        ),
        (
            "gp_sfxi_topn",
            "gaussian_process",
            {
                "alpha": 1.0e-6,
                "normalize_y": True,
                "kernel": {"name": "matern", "length_scale": 0.5, "nu": 1.5, "with_white_noise": True},
            },
            "top_n",
            {"top_k": 1, "score_ref": "sfxi_v1/sfxi", "objective_mode": "maximize"},
        ),
        (
            "gp_sfxi_ei",
            "gaussian_process",
            {
                "alpha": 1.0e-6,
                "normalize_y": True,
                "kernel": {"name": "matern", "length_scale": 0.5, "nu": 1.5, "with_white_noise": True},
            },
            "expected_improvement",
            {
                "top_k": 1,
                "score_ref": "sfxi_v1/sfxi",
                "uncertainty_ref": "sfxi_v1/sfxi",
                "objective_mode": "maximize",
                "alpha": 1.0,
                "beta": 1.0,
            },
        ),
    ],
)
def test_cli_workflow_matrix(
    tmp_path: Path,
    case_name: str,
    model_name: str,
    model_params: dict,
    selection_name: str,
    selection_params: dict,
) -> None:
    workdir = tmp_path / case_name
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = workdir / "records.parquet"
    _write_records_vec8(records_path)

    campaign_yaml = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign_yaml,
        workdir=workdir,
        records_path=records_path,
        slug="demo",
        transforms_y_name="sfxi_vec8_from_table_v1",
        transforms_y_params={},
        objective_name="sfxi_v1",
        objective_params={"setpoint_vector": [1.0, 0.0, 0.0, 0.0], "scaling": {"min_n": 1}},
        y_expected_length=8,
        model_name=model_name,
        model_params=model_params,
        selection_name=selection_name,
        selection_params=selection_params,
    )

    cmd_init(config=campaign_yaml, json=True)
    cmd_run(
        config=campaign_yaml,
        round=0,
        k=None,
        resume=False,
        score_batch_size=10,
        verbose=False,
        json=True,
    )

    ws = CampaignWorkspace(config_path=campaign_yaml, workdir=workdir)
    reader = LedgerReader(ws)

    runs_df = reader.read_runs()
    assert not runs_df.empty
    run_row = runs_df.sort_values(["as_of_round", "run_id"]).tail(1).iloc[0]
    run_id = str(run_row["run_id"])

    selection_def = json.loads(str(run_row["selection_views__defs_json"]))[0]
    assert selection_def["selection_name"] == selection_name
    assert selection_def["score_ref"] == "primary/sfxi"
    if selection_name == "expected_improvement":
        assert selection_def["uncertainty_ref"] == "primary/sfxi"

    pred_df = read_selection_view_predictions(
        reader.paths.predictions_dir,
        selection_view_id="primary",
        columns=["id", "view__selection_score", "view__score_ref", "view__is_selected", "view__uncertainty"],
        round_selector=0,
        run_id=run_id,
        runs_df=pl.from_pandas(runs_df),
    ).to_pandas()
    assert not pred_df.empty
    assert pred_df["view__selection_score"].notna().all()
    assert pred_df["view__score_ref"].astype(str).eq("primary/sfxi").all()
    if selection_name == "expected_improvement":
        assert pred_df["view__uncertainty"].notna().all()


def test_cli_response_magnitude_feasibility_topn_happy_path(tmp_path: Path) -> None:
    workdir = tmp_path / "rf_response_separation_topn"
    workdir.mkdir(parents=True, exist_ok=True)
    records_path = workdir / "records.parquet"
    _write_records_vec8(records_path)

    campaign_yaml = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign_yaml,
        workdir=workdir,
        records_path=records_path,
        slug="demo",
        transforms_y_name="vector_from_table_v1",
        transforms_y_params={"value_columns": ["r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11"]},
        objective_name="response_magnitude_feasibility_v1",
        objective_params={
            "state_ids": ["00", "10", "01", "11"],
            "target_mask": [1, 0, 0, 0],
            "calibration": {
                "response_separation_min": 0.0,
                "on_magnitude_min": -1.0,
                "off_magnitude_max": 0.5,
                "response_separation_scale": 1.0,
                "on_magnitude_scale": 1.0,
                "off_magnitude_scale": 1.0,
            },
        },
        y_expected_length=8,
        model_name="random_forest",
        model_params={"n_estimators": 8, "random_state": 7, "oob_score": False},
        selection_name="top_n",
        selection_params={
            "top_k": 1,
            "score_ref": "response_magnitude_feasibility_v1/feasibility_margin",
            "objective_mode": "maximize",
        },
    )

    cmd_init(config=campaign_yaml, json=True)
    cmd_run(
        config=campaign_yaml,
        round=0,
        k=None,
        resume=False,
        score_batch_size=10,
        verbose=False,
        json=True,
    )

    reader = LedgerReader(CampaignWorkspace(config_path=campaign_yaml, workdir=workdir))
    run_row = reader.read_runs().sort_values(["as_of_round", "run_id"]).tail(1).iloc[0]
    run_id = str(run_row["run_id"])
    selection_def = json.loads(str(run_row["selection_views__defs_json"]))[0]
    assert selection_def["score_ref"] == "primary/feasibility_margin"

    pred_df = read_selection_view_predictions(
        reader.paths.predictions_dir,
        selection_view_id="primary",
        columns=["id", "view__selection_score", "view__score_ref", "pred__score_channels", "view__is_selected"],
        round_selector=0,
        run_id=run_id,
        runs_df=pl.from_pandas(reader.read_runs()),
    ).to_pandas()
    assert len(pred_df) == 1
    assert pred_df["view__selection_score"].notna().all()
    assert pred_df["view__score_ref"].eq("primary/feasibility_margin").all()
    assert all(
        parse_response_magnitude_feasibility_channels(value, selection_view_id="primary")
        for value in pred_df["pred__score_channels"]
    )
    assert int(pred_df["view__is_selected"].sum()) == 1
