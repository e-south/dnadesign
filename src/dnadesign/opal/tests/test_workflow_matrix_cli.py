"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_workflow_matrix_cli.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.opal.src.cli.commands.init import cmd_init
from dnadesign.opal.src.cli.commands.run import cmd_run
from dnadesign.opal.src.storage.ledger import LedgerReader
from dnadesign.opal.src.storage.workspace import CampaignWorkspace

from ._cli_helpers import write_campaign_yaml


def _write_records_vec8(records_path: Path, *, slug: str = "demo") -> None:
    label_hist_col = f"opal__{slug}__label_hist"
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "sequence": ["AAA", "AAT", "ATT"],
            "bio_type": ["dna", "dna", "dna"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.2, 0.4], [0.3, 0.1]],
            "Y": [None, None, None],
            label_hist_col: [
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "ts": "2024-01-01T00:00:00Z",
                        "src": "ingest_y",
                        "y_obs": {
                            "value": [0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 2.0],
                            "dtype": "vector",
                            "schema": {"length": 8},
                        },
                    }
                ],
                [],
                [],
            ],
        }
    )
    df.to_parquet(records_path, index=False)


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

    assert str(run_row["selection__name"]) == selection_name
    assert str(run_row["selection__score_ref"]) == "sfxi_v1/sfxi"
    if selection_name == "expected_improvement":
        assert str(run_row["selection__uncertainty_ref"]) == "sfxi_v1/sfxi"
        assert pd.notna(run_row["stats__unc_mean_sd_targets"])

    pred_df = reader.read_predictions(
        columns=["id", "pred__score_selected", "pred__score_ref", "sel__is_selected"],
        round_selector=0,
        run_id=run_id,
    )
    assert not pred_df.empty
    assert pred_df["pred__score_selected"].notna().all()
    assert pred_df["pred__score_ref"].astype(str).eq("sfxi_v1/sfxi").all()
