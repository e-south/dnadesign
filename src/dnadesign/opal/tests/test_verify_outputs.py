# ABOUTME: Verifies selection outputs against ledger predictions and run metadata.
# ABOUTME: Covers integration paths for verify-outputs and end-to-end run flows.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_verify_outputs.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.opal.src.cli.commands.init import cmd_init
from dnadesign.opal.src.cli.commands.run import cmd_run
from dnadesign.opal.src.cli.commands.verify_outputs import verify_outputs
from dnadesign.opal.src.reporting.verify_outputs import compare_selection_to_ledger, read_selection_table
from dnadesign.opal.src.storage.artifacts import write_selection_csv
from dnadesign.opal.src.storage.ledger import LedgerReader, LedgerWriter
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import SelectionEmit, build_run_meta_event, build_run_pred_events


def test_compare_selection_to_ledger_matches() -> None:
    selection_df = pd.DataFrame({"id": ["a", "b"], "pred__y_obj_scalar": [1.0, 2.0]})
    ledger_df = pd.DataFrame({"id": ["a", "b"], "pred__y_obj_scalar": [1.0, 2.0]})
    summary, mismatches = compare_selection_to_ledger(selection_df, ledger_df, eps=1e-9)
    assert summary["mismatch_count"] == 0
    assert mismatches.empty


def test_compare_selection_to_ledger_mismatch() -> None:
    selection_df = pd.DataFrame({"id": ["a", "b"], "selection_score": [1.0, 2.0]})
    ledger_df = pd.DataFrame({"id": ["a", "b"], "pred__y_obj_scalar": [1.0, 2.5]})
    summary, mismatches = compare_selection_to_ledger(selection_df, ledger_df, eps=1e-9)
    assert summary["mismatch_count"] == 1
    assert mismatches.shape[0] == 1


def test_verify_outputs_integration(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir(parents=True)
    ws = CampaignWorkspace(config_path=tmp_path / "campaign.yaml", workdir=workdir)

    run_id = "run-1"
    as_of_round = 1
    ids = ["a", "b"]
    sequences = ["AAA", "BBB"]
    y_hat = np.array([[0.1], [0.2]])
    y_obj = np.array([0.5, 0.6])
    sel_emit = SelectionEmit(ranks_competition=np.array([1, 2]), selected_bool=np.array([True, False]))
    run_pred = build_run_pred_events(
        run_id=run_id,
        as_of_round=as_of_round,
        ids=ids,
        sequences=sequences,
        y_hat_model=y_hat,
        y_obj_scalar=y_obj,
        y_dim=1,
        y_hat_model_sd=None,
        y_obj_scalar_sd=None,
        obj_diagnostics={},
        sel_emit=sel_emit,
    )

    rdir = ws.round_dir(as_of_round)
    rdir.mkdir(parents=True, exist_ok=True)
    selection_df = pd.DataFrame(
        {
            "id": ids,
            "selection_score": y_obj,
            "pred__y_obj_scalar": y_obj,
        }
    )
    selection_path = rdir / "selection_top_k.csv"
    write_selection_csv(selection_path, selection_df)

    run_meta = build_run_meta_event(
        run_id=run_id,
        as_of_round=as_of_round,
        model_name="dummy",
        model_params={},
        y_ops=[],
        x_transform_name="none",
        x_transform_params={},
        y_ingest_transform_name="none",
        y_ingest_transform_params={},
        objective_name="dummy_obj",
        objective_params={},
        selection_name="top_k",
        selection_params={"top_k": 1, "objective_mode": "maximize"},
        selection_score_field="pred__y_obj_scalar",
        selection_objective_mode="maximize",
        sel_tie_handling="competition_rank",
        stats_n_train=0,
        stats_n_scored=len(ids),
        unc_mean_sd=None,
        pred_rows_df=run_pred,
        artifact_paths_and_hashes={"selection_top_k.csv": ("sha", str(selection_path))},
        objective_summary_stats=None,
    )

    ledger = LedgerWriter(ws)
    ledger.append_run_pred(run_pred)
    ledger.append_run_meta(run_meta)

    reader = LedgerReader(ws)
    ledger_df = reader.read_predictions(
        columns=["id", "pred__y_obj_scalar"],
        round_selector=as_of_round,
        run_id=run_id,
    )
    sel_loaded = read_selection_table(selection_path)
    summary, mismatches = compare_selection_to_ledger(sel_loaded, ledger_df, eps=1e-9)
    assert summary["mismatch_count"] == 0
    assert mismatches.empty


def test_end_to_end_run_and_verify_outputs(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True)
    records_path = workdir / "records.parquet"

    label_hist_col = "opal__demo__label_hist"
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "bio_type": ["dna"] * 4,
            "sequence": ["AAAA", "AAAT", "AATT", "ATTT"],
            "alphabet": ["dna_4"] * 4,
            "x_vec": [[0.1, 0.2], [0.2, 0.4], [0.3, 0.1], [0.4, 0.3]],
            "y_scalar": [None, None, None, None],
            label_hist_col: [
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "ts": "2024-01-01T00:00:00Z",
                        "src": "ingest_y",
                        "y_obs": {"value": [0.1], "dtype": "vector"},
                    }
                ],
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "ts": "2024-01-01T00:00:00Z",
                        "src": "ingest_y",
                        "y_obs": {"value": [0.2], "dtype": "vector"},
                    }
                ],
                [],
                [],
            ],
        }
    )
    df.to_parquet(records_path)

    campaign_yaml = workdir / "campaign.yaml"
    campaign_yaml.write_text(
        "\n".join(
            [
                "campaign:",
                '  name: "Demo"',
                '  slug: "demo"',
                '  workdir: "."',
                "",
                "data:",
                "  location: { kind: local, path: records.parquet }",
                '  x_column_name: "x_vec"',
                '  y_column_name: "y_scalar"',
                "  y_expected_length: 1",
                "",
                "transforms_x:",
                '  name: "identity"',
                "  params: {}",
                "",
                "transforms_y:",
                '  name: "scalar_from_table_v1"',
                "  params: {}",
                "",
                "model:",
                '  name: "random_forest"',
                "  params: { n_estimators: 10, random_state: 7 }",
                "",
                "objective:",
                '  name: "scalar_identity_v1"',
                "  params: {}",
                "",
                "selection:",
                '  name: "top_n"',
                "  params: { top_k: 2, objective_mode: maximize }",
                "",
                "training:",
                "  policy:",
                "    cumulative_training: true",
                "    label_cross_round_deduplication_policy: latest_only",
                "    allow_resuggesting_candidates_until_labeled: true",
            ]
        )
        + "\n"
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
    verify_outputs(
        config=campaign_yaml,
        round="latest",
        run_id=None,
        selection_path=None,
        eps=1e-6,
        json=True,
    )

    outputs_dir = workdir / "outputs"
    assert (outputs_dir / "ledger" / "runs.parquet").is_dir()
    assert (outputs_dir / "ledger" / "labels.parquet").is_file() is False
    assert (outputs_dir / "ledger" / "predictions").exists()
