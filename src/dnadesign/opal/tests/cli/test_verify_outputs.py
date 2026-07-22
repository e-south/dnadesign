"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_verify_outputs.py

Selection-view verification against shared OPAL prediction ledgers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from click import unstyle
from typer.testing import CliRunner

from dnadesign.opal.src.analysis.ledger import read_selection_view_predictions
from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.reporting.verify_outputs import compare_selection_to_ledger
from dnadesign.opal.src.storage.artifacts import write_selection_parquet
from dnadesign.opal.src.storage.ledger import LedgerWriter
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import (
    SelectionViewEmit,
    build_run_meta_event,
    build_run_pred_events,
)
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records


def _write_cli_fixture(tmp_path: Path, *, selection_score: float) -> tuple[Path, str]:
    workdir = tmp_path / "workdir"
    workdir.mkdir(parents=True)
    records_path = tmp_path / "records.parquet"
    campaign_path = tmp_path / "campaign.yaml"
    write_records(records_path)
    write_campaign_yaml(campaign_path, workdir=workdir, records_path=records_path)

    run_id = "run-1"
    selection_path = workdir / "outputs" / "rounds" / "round_1" / "selection" / "selections.parquet"
    write_selection_parquet(
        selection_path,
        pd.DataFrame(
            {
                "selection_view_id": ["primary"],
                "id": ["a"],
                "score": [0.1],
                "selection_score": [selection_score],
            }
        ),
    )
    write_ledger(
        workdir,
        run_id=run_id,
        round_index=1,
        artifact_paths_and_hashes={
            "selection/selections.parquet": ("sha", str(selection_path)),
        },
    )
    return campaign_path, run_id


def test_compare_selection_to_view_ledger_matches() -> None:
    selection_df = pd.DataFrame({"id": ["a", "b"], "selection_score": [1.0, 2.0]})
    ledger_df = pd.DataFrame({"id": ["a", "b"], "view__selection_score": [1.0, 2.0]})

    summary, mismatches = compare_selection_to_ledger(selection_df, ledger_df, eps=1e-9)

    assert summary["mismatch_count"] == 0
    assert mismatches.empty


def test_compare_selection_to_view_ledger_rejects_unknown_ids() -> None:
    selection_df = pd.DataFrame({"id": ["a", "missing"], "selection_score": [1.0, 2.0]})
    ledger_df = pd.DataFrame({"id": ["a", "b"], "view__selection_score": [1.0, 2.0]})

    with pytest.raises(OpalError, match="selected_ids_outside_eval"):
        compare_selection_to_ledger(selection_df, ledger_df, eps=1e-9)


def test_verify_outputs_integration_uses_named_view(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir(parents=True)
    ws = CampaignWorkspace(config_path=tmp_path / "campaign.yaml", workdir=workdir)
    run_id = "run-1"
    view = SelectionViewEmit(
        selection_view_id="ethanol",
        objective_name="scalar_identity_v1",
        selection_name="top_n",
        score=np.asarray([0.5, 0.6]),
        score_ref="ethanol/scalar",
        selection_score=np.asarray([0.5, 0.6]),
        ranks_competition=np.asarray([2, 1]),
        selected_bool=np.asarray([False, True]),
        top_k=1,
        diagnostics={},
    )
    predictions = build_run_pred_events(
        run_id=run_id,
        as_of_round=1,
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=np.asarray([[0.1], [0.2]]),
        y_dim=1,
        selection_views=[view],
    )
    selection_path = workdir / "outputs" / "rounds" / "round_1" / "selection" / "selections.parquet"
    write_selection_parquet(
        selection_path,
        pd.DataFrame(
            {
                "selection_view_id": ["ethanol"],
                "id": ["b"],
                "score": [0.6],
                "selection_score": [0.6],
            }
        ),
    )
    metadata = build_run_meta_event(
        run_id=run_id,
        as_of_round=1,
        model_name="dummy",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="scalar_from_table_v1",
        y_ingest_transform_params={},
        objective_defs=[{"selection_view_id": "ethanol", "objective_name": "scalar_identity_v1"}],
        selection_view_defs=[{"selection_view_id": "ethanol", "selection_name": "top_n"}],
        stats_n_train=2,
        stats_n_scored=2,
        pred_rows_df=predictions,
        artifact_paths_and_hashes={
            "selection/selections.parquet": ("sha", str(selection_path)),
        },
    )
    writer = LedgerWriter(ws)
    writer.append_run_pred(predictions)
    writer.append_run_meta(metadata)

    projected = read_selection_view_predictions(
        ws.ledger_predictions_dir,
        selection_view_id="ethanol",
        columns=["id", "view__selection_score"],
        round_selector=1,
        run_id=run_id,
        runs_df=pl.read_parquet(ws.ledger_runs_path),
        require_run_id=False,
    ).to_pandas()
    selected = pd.read_parquet(selection_path)
    summary, mismatches = compare_selection_to_ledger(selected, projected, eps=1e-9)

    assert summary["mismatch_count"] == 0
    assert mismatches.empty


def test_verify_outputs_cli_reads_view_selection_score(tmp_path: Path) -> None:
    campaign_path, run_id = _write_cli_fixture(tmp_path, selection_score=0.1)

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "verify-outputs",
            "-c",
            str(campaign_path),
            "--view",
            "primary",
            "--run-id",
            run_id,
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["rows_compared"] == 1
    assert payload["summary"]["mismatch_count"] == 0


def test_verify_outputs_cli_json_mismatch_is_contract_violation(tmp_path: Path) -> None:
    campaign_path, run_id = _write_cli_fixture(tmp_path, selection_score=0.9)

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "verify-outputs",
            "-c",
            str(campaign_path),
            "--view",
            "primary",
            "--run-id",
            run_id,
            "--json",
        ],
    )

    assert result.exit_code == 4, result.output
    payload = json.loads(result.output)
    assert payload["summary"]["mismatch_count"] == 1
    assert payload["mismatches"][0]["id"] == "a"


def test_verify_outputs_cli_text_mismatch_is_contract_violation(tmp_path: Path) -> None:
    campaign_path, run_id = _write_cli_fixture(tmp_path, selection_score=0.9)

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "verify-outputs",
            "-c",
            str(campaign_path),
            "--view",
            "primary",
            "--run-id",
            run_id,
            "--no-hints",
        ],
    )

    assert result.exit_code == 4, result.output
    output = unstyle(result.output)
    assert "mismatches: 1" in output
    assert "top mismatches:" in output


def test_verify_outputs_requires_selection_view_argument() -> None:
    result = CliRunner().invoke(_build(), ["verify-outputs", "-c", "campaign.yaml"])

    assert result.exit_code == 2
    assert "--view" in unstyle(result.output)
