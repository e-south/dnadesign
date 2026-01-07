"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_pressure_scalar_y.py

Module Author(s): Eric J. South (extended by Codex)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import write_campaign_yaml


def _write_records(path: Path) -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "sequence": ["AAA", "BBB", "CCC", "DDD"],
            "bio_type": ["dna", "dna", "dna", "dna"],
            "alphabet": ["dna_4", "dna_4", "dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
        }
    )
    df.to_parquet(path, index=False)


def _write_labels(path: Path, *, seqs: list[str], ys: list[float]) -> None:
    df = pd.DataFrame({"sequence": seqs, "y": ys})
    df.to_csv(path, index=False)


def test_cli_pressure_scalar_multi_round(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 2},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout
    df = pd.read_parquet(records)
    assert "opal__demo__label_hist" in df.columns
    assert "opal__demo__latest_as_of_round" in df.columns
    assert "opal__demo__latest_pred_scalar" in df.columns

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    # Round 0 labels
    labels0 = workdir / "labels_r0.csv"
    _write_labels(labels0, seqs=["AAA", "BBB"], ys=[0.2, 0.8])
    res = runner.invoke(
        app,
        ["--no-color", "ingest-y", "-c", str(campaign), "--round", "0", "--csv", str(labels0), "--yes"],
    )
    assert res.exit_code == 0, res.stdout

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "0"])
    assert res.exit_code == 0, res.stdout

    # Round 1 labels (new data)
    labels1 = workdir / "labels_r1.csv"
    _write_labels(labels1, seqs=["CCC"], ys=[0.5])
    res = runner.invoke(
        app,
        ["--no-color", "ingest-y", "-c", str(campaign), "--round", "1", "--csv", str(labels1), "--yes"],
    )
    assert res.exit_code == 0, res.stdout

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "1"])
    assert res.exit_code == 0, res.stdout

    res = runner.invoke(app, ["--no-color", "status", "-c", str(campaign), "--with-ledger", "--json"])
    assert res.exit_code == 0, res.stdout
    status = json.loads(res.stdout)
    assert status["latest_round"]["round_index"] == 1

    res = runner.invoke(app, ["--no-color", "record-show", "-c", str(campaign), "--id", "d", "--json"])
    assert res.exit_code == 0, res.stdout
    report = json.loads(res.stdout)
    assert report.get("id") == "d"
    assert len(report.get("runs", [])) >= 2

    res = runner.invoke(app, ["--no-color", "log", "-c", str(campaign), "--round", "0", "--json"])
    assert res.exit_code == 0, res.stdout
    log0 = json.loads(res.stdout)
    assert log0.get("predict_rows") == 2

    res = runner.invoke(app, ["--no-color", "log", "-c", str(campaign), "--round", "1", "--json"])
    assert res.exit_code == 0, res.stdout
    log1 = json.loads(res.stdout)
    assert log1.get("predict_rows") == 1
