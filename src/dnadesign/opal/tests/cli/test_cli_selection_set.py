"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_selection_set.py

Tests for the public OPAL selection-set contract used by downstream study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal import load_selection_set
from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records, write_state


def _setup_workspace(tmp_path: Path, *, run_ids: tuple[str, ...] = ("run-0",)) -> tuple[Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    for run_id in run_ids:
        write_state(workdir, records_path=records, run_id=run_id, round_index=0)
        write_ledger(workdir, run_id=run_id, round_index=0)
    return workdir, campaign


def _write_selection_artifact(workdir: Path) -> Path:
    selection_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.csv"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "a",
                "sequence": "AAA",
                "pred__score_selected": 0.1,
                "sel__rank_competition": 1,
            }
        ]
    ).to_csv(selection_path, index=False)
    return selection_path


def test_load_selection_set_resolves_latest_round_and_verifies_artifact(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    selection_path = _write_selection_artifact(workdir)

    selection_set = load_selection_set(campaign, round_selector="latest")

    assert selection_set["schema_version"] == "opal.selection_set.v1"
    assert selection_set["campaign"]["slug"] == "demo"
    assert selection_set["as_of_round"] == 0
    assert selection_set["run_id"] == "run-0"
    assert selection_set["selected_count"] == 1
    assert selection_set["selection_path"] == str(selection_path)
    assert selection_set["verification"]["status"] == "pass"
    assert selection_set["rows"] == [
        {
            "id": "a",
            "sequence": "AAA",
            "selection_rank": 1,
            "sel__rank_competition": 1,
            "pred__score_selected": 0.1,
            "run_id": "run-0",
            "as_of_round": 0,
        }
    ]


def test_selection_set_show_and_export_json(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _write_selection_artifact(workdir)
    output_csv = tmp_path / "selection_set.csv"
    app = _build()
    runner = CliRunner()

    res = runner.invoke(
        app, ["--no-color", "selection-set", "show", "-c", str(campaign), "--round", "latest", "--json"]
    )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "opal.selection_set.v1"
    assert payload["rows"][0]["id"] == "a"
    assert payload["verification"]["status"] == "pass"

    res = runner.invoke(
        app,
        [
            "--no-color",
            "selection-set",
            "export",
            "-c",
            str(campaign),
            "--round",
            "latest",
            "--out",
            str(output_csv),
            "--format",
            "csv",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "opal.selection_set_export.v1"
    assert payload["output_path"] == str(output_csv)
    assert payload["row_count"] == 1
    exported = pd.read_csv(output_csv)
    assert exported[["id", "sequence", "selection_rank"]].to_dict("records") == [
        {"id": "a", "sequence": "AAA", "selection_rank": 1}
    ]


def test_selection_set_rejects_ambiguous_reruns_without_run_id(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path, run_ids=("run-a", "run-b"))
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "selection-set", "show", "-c", str(campaign), "--round", "0", "--json"])

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert "Multiple run_id values found for round 0" in payload["error"]["message"]


def test_selection_set_missing_ledger_returns_json_error(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "selection-set", "show", "-c", str(campaign), "--json"])

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert "Missing runs sink" in payload["error"]["message"]
