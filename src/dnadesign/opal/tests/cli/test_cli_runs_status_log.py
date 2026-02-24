"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_runs_status_log.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records, write_state


def _setup_workspace(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    run_id = "run-0"
    write_state(workdir, records_path=records, run_id=run_id, round_index=0)
    write_ledger(workdir, run_id=run_id, round_index=0)
    return workdir, campaign, run_id


def test_runs_list_and_show_json(tmp_path):
    _, campaign, run_id = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "runs", "list", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    runs = json.loads(res.stdout)
    assert any(r.get("run_id") == run_id for r in runs)

    res = runner.invoke(
        app,
        [
            "--no-color",
            "runs",
            "show",
            "-c",
            str(campaign),
            "--run-id",
            run_id,
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    row = json.loads(res.stdout)
    assert row["run_id"] == run_id


def test_status_with_ledger_json(tmp_path):
    _, campaign, run_id = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "status", "-c", str(campaign), "--with-ledger", "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["latest_round"]["run_id"] == run_id
    assert out["latest_round_ledger"]["run_id"] == run_id


def test_status_rejects_round_and_all(tmp_path):
    _, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "status", "-c", str(campaign), "--round", "0", "--all"])
    assert res.exit_code != 0, res.stdout
    assert "only one of --all or --round" in res.output.lower()


def test_log_json_summary(tmp_path):
    _, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "log", "-c", str(campaign), "--round", "latest", "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["events"] == 5
    assert out["predict_rows"] == 2
