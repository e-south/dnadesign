"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_runs_status_log.py

Regression tests for CLI runs status log OPAL CLI.

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
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "opal.runs_list.v1"
    assert payload["campaign"]["config_path"] == str(campaign)
    assert payload["round_selector"] is None
    runs = payload["runs"]
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
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "opal.run_meta.v1"
    assert payload["campaign"]["config_path"] == str(campaign)
    row = payload["run"]
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


def test_progress_json_summary(tmp_path):
    workdir, campaign, _ = _setup_workspace(tmp_path)
    stale_plot = workdir / "outputs" / "plots" / "old.png"
    stale_plot.parent.mkdir(parents=True, exist_ok=True)
    stale_plot.write_bytes(b"stale")
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "progress", "-c", str(campaign), "--round", "all", "--json"])

    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["schema_version"] == "opal.campaign_progress.v1"
    assert out["status"] == "done"
    assert out["round_count"] == 1
    assert out["locks"]["campaign"]["scope"] == "local_host"
    assert "warnings" in out
    assert out["event_contract"]["run_events"] == 5
    assert out["event_contract"]["aborted_rounds"] == []
    assert "run_scope" in out["rounds"][0]["summary"]
    assert out["rounds"][0]["predict"]["rows"] == 2
    assert out["artifact_garden"]["schema_version"] == "opal.artifact_garden.v1"
    assert out["artifact_garden"]["stale_artifact_count"] == 1
    assert out["stale_artifacts"][0]["path"] == str(stale_plot)
    assert any(row["category"] == "StaleArtifactWarning" for row in out["warnings"])

    res = runner.invoke(
        app,
        ["--no-color", "progress", "-c", str(campaign), "--round", "all", "--run-id", "run-0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    scoped = json.loads(res.stdout)
    assert scoped["run_id"] == "run-0"
    assert scoped["rounds"][0]["summary"]["run_scope"]["requested_run_id"] == "run-0"


def test_progress_rejects_missing_explicit_round(tmp_path):
    _, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "progress", "-c", str(campaign), "--round", "7", "--json"])

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert "--round 7 not found" in payload["error"]["message"]


def test_status_json_error_for_missing_config(tmp_path):
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "status", "-c", str(tmp_path / "missing.yaml"), "--json"])

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert payload["error"]["context"] == "status"


def test_runs_list_json_error_for_missing_config(tmp_path):
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "runs", "list", "-c", str(tmp_path / "missing.yaml"), "--json"])

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert payload["error"]["context"] == "runs list"


def test_artifacts_audit_json_error_uses_shared_cli_error_contract(tmp_path):
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "artifacts", "audit", "-c", str(tmp_path / "missing.yaml"), "--json"])

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert payload["error"]["context"] == "artifacts audit"
    assert payload["error"]["category"] == "OpalError"
    assert isinstance(payload["error"]["exit_code"], int)
