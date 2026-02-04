"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_objective_meta.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import write_campaign_yaml, write_ledger, write_records


def test_objective_meta_json_contains_diagnostics(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    write_ledger(workdir, run_id="run-0", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "objective-meta",
            "-c",
            str(campaign),
            "--round",
            "latest",
            "--no-profile",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert "obj__logic_fidelity" in out["row_level_diagnostics_columns"]


def test_objective_meta_accepts_directory_config(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    write_ledger(workdir, run_id="run-0", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "objective-meta",
            "-c",
            str(workdir),
            "--round",
            "latest",
            "--no-profile",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["round"] == 0


def test_objective_meta_requires_run_id_when_multiple_runs(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    # Two runs for the same round (ambiguous)
    write_ledger(workdir, run_id="run-0", round_index=0)
    write_ledger(workdir, run_id="run-1", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--no-color",
            "objective-meta",
            "-c",
            str(campaign),
            "--round",
            "latest",
            "--no-profile",
            "--json",
        ],
    )
    assert res.exit_code != 0
    assert "Multiple run_id" in res.output
