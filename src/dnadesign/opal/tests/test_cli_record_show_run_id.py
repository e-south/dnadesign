"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_record_show_run_id.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import write_campaign_yaml, write_ledger, write_records


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    return workdir, campaign, records


def test_record_show_requires_run_id_when_multiple_runs(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    write_ledger(workdir, run_id="r0", round_index=0)
    write_ledger(workdir, run_id="r1", round_index=0)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "record-show", "-c", str(campaign), "--id", "a", "--json"])
    assert res.exit_code != 0
    assert "Multiple run_id" in res.output

    res_ok = runner.invoke(
        app,
        ["--no-color", "record-show", "-c", str(campaign), "--id", "a", "--run-id", "r1", "--json"],
    )
    assert res_ok.exit_code == 0, res_ok.stdout


def test_record_show_accepts_latest_run_id_alias(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    write_ledger(workdir, run_id="r0", round_index=0)
    write_ledger(workdir, run_id="r2", round_index=2)
    write_ledger(workdir, run_id="r1", round_index=2)

    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["--no-color", "record-show", "-c", str(campaign), "--id", "a", "--run-id", "latest", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    assert '"run_id": "r2"' in res.stdout
