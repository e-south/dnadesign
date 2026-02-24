"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_guidance_hints.py

CLI tests for next-step hint output controls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True, slug="demo")
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug="demo")
    return workdir, campaign


def test_init_prints_hints_by_default_and_hides_with_no_hints(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res_default = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res_default.exit_code == 0, res_default.stdout
    assert "Next steps" in res_default.stdout

    res_no_hints = runner.invoke(app, ["--no-color", "init", "-c", str(campaign), "--no-hints"])
    assert res_no_hints.exit_code == 0, res_no_hints.stdout
    assert "Next steps" not in res_no_hints.stdout


def test_init_json_output_not_polluted_by_hints(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["ok"] is True


def test_explain_surfaces_sfxi_round_label_preflight_warning(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    init_res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert init_res.exit_code == 0, init_res.stdout

    explain_res = runner.invoke(app, ["--no-color", "explain", "-c", str(campaign), "--labels-as-of", "0"])
    assert explain_res.exit_code == 0, explain_res.stdout
    assert "Run preflight" in explain_res.stdout
    assert "current-round labels" in explain_res.stdout
