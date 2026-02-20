"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_guide.py

CLI tests for guided workflow runbook generation.

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


def test_guide_json_includes_campaign_plugins_steps_and_doc_pointers(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "-c", str(campaign), "--format", "json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)

    assert out["campaign"]["slug"] == "demo"
    assert out["plugins"]["model"]["name"] == "random_forest"
    assert out["plugins"]["selection"]["name"] == "top_n"
    assert out["workflow_key"] == "rf_sfxi_topn"
    assert any("opal run -c" in str(step["command"]) for step in out["steps"])
    assert "docs/plugins/objective-sfxi.md" in out["learn_more"]["docs"]
    assert "src/dnadesign/opal/src/models/random_forest.py" in out["learn_more"]["source"]
    assert "src/dnadesign/opal/src/runtime/round/stages.py" in out["learn_more"]["source"]


def test_guide_markdown_contains_round_semantics_and_commands(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "-c", str(campaign), "--format", "markdown"])
    assert res.exit_code == 0, res.stdout
    text = res.stdout

    assert "## Guided Workflow" in text
    assert "--observed-round" in text
    assert "--labels-as-of" in text
    assert "opal init -c" in text
    assert "opal run -c" in text
