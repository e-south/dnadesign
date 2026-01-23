# ABOUTME: Tests the demo-reset CLI command for demo campaign cleanup.
# ABOUTME: Ensures records pruning and output/state removal for demo runs.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_demo_reset.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import write_campaign_yaml, write_records


def test_demo_reset_prunes_and_clears(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    outputs = workdir / "outputs" / "rounds" / "round_0"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "round_ctx.json").write_text("{}")
    state_path = workdir / "state.json"
    state_path.write_text("{}")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "demo-reset", "-c", str(campaign), "--yes", "--no-backup"])
    assert res.exit_code == 0, res.stdout

    assert not (workdir / "outputs").exists()
    assert not state_path.exists()

    df = pd.read_parquet(records)
    assert "opal__demo__label_hist" not in df.columns
    assert "Y" not in df.columns
