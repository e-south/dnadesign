"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_prune_source.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import write_campaign_yaml, write_records


def test_prune_source_preview_and_commit(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "prune-source", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    preview = json.loads(res.stdout)
    assert preview["preview"]["to_delete_count"] > 0

    df_before = pd.read_parquet(records)
    assert any(c.startswith("opal__") for c in df_before.columns)
    assert "Y" in df_before.columns

    res = runner.invoke(app, ["--no-color", "prune-source", "-c", str(campaign), "--apply", "--json"])
    assert res.exit_code == 0, res.stdout
    decoder = json.JSONDecoder()
    idx = 0
    docs = []
    text = res.stdout
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        docs.append(obj)
        idx = end
    out = docs[-1]
    assert out["deleted_count"] > 0
    assert out["backup_path"]
    assert Path(out["backup_path"]).exists()

    df_after = pd.read_parquet(records)
    assert not any(c.startswith("opal__") for c in df_after.columns)
    assert "Y" not in df_after.columns
