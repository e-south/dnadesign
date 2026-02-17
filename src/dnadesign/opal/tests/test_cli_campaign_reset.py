"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_campaign_reset.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build

from ._cli_helpers import write_campaign_yaml, write_records


def test_campaign_reset_requires_config_or_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "campaign-reset", "--apply", "--no-backup"])
    assert res.exit_code != 0
    assert "No config provided" in res.stdout or "No config provided" in res.stderr


def test_campaign_reset_prunes_and_clears(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    outputs = workdir / "outputs" / "rounds" / "round_0"
    outputs.mkdir(parents=True, exist_ok=True)
    ctx_path = outputs / "metadata" / "round_ctx.json"
    ctx_path.parent.mkdir(parents=True, exist_ok=True)
    ctx_path.write_text("{}")
    notebooks_dir = workdir / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    (notebooks_dir / "demo.py").write_text("import marimo\nmarimo.App()")
    state_path = workdir / "state.json"
    state_path.write_text("{}")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "campaign-reset", "-c", str(campaign), "--apply", "--no-backup"])
    assert res.exit_code == 0, res.stdout

    assert not (workdir / "outputs").exists()
    assert not state_path.exists()
    assert not notebooks_dir.exists()

    df = pd.read_parquet(records)
    assert "opal__demo__label_hist" not in df.columns
    assert "Y" not in df.columns


def test_campaign_reset_allows_non_demo_without_flag(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug="alpha")

    app = _build()
    runner = CliRunner()
    outputs = workdir / "outputs" / "rounds" / "round_0"
    outputs.mkdir(parents=True, exist_ok=True)
    state_path = workdir / "state.json"
    state_path.write_text("{}")

    res = runner.invoke(app, ["--no-color", "campaign-reset", "-c", str(campaign), "--apply", "--no-backup"])
    assert res.exit_code == 0, res.stdout
    assert not (workdir / "outputs").exists()
    assert not state_path.exists()


def test_campaign_reset_allows_demo_prefixed_slug_without_flag(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True, slug="demo_gp_topn")

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug="demo_gp_topn")

    outputs = workdir / "outputs" / "rounds" / "round_0"
    outputs.mkdir(parents=True, exist_ok=True)
    state_path = workdir / "state.json"
    state_path.write_text("{}")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "campaign-reset", "-c", str(campaign), "--apply", "--no-backup"])
    assert res.exit_code == 0, res.stdout
    assert not (workdir / "outputs").exists()
    assert not state_path.exists()


def test_campaign_reset_rejects_removed_allow_non_demo_flag(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True, slug="alpha")

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug="alpha")

    app = _build()
    runner = CliRunner()
    res = runner.invoke(app, ["--no-color", "campaign-reset", "-c", str(campaign), "--allow-non-demo"])
    assert res.exit_code != 0
    text = f"{res.stdout}\n{res.stderr}".lower()
    assert "no such option" in text or "--allow-non-demo" in text
