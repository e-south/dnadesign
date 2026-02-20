"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_guide_next.py

State-aware CLI tests for guided next-step recommendations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records, write_state


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True, slug="demo")
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug="demo")
    return workdir, campaign, records


def _write_round0_label(records_path: Path) -> None:
    df = pd.read_parquet(records_path)
    lh_col = "opal__demo__label_hist"
    df.at[0, lh_col] = [
        {
            "kind": "label",
            "observed_round": 0,
            "ts": "2026-01-01T00:00:00Z",
            "src": "ingest_y",
            "y_obs": {
                "value": [0.1, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 2.0],
                "dtype": "vector",
                "schema": {"length": 8},
            },
        }
    ]
    df.to_parquet(records_path, index=False)


def test_guide_next_recommends_init_when_state_missing(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "next", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "init"
    assert "opal init -c" in out["next_commands"][0]


def test_guide_next_recommends_ingest_when_state_exists_but_round_has_no_labels(tmp_path: Path) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path)
    write_state(workdir, records_path=records, run_id="seed", round_index=0)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(
        app,
        ["--no-color", "guide", "next", "-c", str(campaign), "--labels-as-of", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "ingest"
    assert "--observed-round 0" in out["next_commands"][0]


def test_guide_next_recommends_run_after_labels_exist(tmp_path: Path) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path)
    _ = workdir
    app = _build()
    runner = CliRunner()

    init_res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert init_res.exit_code == 0, init_res.stdout
    _write_round0_label(records)

    res = runner.invoke(
        app,
        ["--no-color", "guide", "next", "-c", str(campaign), "--labels-as-of", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "run"
    assert "--labels-as-of 0" in out["next_commands"][0]


def test_guide_next_recommends_verify_after_round_exists(tmp_path: Path) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path)
    _write_round0_label(records)
    write_state(workdir, records_path=records, run_id="r0", round_index=0)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(
        app,
        ["--no-color", "guide", "next", "-c", str(campaign), "--labels-as-of", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "post_run"
    assert any("verify-outputs" in cmd for cmd in out["next_commands"])
