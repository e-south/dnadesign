"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_objective_meta.py

Regression tests for CLI objective meta OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import polars as pl
import pytest
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records


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
            "--view",
            "primary",
            "--round",
            "latest",
            "--no-profile",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert "logic_fidelity" in out["diagnostic_keys"]


def test_objective_meta_profiles_numpy_backed_msrb_diagnostics(tmp_path, monkeypatch):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    write_ledger(workdir, run_id="run-0", round_index=0)

    predictions = pl.DataFrame(
        {
            "view__score": [0.2, 0.4],
            "view__selection_score": [0.2, 0.4],
            "view__rank_competition": [2, 1],
            "view__uncertainty": [None, None],
            "view__diagnostics": [
                [
                    {"name": "response_family_score", "value": 0.3},
                    {"name": "on_signal_family_score", "value": 0.5},
                    {"name": "off_signal_suppression_family_score", "value": -0.1},
                    {"name": "hard_bottleneck_clearance", "value": -0.4},
                ],
                [
                    {"name": "response_family_score", "value": 0.6},
                    {"name": "on_signal_family_score", "value": 0.7},
                    {"name": "off_signal_suppression_family_score", "value": 0.2},
                    {"name": "hard_bottleneck_clearance", "value": -0.2},
                ],
            ],
        }
    )
    monkeypatch.setattr(
        "dnadesign.opal.src.cli.commands.objective_meta.read_selection_view_predictions",
        lambda *args, **kwargs: predictions,
    )

    res = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "objective-meta",
            "-c",
            str(campaign),
            "--view",
            "primary",
            "--round",
            "latest",
            "--profile",
            "--json",
        ],
    )

    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["diagnostic_keys"] == [
        "hard_bottleneck_clearance",
        "off_signal_suppression_family_score",
        "on_signal_family_score",
        "response_family_score",
    ]
    profile = {row["column"]: row for row in out["profile"]["columns"]}
    response_profile = profile["diagnostic/response_family_score"]
    assert response_profile["count"] == 2
    assert response_profile["finite_count"] == 2
    assert response_profile["min"] == pytest.approx(0.3)
    assert response_profile["median"] == pytest.approx(0.45)
    assert response_profile["max"] == pytest.approx(0.6)


def test_objective_meta_accepts_directory_config(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "configs" / "campaign.yaml"
    campaign.parent.mkdir(parents=True)
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
            "--view",
            "primary",
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
            "--view",
            "primary",
            "--round",
            "latest",
            "--no-profile",
            "--json",
        ],
    )
    assert res.exit_code != 0
    assert "Multiple run_id" in res.output
