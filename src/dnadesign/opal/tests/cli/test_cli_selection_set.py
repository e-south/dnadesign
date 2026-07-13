"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_selection_set.py

Public selection-view and selection-batch inspection contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal import load_selection_batch, load_selection_set
from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records, write_state


def _setup_workspace(tmp_path: Path, *, run_ids: tuple[str, ...] = ("run-0",)) -> tuple[Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    selections, batch = _write_selection_artifacts(workdir)
    artifacts = {
        "selection/selections.parquet": ("selection-sha", str(selections)),
        "selection/selection_batch.parquet": ("batch-sha", str(batch)),
    }
    for run_id in run_ids:
        write_state(workdir, records_path=records, run_id=run_id, round_index=0)
        write_ledger(
            workdir,
            run_id=run_id,
            round_index=0,
            artifact_paths_and_hashes=artifacts,
        )
    return workdir, campaign


def _write_selection_artifacts(workdir: Path) -> tuple[Path, Path]:
    selection_dir = workdir / "outputs" / "rounds" / "round_0" / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selections = selection_dir / "selections.parquet"
    batch = selection_dir / "selection_batch.parquet"
    pd.DataFrame(
        [
            {
                "selection_view_id": "primary",
                "id": "a",
                "sequence": "AAA",
                "score": 0.1,
                "selection_score": 0.1,
                "rank_competition": 1,
            }
        ]
    ).to_parquet(selections, index=False)
    pd.DataFrame(
        [
            {
                "id": "a",
                "selection_batch_key": "a",
                "deduplicate_by": "id",
                "selection_view_ids": ["primary"],
                "selection_memberships": [
                    {
                        "selection_view_id": "primary",
                        "rank": 1,
                        "score": 0.1,
                        "selection_score": 0.1,
                        "score_ref": "primary/sfxi",
                    }
                ],
            }
        ]
    ).to_parquet(batch, index=False)
    return selections, batch


def test_load_selection_set_requires_and_projects_named_view(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    selection_path, _ = _write_selection_artifacts(workdir)

    payload = load_selection_set(campaign, selection_view_id="primary", round_selector="latest")

    assert payload["schema_version"] == "opal.selection_set.v2"
    assert payload["selection_view_id"] == "primary"
    assert payload["selection_path"] == str(selection_path)
    assert payload["verification"]["status"] == "pass"
    assert payload["rows"] == [
        {
            "id": "a",
            "sequence": "AAA",
            "selection_rank": 1,
            "rank_competition": 1,
            "score": 0.1,
            "selection_score": 0.1,
            "run_id": "run-0",
            "as_of_round": 0,
        }
    ]


def test_load_selection_batch_returns_logical_union(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _, batch_path = _write_selection_artifacts(workdir)

    payload = load_selection_batch(campaign, round_selector="latest")

    assert payload["schema_version"] == "opal.selection_batch.v1"
    assert payload["selection_batch_path"] == str(batch_path)
    assert payload["unique_count"] == 1
    assert payload["rows"][0]["selection_view_ids"] == ["primary"]


def test_selection_set_and_batch_cli_show_and_export(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    _write_selection_artifacts(workdir)
    output_csv = tmp_path / "selection_set.csv"
    runner = CliRunner()
    app = _build()

    shown = runner.invoke(
        app,
        [
            "--no-color",
            "selection-set",
            "show",
            "-c",
            str(campaign),
            "--round",
            "latest",
            "--view",
            "primary",
            "--json",
        ],
    )
    assert shown.exit_code == 0, shown.stdout
    assert json.loads(shown.stdout)["rows"][0]["id"] == "a"

    exported = runner.invoke(
        app,
        [
            "--no-color",
            "selection-set",
            "export",
            "-c",
            str(campaign),
            "--view",
            "primary",
            "--out",
            str(output_csv),
            "--json",
        ],
    )
    assert exported.exit_code == 0, exported.stdout
    assert pd.read_csv(output_csv)["id"].tolist() == ["a"]

    batch = runner.invoke(
        app,
        ["--no-color", "selection-batch", "show", "-c", str(campaign), "--round", "latest", "--json"],
    )
    assert batch.exit_code == 0, batch.stdout
    assert json.loads(batch.stdout)["unique_count"] == 1


def test_selection_set_rejects_ambiguous_reruns_without_run_id(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path, run_ids=("run-a", "run-b"))
    _write_selection_artifacts(workdir)

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "selection-set", "show", "-c", str(campaign), "--view", "primary", "--round", "0", "--json"],
    )

    assert result.exit_code != 0
    assert "Multiple run_id values found for round 0" in json.loads(result.stdout)["error"]["message"]


def test_selection_set_does_not_infer_missing_artifact_reference(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    write_state(workdir, records_path=records, run_id="run-0", round_index=0)
    write_ledger(workdir, run_id="run-0", round_index=0)
    _write_selection_artifacts(workdir)

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "selection-set", "show", "-c", str(campaign), "--view", "primary", "--json"],
    )

    assert result.exit_code != 0
    assert (
        "missing the selection/selections.parquet artifact reference" in json.loads(result.stdout)["error"]["message"]
    )
