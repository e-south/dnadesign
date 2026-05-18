"""State schema fail-fast contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.storage.state import CampaignState
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def test_state_version_1_without_run_id_is_rejected(tmp_path: Path) -> None:
    state_path = Path(tmp_path) / "state.json"
    v1_state = {
        "version": 1,
        "campaign_slug": "demo",
        "campaign_name": "Demo",
        "workdir": str(tmp_path),
        "data_location": {
            "kind": "local",
            "path": str(tmp_path),
            "records_path": str(tmp_path / "records.parquet"),
        },
        "x_column_name": "X",
        "y_column_name": "Y",
        "rounds": [
            {
                "round_index": 0,
                "round_name": "round_0",
                "round_dir": str(tmp_path / "outputs" / "rounds" / "round_0"),
                "labels_used_rounds": [0],
                "number_of_training_examples_used_in_round": 2,
                "number_of_candidates_scored_in_round": 3,
                "selection_top_k_requested": 1,
                "selection_top_k_effective_after_ties": 1,
                "model": {},
                "metrics": {},
                "durations_sec": {},
                "seeds": {},
                "artifacts": {},
                "writebacks": {},
                "warnings": [],
            }
        ],
    }
    state_path.write_text(json.dumps(v1_state))

    with pytest.raises(ValueError, match="state.json version must be 2"):
        CampaignState.load(state_path)


def test_status_cli_reports_malformed_state_as_bad_args(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    (workdir / "state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "campaign_slug": "demo",
                "campaign_name": "Demo",
                "workdir": str(workdir),
                "data_location": {"kind": "local", "path": str(records)},
                "x_column_name": "X",
                "y_column_name": "Y",
                "rounds": [],
            }
        ),
        encoding="utf-8",
    )

    res = CliRunner().invoke(_build(), ["--no-color", "status", "-c", str(campaign), "--json"])

    assert res.exit_code != 0
    assert "Failed to load state.json" in res.output
    assert "version must be 2" in res.output
