"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/reporting/test_review.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.reporting.review import build_campaign_review
from dnadesign.opal.tests._cli_helpers import (
    write_campaign_yaml,
    write_ledger,
    write_records,
    write_round_log,
    write_state,
)


def _setup_review_campaign(tmp_path: Path) -> tuple[Path, Path, str]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    run_id = "run-0"
    write_state(workdir, records_path=records, run_id=run_id, round_index=0)
    write_ledger(workdir, run_id=run_id, round_index=0)
    feature_dir = workdir / "outputs" / "rounds" / "round_0" / "model"
    feature_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature_index": [0, 1], "importance": [0.25, 0.75]}).to_csv(
        feature_dir / "feature_importance.csv",
        index=False,
    )
    return workdir, campaign, run_id


def test_build_campaign_review_writes_portable_artifacts(tmp_path: Path) -> None:
    workdir, campaign, run_id = _setup_review_campaign(tmp_path)

    result = build_campaign_review(campaign, run_id=run_id)

    assert result.manifest_path == workdir / "outputs" / "review" / "manifest.json"
    assert result.review_path.exists()
    assert result.index_path.exists()
    assert result.manifest["schema_version"] == "opal.campaign_review.v1"
    assert result.manifest["review_scope"]["run_id"] == run_id
    assert result.manifest["selection"]["selected_count"] == 1
    assert any(path.name.startswith("score_vs_rank") for path in result.plot_paths)
    assert any(path.name.startswith("feature_importance_top") for path in result.plot_paths)
    text = result.review_path.read_text(encoding="utf-8")
    assert "# OPAL campaign review" in text
    assert "## Progress" in text
    html = result.index_path.read_text(encoding="utf-8")
    assert "OPAL campaign review" in html
    assert "score_vs_rank" in html
    assert result.manifest["campaign"]["x_contract"]["canonical"] is True


def test_review_cli_writes_json_summary(tmp_path: Path) -> None:
    _, campaign, run_id = _setup_review_campaign(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "review", "-c", str(campaign), "--run-id", run_id, "--json"])

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["manifest"]["schema_version"] == "opal.campaign_review.v1"
    assert payload["manifest"]["review_scope"]["run_id"] == run_id
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["index_path"]).exists()


def test_review_cli_writes_json_error_for_not_started_campaign(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    records = workdir / "records.parquet"
    records.parent.mkdir(parents=True, exist_ok=True)
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "review", "-c", str(campaign), "--json"])

    assert res.exit_code != 0
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert "outputs/ledger/runs.parquet" in payload["error"]["message"]


def test_build_campaign_review_rejects_run_id_round_mismatch(tmp_path: Path) -> None:
    workdir, campaign, run_id = _setup_review_campaign(tmp_path)
    write_ledger(workdir, run_id="run-1", round_index=1)

    with pytest.raises(OpalError, match="but --round selected 1"):
        build_campaign_review(campaign, round_selector="1", run_id=run_id)


def test_build_campaign_review_requires_run_scoped_round_log(tmp_path: Path) -> None:
    workdir, campaign, run_id = _setup_review_campaign(tmp_path)
    write_round_log(workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl")

    with pytest.raises(OpalError, match=f"round.log.jsonl has no events for run_id={run_id}"):
        build_campaign_review(campaign, run_id=run_id)


def test_build_campaign_review_reports_stale_plot_files_when_plots_disabled(tmp_path: Path) -> None:
    workdir, campaign, run_id = _setup_review_campaign(tmp_path)
    stale_dir = workdir / "outputs" / "review" / "plots"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_file = stale_dir / "old_score.png"
    stale_file.write_bytes(b"old")

    result = build_campaign_review(campaign, run_id=run_id, include_plots=False)

    assert result.plot_paths == ()
    assert result.manifest["plots"] == []
    assert result.manifest["stale_artifacts"]
    assert result.manifest["warnings"][0]["category"] == "StaleArtifactWarning"
    assert result.manifest["stale_artifacts"][0]["path"] == str(stale_file)


def test_build_campaign_review_rejects_duplicate_prediction_ids(tmp_path: Path) -> None:
    workdir, campaign, run_id = _setup_review_campaign(tmp_path)
    predictions_dir = workdir / "outputs" / "ledger" / "predictions"
    predictions = pd.read_parquet(predictions_dir)
    duplicate = predictions.loc[predictions["id"].astype(str) == "a"].head(1).copy()
    pq.write_table(pa.Table.from_pandas(duplicate, preserve_index=False), predictions_dir / "part-duplicate.parquet")

    with pytest.raises(OpalError, match="duplicate_prediction_ids"):
        build_campaign_review(campaign, run_id=run_id)
