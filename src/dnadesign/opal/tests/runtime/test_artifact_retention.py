from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.opal.src.config.loader import load_config
from dnadesign.opal.src.runtime.retention import apply_runtime_artifact_retention
from dnadesign.opal.src.storage.parquet_io import read_parquet_df
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_ledger, write_records


def test_production_review_retention_keeps_latest_full_and_selected_history(tmp_path: Path) -> None:
    workdir = tmp_path / ".var" / "campaign"
    workdir.mkdir(parents=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    _set_artifact_retention(campaign, final_round=2)
    cfg = load_config(campaign)
    ws = CampaignWorkspace.from_config(cfg, campaign)

    write_ledger(workdir, run_id="run-r0", round_index=0)
    write_ledger(workdir, run_id="run-r1", round_index=1)
    (workdir / "outputs" / "rounds" / "round_0" / "model").mkdir(parents=True)
    (workdir / "outputs" / "rounds" / "round_0" / "model" / "model.joblib").write_bytes(b"old")
    (workdir / "outputs" / "rounds" / "round_1" / "model").mkdir(parents=True)
    (workdir / "outputs" / "rounds" / "round_1" / "model" / "model.joblib").write_bytes(b"latest")

    manifest = apply_runtime_artifact_retention(cfg, ws)

    assert manifest["status"] == "PASS"
    assert (workdir / "outputs" / "retention_manifest.json").exists()
    retained = read_parquet_df(workdir / "outputs" / "ledger" / "predictions")
    assert len(retained) == 3
    assert set(retained.loc[retained["as_of_round"] == 1, "id"]) == {"a", "b"}
    assert set(retained.loc[retained["as_of_round"] == 0, "id"]) == {"a"}
    assert not (workdir / "outputs" / "rounds" / "round_0" / "model").exists()
    assert (workdir / "outputs" / "rounds" / "round_1" / "model").exists()


def test_audit_full_retention_only_writes_manifest(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir()
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    cfg = load_config(campaign)
    ws = CampaignWorkspace.from_config(cfg, campaign)

    manifest = apply_runtime_artifact_retention(cfg, ws)

    assert manifest["policy"]["mode"] == "audit_full"
    assert manifest["actions"] == []
    assert (workdir / "outputs" / "retention_manifest.json").exists()


def _set_artifact_retention(campaign: Path, *, final_round: int) -> None:
    payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    payload["artifact_retention"] = {
        "mode": "production_review",
        "prediction_ledger": "latest_full_plus_selected_history",
        "plot_tidy_data": "compact",
        "model_artifacts": "latest",
        "tabular_format": "parquet_zstd",
        "max_estimated_bytes": 50_000_000_000,
        "fail_if_estimate_exceeds": True,
        "final_round": int(final_round),
    }
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
