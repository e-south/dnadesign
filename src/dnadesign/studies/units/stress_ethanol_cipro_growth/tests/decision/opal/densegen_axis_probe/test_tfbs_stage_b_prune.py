"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_b_prune.py

Regression tests for TFBS stage b prune studies units stress ethanol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .probe_modules import probe_module

main = probe_module("cli").main
prune_tfbs_stage_b_campaigns = probe_module("tfbs.stage_b.prune").prune_tfbs_stage_b_campaigns


def test_stage_b_prune_hard_deletes_campaigns_and_rewrites_manifest(tmp_path: Path) -> None:
    manifest_path = _write_prune_fixture(tmp_path)

    result = prune_tfbs_stage_b_campaigns(
        manifest_path,
        prune_label_names=("lexA_in_slot0",),
        delete_review_artifacts=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result.status == "PASS"
    assert manifest["campaign_count"] == 2
    assert manifest["sentinel_labels"] == ["lexA_present"]
    assert [row["label_name"] for row in manifest["pairs"]] == ["lexA_present"]
    assert {row["label_name"] for row in manifest["campaigns"]} == {"lexA_present"}
    assert {row["campaign_key"] for row in manifest["validation"]["reports"]} == {
        "tfbs_lexA_present_positive_random_id_seed7",
        "tfbs_lexA_present_matched_null_random_id_seed7",
    }
    assert not (tmp_path / "campaigns" / "tfbs_lexA_in_slot0_positive_random_id_seed7").exists()
    assert not (tmp_path / "campaigns" / "tfbs_lexA_in_slot0_matched_null_random_id_seed7").exists()
    assert not (
        tmp_path / "validation_reports" / "tfbs_lexA_in_slot0_positive_random_id_seed7.opal_validate.json"
    ).exists()
    assert not (tmp_path / "review" / "realized_labels").exists()
    assert not (tmp_path / "notebooks" / "collection_visuals").exists()
    assert (tmp_path / "manifests" / "stage_b_sentinel_prune_manifest.json").exists()


def test_stage_b_prune_cli_emits_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = _write_prune_fixture(tmp_path)

    assert (
        main(
            [
                "tfbs-stage-b-prune",
                "--config-manifest",
                str(manifest_path),
                "--label-name",
                "lexA_in_slot0",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["pruned_campaign_count"] == 2
    assert payload["retained_campaign_count"] == 2


def test_stage_b_prune_uses_manifest_labels_for_non_seed7_campaign_keys(tmp_path: Path) -> None:
    manifest_path = _write_prune_fixture(tmp_path, seed=17)

    prune_tfbs_stage_b_campaigns(
        manifest_path,
        prune_label_names=("lexA_in_slot0",),
        delete_review_artifacts=False,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {row["campaign_key"] for row in manifest["validation"]["reports"]} == {
        "tfbs_lexA_present_positive_random_id_seed17",
        "tfbs_lexA_present_matched_null_random_id_seed17",
    }
    assert not (
        tmp_path / "validation_reports" / "tfbs_lexA_in_slot0_positive_random_id_seed17.opal_validate.json"
    ).exists()


def _write_prune_fixture(root: Path, *, seed: int = 7) -> Path:
    (root / "manifests").mkdir(parents=True)
    campaigns = []
    pairs = []
    validation_reports = []
    for label_name in ("lexA_present", "lexA_in_slot0"):
        role_keys = {}
        for role in ("positive", "matched_null"):
            key = f"tfbs_{label_name}_{role}_random_id_seed{seed}"
            workdir = root / "campaigns" / key
            config_path = workdir / "configs" / "campaign.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("campaign:\n  workdir: placeholder\n", encoding="utf-8")
            (workdir / "outputs").mkdir()
            report_path = root / "validation_reports" / f"{key}.opal_validate.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("{}", encoding="utf-8")
            role_keys[role] = key
            campaigns.append(
                {
                    "campaign_key": key,
                    "label_name": label_name,
                    "oracle_role": role,
                    "config_path": str(config_path),
                    "label_sidecar_path": str(root / "scratch_usr" / "_opal" / key / "observed_labels.parquet"),
                }
            )
            validation_reports.append(
                {
                    "campaign_key": key,
                    "report_path": str(report_path),
                    "status": "PASS",
                }
            )
        pairs.append(
            {
                "label_name": label_name,
                "positive_campaign_key": role_keys["positive"],
                "null_campaign_key": role_keys["matched_null"],
            }
        )
    (root / "review" / "realized_labels").mkdir(parents=True)
    (root / "notebooks" / "collection_visuals").mkdir(parents=True)
    manifest = {
        "schema_version": "fixture.stage_b",
        "status": "PASS",
        "campaign_count": 4,
        "sentinel_labels": ["lexA_present", "lexA_in_slot0"],
        "campaigns": campaigns,
        "pairs": pairs,
        "validation": {"status": "PASS", "campaign_count": 4, "reports": validation_reports},
    }
    manifest_path = root / "manifests" / "stage_b_sentinel_config_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
