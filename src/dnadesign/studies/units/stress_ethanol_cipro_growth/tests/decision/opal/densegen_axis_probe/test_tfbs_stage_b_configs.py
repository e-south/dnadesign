from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from .probe_modules import probe_module
from .stage_b_fixtures import write_tfbs_stage_b_source_fixture

X_COLUMN = probe_module("core.constants").X_COLUMN
TFBS_LEARNABILITY_SENTINEL_TARGET_SET = probe_module("tfbs.schema").TFBS_LEARNABILITY_SENTINEL_TARGET_SET

_stage_a = probe_module("tfbs.stage_a.materialization")
TfbsStageAConfig = _stage_a.TfbsStageAConfig
materialize_tfbs_stage_a = _stage_a.materialize_tfbs_stage_a

_stage_b_configs = probe_module("tfbs.stage_b.configs")
TfbsStageBConfig = _stage_b_configs.TfbsStageBConfig
materialize_tfbs_stage_b_sentinel_configs = _stage_b_configs.materialize_tfbs_stage_b_sentinel_configs

TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM = probe_module(
    "tfbs.stage_b.seed"
).TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM


def test_tfbs_stage_b_generates_sentinel_configs_from_stage_a_artifacts(tmp_path: Path) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
        )
    )

    result = materialize_tfbs_stage_b_sentinel_configs(
        TfbsStageBConfig(
            stage_a_run_root=stage_a_root,
            out_dir=stage_a_root / "stage_b_sentinel_configs",
            validate_configs=False,
        )
    )

    assert result.status == "PASS"
    assert result.campaign_count == 10
    assert result.validation_status == "SKIPPED"
    assert result.config_manifest_path.exists()
    assert result.collection_manifest_path.exists()

    manifest = _read_json(result.config_manifest_path)
    assert manifest["schema_version"].endswith(".stage_b_sentinel_config_manifest")
    assert manifest["status"] == "PASS"
    assert manifest["stage"] == "B"
    assert manifest["stage_a_manifest_path"] == str(stage_a_root / "manifests" / "tfbs_stage_a_manifest.json")
    assert manifest["pairing_manifest_path"] == str(stage_a_root / "manifests" / "pairing_manifest.json")
    assert manifest["collection_manifest_path"] == str(result.collection_manifest_path)
    assert manifest["collection_manifest_hash"] == _sha256(result.collection_manifest_path)
    assert manifest["campaign_count"] == 10
    assert manifest["initial_seed_policy"] == TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM
    assert manifest["sentinel_labels"] == list(TFBS_LEARNABILITY_SENTINEL_TARGET_SET)
    assert manifest["validation"]["status"] == "SKIPPED"
    assert len(manifest["campaigns"]) == 10
    assert len(manifest["pairs"]) == 5

    roles_by_label: dict[str, set[str]] = {}
    for campaign in manifest["campaigns"]:
        roles_by_label.setdefault(campaign["label_name"], set()).add(campaign["oracle_role"])
        config_path = Path(campaign["config_path"])
        initial_label_input_path = Path(campaign["initial_label_input_path"])
        candidate_scope_path = Path(campaign["candidate_scope_path"])

        assert config_path.exists()
        assert initial_label_input_path.exists()
        assert candidate_scope_path.exists()
        assert campaign["campaign_config_hash"] == _sha256(config_path)
        assert campaign["initial_label_input_hash"] == _sha256(initial_label_input_path)
        assert campaign["candidate_scope_hash"] == _sha256(candidate_scope_path)
        assert "--in" in campaign["ingest_round0_command"]
        assert "--labels" not in campaign["ingest_round0_command"]
        assert "--apply" in campaign["ingest_round0_command"]

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        label_name = campaign["label_name"]
        role = campaign["oracle_role"]
        assert cfg["campaign"]["metadata"]["probe_family"] == "densegen_tfbs_learnability_probe_v1"
        assert cfg["campaign"]["metadata"]["probe_stage"] == "B"
        assert cfg["campaign"]["metadata"]["label_name"] == label_name
        assert cfg["campaign"]["metadata"]["oracle_role"] == role
        assert cfg["campaign"]["metadata"]["target"] == label_name
        assert cfg["campaign"]["metadata"]["target_label"]
        assert cfg["campaign"]["metadata"]["label_oracle_kind"] == ("positive" if role == "positive" else "null")
        assert cfg["campaign"]["metadata"]["label_split_id"] == "random_id"
        assert cfg["campaign"]["metadata"]["split_id"] == "random_id"
        assert cfg["campaign"]["metadata"]["seed"] == 7
        assert cfg["campaign"]["metadata"]["retention_mode"] == "production_review"
        assert cfg["campaign"]["metadata"]["rounds"] == 24
        assert cfg["campaign"]["metadata"]["selection_k"] == 6
        assert (
            cfg["campaign"]["metadata"]["initial_seed_policy"]
            == TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM
        )
        assert cfg["campaign"]["metadata"]["selection_tie_handling"] == "ordinal"
        assert cfg["campaign"]["metadata"]["selection_budget_mode"] == "exact_top_k"
        assert cfg["data"]["x_column_name"] == X_COLUMN
        assert cfg["data"]["y_expected_length"] == 1
        assert cfg["data"]["candidate_scope"]["kind"] == "id_list"
        assert cfg["labels"]["source"]["kind"] == "usr_sidecar"
        assert cfg["labels"]["source"]["dataset"] == cfg["data"]["location"]["dataset"]
        assert not Path(cfg["labels"]["source"]["path"]).is_absolute()
        assert cfg["labels"]["y_space"] == "numeric_vector"
        assert cfg["transforms_y"]["name"] == "vector_from_table_v1"
        assert cfg["transforms_y"]["params"]["value_columns"] == [label_name]
        assert cfg["objectives"] == [
            {
                "name": "vector_channel_v1",
                "params": {"channel_index": 0, "channel_name": label_name, "mode": "maximize"},
            }
        ]
        assert cfg["selection"]["params"]["score_ref"] == f"vector_channel_v1/{label_name}"
        assert cfg["selection"]["params"]["objective_mode"] == "maximize"
        assert cfg["selection"]["params"]["top_k"] == 6
        assert cfg["selection"]["params"]["tie_handling"] == "ordinal"
        assert cfg["writeback"]["prediction_records"] == "ledger_only"
        assert cfg["artifact_retention"] == {
            "mode": "production_review",
            "prediction_ledger": "latest_full_plus_selected_history",
            "plot_tidy_data": "compact",
            "model_artifacts": "latest",
            "tabular_format": "parquet_zstd",
            "max_estimated_bytes": 1_000_000_000,
            "fail_if_estimate_exceeds": True,
            "final_round": 23,
        }
        assert cfg.get("training", {}).get("y_ops") in (None, [])
        plot_cfg = yaml.safe_load((config_path.parent / "plots.yaml").read_text(encoding="utf-8"))
        assert plot_cfg["plot_defaults"]["output"]["save_data"] is True

        labels = pd.read_parquet(initial_label_input_path)
        assert labels.columns.tolist() == ["id", "sequence", label_name]
        assert len(labels) == 6

    assert roles_by_label == {label: {"positive", "matched_null"} for label in TFBS_LEARNABILITY_SENTINEL_TARGET_SET}

    collection = _read_json(result.collection_manifest_path)
    assert collection["schema_version"] == "opal.campaign_collection.v2"
    assert collection["collection_id"] == "densegen_tfbs_stage_b_exact_budget_random_id_seed7"
    assert [row["id"] for row in collection["dimensions"]] == [
        "target",
        "label_oracle_kind",
        "label_family_id",
        "label_split_id",
        "seed",
    ]
    assert collection["relationships"] == [
        {
            "id": "positive_vs_null",
            "kind": "control_pair",
            "label": "Positive vs matched-null oracle control",
            "role_dimension": "label_oracle_kind",
            "left_role": "positive",
            "right_role": "null",
            "match_on": ["target", "label_family_id", "label_split_id", "seed"],
            "replicate_on": ["seed"],
        }
    ]
    assert collection["comparison_views"] == []

    for pair in manifest["pairs"]:
        assert pair["label_name"] in TFBS_LEARNABILITY_SENTINEL_TARGET_SET
        assert pair["initial_seed_policy"] == TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM
        assert Path(pair["positive_campaign_config_path"]).exists()
        assert Path(pair["null_campaign_config_path"]).exists()
        assert pair["positive_campaign_config_hash"] == _sha256(Path(pair["positive_campaign_config_path"]))
        assert pair["null_campaign_config_hash"] == _sha256(Path(pair["null_campaign_config_path"]))


def test_tfbs_stage_b_fails_closed_when_stage_a_retention_failed(tmp_path: Path) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-failed-retention"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
        )
    )
    retention_path = stage_a_root / "manifests" / "retention_estimate.json"
    retention = _read_json(retention_path)
    retention["status"] = "FAIL"
    retention_path.write_text(json.dumps(retention, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Stage A retention status must be PASS"):
        materialize_tfbs_stage_b_sentinel_configs(
            TfbsStageBConfig(
                stage_a_run_root=stage_a_root,
                out_dir=stage_a_root / "stage_b_sentinel_configs",
                validate_configs=False,
            )
        )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
