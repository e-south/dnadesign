from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from .probe_modules import probe_module
from .stage_b_fixtures import write_tfbs_count_fixed_stage_b_source_fixture, write_tfbs_stage_b_source_fixture

X_COLUMN = probe_module("core.constants").X_COLUMN
TFBS_LEARNABILITY_SENTINEL_TARGET_SET = probe_module("tfbs.schema").TFBS_LEARNABILITY_SENTINEL_TARGET_SET
tfbs_label_title = probe_module("tfbs.label_text").tfbs_label_title
_profiles = probe_module("tfbs.profiles")
SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID = _profiles.SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID
SLOT_POSITION_COUNT_FIXED_SENTINEL_LABELS = _profiles.slot_position_count_fixed_sentinel_label_names()

_stage_a = probe_module("tfbs.stage_a.materialization")
TfbsStageAConfig = _stage_a.TfbsStageAConfig
materialize_tfbs_stage_a = _stage_a.materialize_tfbs_stage_a

_stage_b_configs = probe_module("tfbs.stage_b.configs")
TfbsStageBConfig = _stage_b_configs.TfbsStageBConfig
materialize_tfbs_stage_b_sentinel_configs = _stage_b_configs.materialize_tfbs_stage_b_sentinel_configs
prepare_stage_b_out_dir = probe_module("tfbs.stage_b.io").prepare_stage_b_out_dir

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
    assert result.campaign_count == 6
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
    assert manifest["null_permutation_seed"] == 7
    assert manifest["null_permutation_seed_context"] == "tfbs_stage_a_matched_null_permutation_v1:seed=7"
    assert manifest["campaign_count"] == 6
    assert manifest["initial_seed_policy"] == TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM
    assert manifest["target_profile"]["profile_id"] == "tfbs_count_fraction_probe_v1"
    assert manifest["target_profile"]["canonical"] is True
    assert manifest["target_profile"]["label_names"] == list(TFBS_LEARNABILITY_SENTINEL_TARGET_SET)
    assert manifest["sentinel_labels"] == list(TFBS_LEARNABILITY_SENTINEL_TARGET_SET)
    assert manifest["validation"]["status"] == "SKIPPED"
    assert len(manifest["campaigns"]) == 6
    assert len(manifest["pairs"]) == 3

    roles_by_label: dict[str, set[str]] = {}
    initial_ids_by_label_role: dict[tuple[str, str], list[str]] = {}
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
        target_label = tfbs_label_title(label_name)
        role_label = "DenseGen label" if role == "positive" else "matched scrambled-label control"
        assert cfg["campaign"]["name"] == f"DenseGen TFBS learnability: {target_label}, {role_label}, seed 7"
        assert label_name not in cfg["campaign"]["name"]
        assert cfg["campaign"]["metadata"]["probe_family"] == "densegen_tfbs_learnability_probe_v1"
        assert cfg["campaign"]["metadata"]["probe_stage"] == "B"
        assert cfg["campaign"]["metadata"]["label_name"] == label_name
        assert cfg["campaign"]["metadata"]["oracle_role"] == role
        assert cfg["campaign"]["metadata"]["target"] == label_name
        assert cfg["campaign"]["metadata"]["target_label"] == target_label
        assert cfg["campaign"]["metadata"]["target_dropdown_label"].endswith("(count / 3)")
        assert cfg["campaign"]["metadata"]["label_oracle_kind"] == ("positive" if role == "positive" else "null")
        assert cfg["campaign"]["metadata"]["label_split_id"] == "random_id"
        assert cfg["campaign"]["metadata"]["split_id"] == "random_id"
        assert cfg["campaign"]["metadata"]["seed"] == 7
        assert cfg["campaign"]["metadata"]["retention_mode"] == "production_review"
        assert cfg["campaign"]["metadata"]["rounds"] == 24
        assert cfg["campaign"]["metadata"]["selection_k"] == 6
        assert cfg["campaign"]["metadata"]["baserender_metadata_records_path"] == str(sidecar_path)
        assert cfg["campaign"]["metadata"]["baserender_metadata_source"] == "densegen_sidecar"
        assert (
            cfg["campaign"]["metadata"]["initial_seed_policy"]
            == TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM
        )
        assert cfg["campaign"]["metadata"]["initial_seed_context"].startswith("tfbs_stage_b_shared_initial_seed_v1:")
        assert cfg["campaign"]["metadata"]["initial_seed_source_role"] == "positive"
        assert cfg["campaign"]["metadata"]["initial_seed_pairing"] == "shared_positive_null_starting_ids"
        assert cfg["campaign"]["metadata"]["replicate_dimension"] == "seed"
        assert cfg["campaign"]["metadata"]["replicate_seed"] == 7
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
        assert plot_cfg["plots"][0]["params"]["title"] == (f"Predicted selected label value by round: {target_label}")
        assert plot_cfg["plots"][0]["params"]["surface_label"] == (f"Predicted selected label value: {target_label}")
        assert label_name not in plot_cfg["plots"][0]["params"]["title"]

        labels = pd.read_parquet(initial_label_input_path)
        assert labels.columns.tolist() == ["id", "sequence", label_name]
        assert len(labels) == 6
        initial_ids_by_label_role[(label_name, role)] = labels["id"].astype(str).tolist()

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
    assert [row["label"] for row in collection["dimensions"]] == [
        "TFBS label",
        "Label source",
        "Label family",
        "Split",
        "Seed",
    ]
    assert collection["relationships"] == [
        {
            "id": "positive_vs_null",
            "kind": "control_pair",
            "label": "DenseGen label vs matched scrambled-label control",
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
        assert pair["initial_seed_context"].startswith("tfbs_stage_b_shared_initial_seed_v1:")
        assert pair["initial_seed_source_role"] == "positive"
        assert pair["initial_seed_pairing"] == "shared_positive_null_starting_ids"
        assert pair["null_permutation_seed"] == 7
        assert pair["null_permutation_seed_context"] == "tfbs_stage_a_matched_null_permutation_v1:seed=7"
        assert pair["initial_label_ids_hash"]
        assert Path(pair["positive_campaign_config_path"]).exists()
        assert Path(pair["null_campaign_config_path"]).exists()
        assert pair["positive_campaign_config_hash"] == _sha256(Path(pair["positive_campaign_config_path"]))
        assert pair["null_campaign_config_hash"] == _sha256(Path(pair["null_campaign_config_path"]))
        assert (
            initial_ids_by_label_role[(pair["label_name"], "positive")]
            == initial_ids_by_label_role[(pair["label_name"], "matched_null")]
        )


def test_tfbs_stage_b_count_fixed_profile_writes_label_specific_scopes(tmp_path: Path) -> None:
    candidate_path, sidecar_path = write_tfbs_count_fixed_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-count-fixed"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
            target_profile_id=SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID,
        )
    )

    result = materialize_tfbs_stage_b_sentinel_configs(
        TfbsStageBConfig(
            stage_a_run_root=stage_a_root,
            out_dir=stage_a_root / "stage_b_count_fixed_slot_position_configs",
            label_names=SLOT_POSITION_COUNT_FIXED_SENTINEL_LABELS,
            target_profile_id=SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID,
            validate_configs=False,
        )
    )

    manifest = _read_json(result.config_manifest_path)
    assert result.campaign_count == 4
    assert manifest["target_profile"]["profile_id"] == SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID
    assert manifest["candidate_scope_mode"] == "label_specific_count_fixed"
    assert manifest["collection_manifest_path"] == str(result.collection_manifest_path)
    collection = _read_json(result.collection_manifest_path)
    assert collection["relationships"][0]["label"] == "DenseGen label vs count-fixed shuffled-slot control"
    assert {row["label_name"] for row in manifest["candidate_scopes"]} == set(SLOT_POSITION_COUNT_FIXED_SENTINEL_LABELS)
    assert {row["candidate_scope_policy_id"] for row in manifest["candidate_scopes"]} == {
        "tfbs_slot_position_target_count_eq_1_v1"
    }

    by_label_role = {(row["label_name"], row["oracle_role"]): row for row in manifest["campaigns"]}
    for label_name in SLOT_POSITION_COUNT_FIXED_SENTINEL_LABELS:
        positive = by_label_role[(label_name, "positive")]
        control = by_label_role[(label_name, "matched_null")]
        assert positive["candidate_scope_path"] == control["candidate_scope_path"]
        assert positive["candidate_scope_hash"] == control["candidate_scope_hash"]
        assert positive["initial_label_ids_hash"] == control["initial_label_ids_hash"]
        assert positive["candidate_scope_policy_id"] == "tfbs_slot_position_target_count_eq_1_v1"
        count_column = positive["target_family_count_column"]
        positive_labels = pd.read_parquet(positive["label_table_path"])
        control_labels = pd.read_parquet(control["label_table_path"])
        scope_ids = pd.read_parquet(positive["candidate_scope_path"])["id"].astype(str).tolist()
        assert len(scope_ids) == 6
        assert positive_labels["id"].astype(str).tolist() == scope_ids
        assert control_labels["id"].astype(str).tolist() == scope_ids
        assert (positive_labels[count_column] == 1).all()
        assert (control_labels[count_column] == 1).all()
        assert positive_labels[label_name].value_counts().sort_index().to_dict() == (
            control_labels[label_name].value_counts().sort_index().to_dict()
        )
        cfg = yaml.safe_load(Path(control["config_path"]).read_text(encoding="utf-8"))
        assert cfg["campaign"]["metadata"]["candidate_scope_policy_id"] == "tfbs_slot_position_target_count_eq_1_v1"
        assert cfg["data"]["candidate_scope"]["path"] == str(Path(control["candidate_scope_path"]).resolve())
        assert "count-fixed shuffled-slot control" in cfg["campaign"]["name"]


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


def test_stage_b_out_dir_refresh_mode_preserves_existing_execution_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "stage_b"
    marker = out_dir / "campaigns" / "tfbs_example" / "outputs" / "rounds" / "round_0" / "selection.csv"
    marker.parent.mkdir(parents=True)
    marker.write_text("id\nexample\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refuses to overwrite execution state"):
        prepare_stage_b_out_dir(out_dir, replace=False)

    prepare_stage_b_out_dir(out_dir, replace=False, refresh_existing_execution_state=True)

    assert marker.read_text(encoding="utf-8") == "id\nexample\n"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
