"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_b_config_cli.py

Regression tests for TFBS stage b config CLI studies units stress.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .probe_modules import probe_module
from .stage_b_fixtures import write_tfbs_count_fixed_stage_b_source_fixture, write_tfbs_stage_b_source_fixture

_stage_a = probe_module("tfbs.stage_a.materialization")
TfbsStageAConfig = _stage_a.TfbsStageAConfig
materialize_tfbs_stage_a = _stage_a.materialize_tfbs_stage_a

_stage_b_configs = probe_module("tfbs.stage_b.configs")
TfbsStageBConfig = _stage_b_configs.TfbsStageBConfig
materialize_tfbs_stage_b_sentinel_configs = _stage_b_configs.materialize_tfbs_stage_b_sentinel_configs
main = probe_module("cli").main

_schema = probe_module("tfbs.schema")
TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET = _schema.TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET
TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET = _schema.TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET
_profiles = probe_module("tfbs.profiles")
SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID = _profiles.SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID
SLOT_POSITION_PROFILE_ID = _profiles.SLOT_POSITION_PROFILE_ID
SLOT_POSITION_SENTINEL_PROFILE_ID = _profiles.SLOT_POSITION_SENTINEL_PROFILE_ID


def test_tfbs_stage_b_cli_generates_configs_from_stage_a_run_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-cli"
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

    assert (
        main(
            [
                "tfbs-stage-b-configs",
                "--stage-a-run-root",
                str(stage_a_root),
                "--out-dir",
                str(stage_a_root / "stage_b_sentinel_configs"),
                "--no-validate-configs",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["campaign_count"] == 6
    assert payload["validation_status"] == "SKIPPED"
    assert Path(payload["config_manifest_path"]).exists()
    assert Path(payload["collection_manifest_path"]).exists()


def test_tfbs_stage_b_cli_can_generate_restricted_label_set(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-cli-labels"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
            label_names=("lexA_present",),
        )
    )

    assert (
        main(
            [
                "tfbs-stage-b-configs",
                "--stage-a-run-root",
                str(stage_a_root),
                "--out-dir",
                str(stage_a_root / "stage_b_sentinel_configs"),
                "--label-name",
                "lexA_present",
                "--no-validate-configs",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(payload["config_manifest_path"]).read_text(encoding="utf-8"))
    assert payload["campaign_count"] == 2
    assert manifest["target_profile"]["profile_id"] == "custom_tfbs_learnability_label_set"
    assert manifest["target_profile"]["canonical"] is False
    assert manifest["sentinel_labels"] == ["lexA_present"]
    assert {row["label_name"] for row in manifest["campaigns"]} == {"lexA_present"}


def test_tfbs_stage_b_cli_can_generate_named_slot_position_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-cli-slot-profile"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
            label_names=TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET,
        )
    )

    assert (
        main(
            [
                "tfbs-stage-b-configs",
                "--stage-a-run-root",
                str(stage_a_root),
                "--out-dir",
                str(stage_a_root / "stage_b_slot_position_configs"),
                "--target-profile",
                SLOT_POSITION_PROFILE_ID,
                "--no-validate-configs",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(payload["config_manifest_path"]).read_text(encoding="utf-8"))
    assert payload["campaign_count"] == 12
    assert manifest["target_profile"]["profile_id"] == SLOT_POSITION_PROFILE_ID
    assert manifest["target_profile"]["profile_role"] == "boundary_stage_b_probe"
    assert manifest["target_profile"]["canonical"] is False
    assert manifest["sentinel_labels"] == list(TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET)
    assert {row["label_family_id"] for row in manifest["campaigns"]} == {"tf_slot_family_presence"}


def test_tfbs_stage_b_cli_can_generate_named_slot_position_sentinel_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-cli-slot-sentinel-profile"
    materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=stage_a_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
            label_names=TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET,
            target_profile_id=SLOT_POSITION_SENTINEL_PROFILE_ID,
        )
    )

    assert (
        main(
            [
                "tfbs-stage-b-configs",
                "--stage-a-run-root",
                str(stage_a_root),
                "--out-dir",
                str(stage_a_root / "stage_b_slot_position_sentinel_configs"),
                "--target-profile",
                SLOT_POSITION_SENTINEL_PROFILE_ID,
                "--no-validate-configs",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(payload["config_manifest_path"]).read_text(encoding="utf-8"))
    assert payload["campaign_count"] == 4
    assert manifest["target_profile"]["profile_id"] == SLOT_POSITION_SENTINEL_PROFILE_ID
    assert manifest["target_profile"]["profile_role"] == "boundary_stage_b_sentinel_probe"
    assert manifest["target_profile"]["canonical"] is False
    assert manifest["sentinel_labels"] == list(TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET)
    assert {row["label_name"] for row in manifest["campaigns"]} == set(
        TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET
    )


def test_tfbs_stage_b_cli_can_generate_count_fixed_slot_position_sentinel_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_count_fixed_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-cli-count-fixed-slot-sentinel-profile"
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

    assert (
        main(
            [
                "tfbs-stage-b-configs",
                "--stage-a-run-root",
                str(stage_a_root),
                "--out-dir",
                str(stage_a_root / "stage_b_count_fixed_slot_position_sentinel_configs"),
                "--target-profile",
                SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID,
                "--no-validate-configs",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(payload["config_manifest_path"]).read_text(encoding="utf-8"))
    assert payload["campaign_count"] == 4
    assert manifest["target_profile"]["profile_id"] == SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID
    assert manifest["target_profile"]["profile_role"] == "boundary_stage_b_count_fixed_sentinel_probe"
    assert manifest["candidate_scope_mode"] == "label_specific_count_fixed"
    assert {row["candidate_scope_policy_id"] for row in manifest["candidate_scopes"]} == {
        "tfbs_slot_position_target_count_eq_1_v1"
    }


def test_tfbs_stage_b_exact_budget_requires_seed_batch_matching_selection_k(tmp_path: Path) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    stage_a_root = tmp_path / "stage-a-budget"
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

    with pytest.raises(ValueError, match="exact-budget acquisition requires initial_label_count == selection_k"):
        materialize_tfbs_stage_b_sentinel_configs(
            TfbsStageBConfig(
                stage_a_run_root=stage_a_root,
                out_dir=stage_a_root / "stage_b_sentinel_configs",
                initial_label_count=3,
                selection_k=6,
                validate_configs=False,
            )
        )
