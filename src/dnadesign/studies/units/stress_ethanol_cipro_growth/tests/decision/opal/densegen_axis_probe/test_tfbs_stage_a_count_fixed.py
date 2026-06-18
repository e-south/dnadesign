"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_a_count_fixed.py

Regression tests for TFBS stage a count fixed studies units stress.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .probe_modules import probe_module
from .stage_b_fixtures import write_tfbs_stage_b_source_fixture

_schema = probe_module("tfbs.schema")
TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_NULL_VERSION = (
    _schema.TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_NULL_VERSION
)
TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET = _schema.TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET

_profiles = probe_module("tfbs.profiles")
SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID = _profiles.SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID
SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID = _profiles.SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID

main = probe_module("cli").main


def test_tfbs_stage_a_cli_materializes_count_fixed_slot_position_sentinel_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    run_root = tmp_path / "cli-stage-a-count-fixed-slot-position-sentinel"

    assert (
        main(
            [
                "tfbs-stage-a",
                "--candidate-records",
                str(candidate_path),
                "--densegen-sidecar",
                str(sidecar_path),
                "--run-root",
                str(run_root),
                "--allow-custom-run-root",
                "--skip-live-label-rate-sanity",
                "--max-estimated-bytes",
                "1000000000",
                "--target-profile",
                SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID,
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    stage_manifest = _read_json(Path(payload["stage_a_manifest_path"]))
    pairing_manifest = _read_json(Path(payload["pairing_manifest_path"]))
    assert payload["null_artifact_count"] == len(TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET)
    assert stage_manifest["target_profile"]["profile_id"] == SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID
    assert pairing_manifest["target_profile"]["profile_id"] == SLOT_POSITION_COUNT_FIXED_SENTINEL_PROFILE_ID
    assert {row["null_version"] for row in stage_manifest["null_artifacts"]} == {
        TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_NULL_VERSION
    }
    assert {row["null_control_role"] for row in stage_manifest["null_artifacts"]} == {
        "count_fixed_shuffled_slot_negative_control"
    }
    assert {row["negative_control_claim_status"] for row in pairing_manifest["pairs"]} == {"VALID_AS_NEGATIVE_CONTROL"}
    assert {row["candidate_scope_policy_id"] for row in pairing_manifest["pairs"]} == {
        "tfbs_slot_position_target_count_eq_1_v1"
    }
    assert {row["required_count_value"] for row in pairing_manifest["pairs"]} == {1}


def test_tfbs_stage_a_cli_materializes_baer_middle_count_fixed_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = write_tfbs_stage_b_source_fixture(tmp_path)
    run_root = tmp_path / "cli-stage-a-count-fixed-baer-middle"

    assert (
        main(
            [
                "tfbs-stage-a",
                "--candidate-records",
                str(candidate_path),
                "--densegen-sidecar",
                str(sidecar_path),
                "--run-root",
                str(run_root),
                "--allow-custom-run-root",
                "--skip-live-label-rate-sanity",
                "--max-estimated-bytes",
                "1000000000",
                "--target-profile",
                SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID,
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    stage_manifest = _read_json(Path(payload["stage_a_manifest_path"]))
    pairing_manifest = _read_json(Path(payload["pairing_manifest_path"]))
    assert payload["null_artifact_count"] == 1
    assert stage_manifest["target_profile"]["profile_id"] == SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID
    assert pairing_manifest["target_profile"]["profile_id"] == SLOT_POSITION_COUNT_FIXED_BAER_MIDDLE_PROFILE_ID
    assert {row["label_name"] for row in stage_manifest["null_artifacts"]} == {"baeR_in_slot1"}
    assert {row["null_version"] for row in stage_manifest["null_artifacts"]} == {
        TFBS_LEARNABILITY_SLOT_POSITION_COUNT_FIXED_NULL_VERSION
    }
    assert {row["target_family_count_column"] for row in pairing_manifest["pairs"]} == {"baeR_count"}
    assert {row["negative_control_claim_status"] for row in pairing_manifest["pairs"]} == {"VALID_AS_NEGATIVE_CONTROL"}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
