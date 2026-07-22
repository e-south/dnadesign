"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_a_materialization.py

Regression tests for TFBS stage a materialization studies units stress ethanol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from .probe_modules import probe_module

_schema = probe_module("tfbs.schema")
TFBS_LEARNABILITY_ORACLE_VERSION = _schema.TFBS_LEARNABILITY_ORACLE_VERSION
TFBS_LEARNABILITY_SENTINEL_TARGET_SET = _schema.TFBS_LEARNABILITY_SENTINEL_TARGET_SET
TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET = _schema.TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET
TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET = _schema.TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET
_profiles = probe_module("tfbs.profiles")
SLOT_POSITION_PROFILE_ID = _profiles.SLOT_POSITION_PROFILE_ID
SLOT_POSITION_SENTINEL_PROFILE_ID = _profiles.SLOT_POSITION_SENTINEL_PROFILE_ID

_stage_a = probe_module("tfbs.stage_a.materialization")
TfbsStageAConfig = _stage_a.TfbsStageAConfig
main = probe_module("cli").main
materialize_tfbs_stage_a = _stage_a.materialize_tfbs_stage_a

SEQ60 = "A" * 60


def test_tfbs_stage_a_materializes_positive_labels_sentinel_nulls_and_manifests(tmp_path: Path) -> None:
    candidate_path, sidecar_path = _write_sources(tmp_path)
    run_root = tmp_path / "stage-a"

    result = materialize_tfbs_stage_a(
        TfbsStageAConfig(
            candidate_records_path=candidate_path,
            densegen_sidecar_path=sidecar_path,
            run_root=run_root,
            seed=7,
            max_estimated_bytes=1_000_000_000,
            enforce_live_label_rate_sanity=False,
        )
    )

    positive_label_path = run_root / "labels" / f"{TFBS_LEARNABILITY_ORACLE_VERSION}.parquet"
    assert positive_label_path.exists()
    assert result.positive_label_table_path == positive_label_path
    assert pd.read_parquet(positive_label_path)["id"].tolist() == [f"id-{idx}" for idx in range(6)]

    manifests_dir = run_root / "manifests"
    row_universe = _read_json(manifests_dir / "row_universe_manifest.json")
    label_manifest = _read_json(manifests_dir / "label_manifest.json")
    source_hash = _read_json(manifests_dir / "source_hash_manifest.json")
    stage_manifest = _read_json(manifests_dir / "tfbs_stage_a_manifest.json")
    pairing_manifest = _read_json(manifests_dir / "pairing_manifest.json")
    retention_estimate = _read_json(manifests_dir / "retention_estimate.json")

    assert row_universe["candidate_records_path"] == str(candidate_path)
    assert row_universe["candidate_records_hash"] == _sha256(candidate_path)
    assert row_universe["densegen_sidecar_path"] == str(sidecar_path)
    assert row_universe["densegen_sidecar_hash"] == _sha256(sidecar_path)
    assert label_manifest["label_table_hash"]
    assert source_hash["source_records_path_hash_row_schema"]["path"] == str(candidate_path)
    assert source_hash["source_records_path_hash_row_schema"]["hash"] == _sha256(candidate_path)
    assert source_hash["densegen_sidecar_path_hash_row_schema"]["path"] == str(sidecar_path)
    assert source_hash["densegen_sidecar_path_hash_row_schema"]["hash"] == _sha256(sidecar_path)

    assert stage_manifest["stage"] == "A"
    assert stage_manifest["status"] == "PASS"
    assert stage_manifest["null_permutation_seed"] == 7
    assert stage_manifest["null_permutation_seed_context"] == "tfbs_stage_a_matched_null_permutation_v1:seed=7"
    assert stage_manifest["positive_oracle_version"] == TFBS_LEARNABILITY_ORACLE_VERSION
    assert stage_manifest["target_profile"]["profile_id"] == "tfbs_count_fraction_probe_v1"
    assert stage_manifest["target_profile"]["canonical"] is True
    assert stage_manifest["target_profile"]["label_names"] == list(TFBS_LEARNABILITY_SENTINEL_TARGET_SET)
    assert set(stage_manifest["sentinel_labels"]) == set(TFBS_LEARNABILITY_SENTINEL_TARGET_SET)
    assert len(stage_manifest["null_artifacts"]) == len(TFBS_LEARNABILITY_SENTINEL_TARGET_SET)
    assert sorted(row["label_name"] for row in stage_manifest["null_artifacts"]) == sorted(
        TFBS_LEARNABILITY_SENTINEL_TARGET_SET
    )
    assert all(Path(row["null_label_table_path"]).exists() for row in stage_manifest["null_artifacts"])
    assert all(Path(row["null_viability_report_path"]).exists() for row in stage_manifest["null_artifacts"])

    assert pairing_manifest["schema_version"].endswith(".pairing_manifest")
    assert pairing_manifest["null_permutation_seed"] == 7
    assert pairing_manifest["null_permutation_seed_context"] == "tfbs_stage_a_matched_null_permutation_v1:seed=7"
    assert pairing_manifest["target_profile"]["profile_id"] == "tfbs_count_fraction_probe_v1"
    assert len(pairing_manifest["pairs"]) == len(TFBS_LEARNABILITY_SENTINEL_TARGET_SET)
    assert {row["label_name"] for row in pairing_manifest["pairs"]} == set(TFBS_LEARNABILITY_SENTINEL_TARGET_SET)
    assert all(row["positive_oracle_version"] == TFBS_LEARNABILITY_ORACLE_VERSION for row in pairing_manifest["pairs"])
    assert all(row["null_permutation_seed"] == 7 for row in pairing_manifest["pairs"])
    assert all(
        row["retention_policy_hash"] == retention_estimate["retention_policy_hash"] for row in pairing_manifest["pairs"]
    )

    assert retention_estimate["status"] == "PASS"
    assert retention_estimate["max_estimated_bytes"] == 1_000_000_000
    assert retention_estimate["estimates"]["sentinel_initial"]["planned_campaign_count"] == 6
    assert retention_estimate["estimates"]["full_matrix"]["planned_campaign_count"] == 144


def test_tfbs_stage_a_fails_closed_when_retention_budget_is_exceeded(tmp_path: Path) -> None:
    candidate_path, sidecar_path = _write_sources(tmp_path)
    run_root = tmp_path / "too-large"

    with pytest.raises(ValueError, match="retention estimate exceeds configured budget"):
        materialize_tfbs_stage_a(
            TfbsStageAConfig(
                candidate_records_path=candidate_path,
                densegen_sidecar_path=sidecar_path,
                run_root=run_root,
                seed=7,
                max_estimated_bytes=1,
                enforce_live_label_rate_sanity=False,
            )
        )

    assert not run_root.exists()


def test_tfbs_stage_a_cli_materializes_from_explicit_sources(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = _write_sources(tmp_path)
    run_root = tmp_path / "cli-stage-a"

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
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert Path(payload["positive_label_table_path"]).exists()
    assert Path(payload["stage_a_manifest_path"]).exists()
    assert Path(payload["retention_estimate_path"]).exists()


def test_tfbs_stage_a_cli_materializes_named_slot_position_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = _write_sources(tmp_path)
    run_root = tmp_path / "cli-stage-a-slot-position"

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
                SLOT_POSITION_PROFILE_ID,
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    stage_manifest = _read_json(Path(payload["stage_a_manifest_path"]))
    pairing_manifest = _read_json(Path(payload["pairing_manifest_path"]))
    assert payload["null_artifact_count"] == len(TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET)
    assert stage_manifest["target_profile"]["profile_id"] == SLOT_POSITION_PROFILE_ID
    assert stage_manifest["sentinel_labels"] == list(TFBS_LEARNABILITY_SLOT_POSITION_TARGET_SET)
    assert pairing_manifest["target_profile"]["profile_id"] == SLOT_POSITION_PROFILE_ID
    assert {row["null_control_role"] for row in stage_manifest["null_artifacts"]} == {
        "count_preserving_slot_confound_control"
    }


def test_tfbs_stage_a_cli_materializes_named_slot_position_sentinel_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_path, sidecar_path = _write_sources(tmp_path)
    run_root = tmp_path / "cli-stage-a-slot-position-sentinel"

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
                SLOT_POSITION_SENTINEL_PROFILE_ID,
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    stage_manifest = _read_json(Path(payload["stage_a_manifest_path"]))
    pairing_manifest = _read_json(Path(payload["pairing_manifest_path"]))
    assert payload["null_artifact_count"] == len(TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET)
    assert stage_manifest["target_profile"]["profile_id"] == SLOT_POSITION_SENTINEL_PROFILE_ID
    assert stage_manifest["sentinel_labels"] == list(TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET)
    assert pairing_manifest["target_profile"]["profile_id"] == SLOT_POSITION_SENTINEL_PROFILE_ID
    assert {row["label_name"] for row in stage_manifest["null_artifacts"]} == set(
        TFBS_LEARNABILITY_SLOT_POSITION_SENTINEL_TARGET_SET
    )


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    candidate_path = tmp_path / "records.parquet"
    sidecar_path = tmp_path / "densegen.parquet"
    pd.DataFrame({"id": [f"id-{idx}" for idx in range(6)], "sequence": [SEQ60] * 6}).to_parquet(
        candidate_path,
        index=False,
    )
    pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(6)] + ["sidecar-only"],
            "densegen__used_tfbs_detail": [
                _detail("LexA", "BaeR", "background"),
                _detail("BaeR", "LexA", "background"),
                _detail("background", "LexA", "BaeR"),
                _detail("LexA", "background", "BaeR"),
                _detail("BaeR", "background", "LexA"),
                _detail("background", "BaeR", "LexA"),
                _detail("CpxR", "BaeR", "background"),
            ],
        }
    ).to_parquet(sidecar_path, index=False)
    return candidate_path, sidecar_path


def _detail(slot0: str, slot1: str, slot2: str) -> list[dict[str, object]]:
    return [
        _tfbs(slot0, 10),
        _tfbs(slot1, 21),
        _tfbs(slot2, 32),
        _fixed("upstream_sigma70_core", 0, variant_id="f"),
        _fixed("downstream_sigma70_core", 22, sequence="TATAAT"),
    ]


def _tfbs(regulator: str, offset_raw: int) -> dict[str, object]:
    return {
        "part_kind": "tfbs",
        "regulator": regulator,
        "offset_raw": offset_raw,
        "length": 6,
        "end_raw": offset_raw + 6,
    }


def _fixed(
    role: str,
    offset_raw: int,
    *,
    variant_id: str | None = None,
    sequence: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "part_kind": "fixed_element",
        "role": role,
        "offset_raw": offset_raw,
        "length": 6,
        "end_raw": offset_raw + 6,
        "spacer_length": 16,
    }
    if variant_id is not None:
        row["variant_id"] = variant_id
    if sequence is not None:
        row["sequence"] = sequence
    return row


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
