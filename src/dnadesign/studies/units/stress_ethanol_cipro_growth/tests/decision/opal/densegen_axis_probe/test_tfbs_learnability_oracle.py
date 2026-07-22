"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_learnability_oracle.py

Regression tests for TFBS learnability oracle studies units stress ethanol cipro.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from .probe_modules import probe_module

tfbs_learnability_label_family_manifest = probe_module(
    "plan_logic.label_families"
).tfbs_learnability_label_family_manifest
write_tfbs_learnability_oracle_artifacts = probe_module("tfbs.manifests").write_tfbs_learnability_oracle_artifacts

_oracle = probe_module("tfbs.oracle")
build_tfbs_learnability_oracle = _oracle.build_tfbs_learnability_oracle
validate_tfbs_label_algebra = _oracle.validate_tfbs_label_algebra

_schema = probe_module("tfbs.schema")
TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES = _schema.TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES
TFBS_LEARNABILITY_ORACLE_VERSION = _schema.TFBS_LEARNABILITY_ORACLE_VERSION
TFBS_LEARNABILITY_REQUIRED_LABEL_COLUMNS = _schema.TFBS_LEARNABILITY_REQUIRED_LABEL_COLUMNS

SEQ60 = "A" * 60


def test_build_tfbs_learnability_oracle_writes_schema_manifests_and_sidecar_universe() -> None:
    build = build_tfbs_learnability_oracle(
        pd.DataFrame({"id": ["a", "b"], "sequence": [SEQ60, SEQ60]}),
        densegen_sidecar=pd.DataFrame(
            [
                {"id": "a", "densegen__used_tfbs_detail": _detail("LexA", "BaeR", "background")},
                {"id": "b", "densegen__used_tfbs_detail": _detail("CpxR", "BaeR", "background")},
                {"id": "sidecar-only", "densegen__used_tfbs_detail": _detail("LexA", "LexA", "background")},
            ]
        ),
    )

    assert list(build.labels.columns) == list(TFBS_LEARNABILITY_REQUIRED_LABEL_COLUMNS)
    assert build.labels["oracle_version"].unique().tolist() == [TFBS_LEARNABILITY_ORACLE_VERSION]
    assert build.row_universe_manifest["candidate_records_row_count"] == 2
    assert build.row_universe_manifest["densegen_sidecar_row_count"] == 3
    assert build.row_universe_manifest["candidate_sidecar_intersection_count"] == 2
    assert build.row_universe_manifest["sidecar_only_id_count"] == 1
    assert build.row_universe_manifest["active_row_count"] == 2
    assert build.label_manifest["active_label_families"] == list(TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES)
    assert build.label_manifest["algebraic_consistency_summary"]["status"] == "PASS"
    assert build.source_hash_manifest["x_column"]

    labels = build.labels.set_index("id")
    assert labels.loc["a", "lexA_present"] == 1
    assert labels.loc["a", "cpxR_or_baeR_count"] == 1
    assert labels.loc["b", "cpxR_or_baeR_count"] == 2
    assert labels.loc["b", "cpxR_or_baeR_count_fraction"] == pytest.approx(2 / 3)


def test_build_tfbs_learnability_oracle_rejects_candidate_only_sidecar_gap() -> None:
    candidates = pd.DataFrame({"id": ["a", "missing"], "sequence": [SEQ60, SEQ60]})
    sidecar = pd.DataFrame([{"id": "a", "densegen__used_tfbs_detail": _detail("LexA", "BaeR", "background")}])

    with pytest.raises(ValueError, match="missing required DenseGen sidecar metadata"):
        build_tfbs_learnability_oracle(candidates, densegen_sidecar=sidecar)


def test_build_tfbs_learnability_oracle_rejects_duplicate_ids() -> None:
    candidates = pd.DataFrame({"id": ["a", "a"], "sequence": [SEQ60, SEQ60]})

    with pytest.raises(ValueError, match="duplicate id"):
        build_tfbs_learnability_oracle(candidates)


def test_tfbs_learnability_label_algebra_fails_fast_on_drift() -> None:
    build = build_tfbs_learnability_oracle(_embedded_candidates())
    bad = build.labels.copy()
    bad.loc[0, "lexA_present"] = 0

    with pytest.raises(ValueError, match="label algebra failed"):
        validate_tfbs_label_algebra(bad)


def test_tfbs_learnability_label_family_manifest_is_v1_only() -> None:
    build = build_tfbs_learnability_oracle(_embedded_candidates())

    manifest = tfbs_learnability_label_family_manifest(build.labels)

    assert manifest["active_label_families"] == list(TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES)
    assert {row["label_family_id"] for row in manifest["families"]} == set(TFBS_LEARNABILITY_ACTIVE_LABEL_FAMILIES)
    assert "densegen_plan_logic4" not in {row["label_family_id"] for row in manifest["families"]}
    assert manifest["columns_missing"] == []


def test_write_tfbs_learnability_oracle_artifacts_records_replay_hashes(tmp_path) -> None:
    build = build_tfbs_learnability_oracle(_embedded_candidates())

    written = write_tfbs_learnability_oracle_artifacts(build, tmp_path)

    label_path = tmp_path / "labels" / "densegen_tfbs_learnability_positive_v1.parquet"
    assert label_path.exists()
    assert (tmp_path / "manifests" / "row_universe_manifest.json").exists()
    assert (tmp_path / "manifests" / "label_manifest.json").exists()
    assert (tmp_path / "manifests" / "source_hash_manifest.json").exists()
    assert written.label_manifest["label_table_hash"]
    assert written.label_manifest["row_universe_manifest_hash"]
    assert pd.read_parquet(label_path)["oracle_version"].unique().tolist() == [TFBS_LEARNABILITY_ORACLE_VERSION]


def _embedded_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a"],
            "sequence": [SEQ60],
            "densegen__used_tfbs_detail": [_detail("LexA", "BaeR", "background")],
        }
    )


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
