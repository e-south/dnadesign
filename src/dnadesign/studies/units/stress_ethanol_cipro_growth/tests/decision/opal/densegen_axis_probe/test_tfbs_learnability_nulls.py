from __future__ import annotations

import pandas as pd
import pytest

from .probe_modules import probe_module

write_tfbs_null_artifacts = probe_module("tfbs.null_artifacts").write_tfbs_null_artifacts

_nulls = probe_module("tfbs.nulls")
TfbsNullConfig = _nulls.TfbsNullConfig
build_tfbs_family_content_matched_null = _nulls.build_tfbs_family_content_matched_null
build_tfbs_slot_geometry_count_matched_null = _nulls.build_tfbs_slot_geometry_count_matched_null

_oracle = probe_module("tfbs.oracle")
build_tfbs_learnability_oracle = _oracle.build_tfbs_learnability_oracle
validate_tfbs_label_algebra = _oracle.validate_tfbs_label_algebra

_schema = probe_module("tfbs.schema")
TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION = _schema.TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION
TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION = _schema.TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION

SEQ60 = "A" * 60


def test_family_content_null_preserves_joint_label_algebra_and_records_viability() -> None:
    labels = _content_labels()

    build = build_tfbs_family_content_matched_null(labels, seed=17, label_name="lexA_present")
    repeat = build_tfbs_family_content_matched_null(labels, seed=17, label_name="lexA_present")

    assert build.labels["id"].tolist() == labels["id"].tolist()
    assert build.labels["null_version"].unique().tolist() == [TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION]
    assert build.labels["null_recipe_hash"].nunique() == 1
    assert build.labels["null_seed"].unique().tolist() == [17]
    assert build.labels["lexA_present"].sum() == labels["lexA_present"].sum()
    assert build.labels["cpxR_or_baeR_count"].sum() == labels["cpxR_or_baeR_count"].sum()
    assert build.labels["lexA_present"].tolist() == repeat.labels["lexA_present"].tolist()
    assert build.null_viability_report["viability_status"] == "PASS"
    assert build.null_viability_report["label_marginal_before"] == build.null_viability_report["label_marginal_after"]
    assert build.null_viability_report["unchanged_label_fraction_after_permutation"] < 1.0
    validate_tfbs_label_algebra(build.labels)


def test_slot_geometry_null_preserves_counts_but_permutates_slot_labels() -> None:
    labels = _slot_labels()

    build = build_tfbs_slot_geometry_count_matched_null(labels, label_name="lexA_in_slot0", seed=29)

    assert build.labels["null_version"].unique().tolist() == [TFBS_LEARNABILITY_SLOT_GEOMETRY_NULL_VERSION]
    assert build.null_viability_report["label_name"] == "lexA_in_slot0"
    assert build.null_viability_report["viability_status"] == "PASS"
    assert build.null_viability_report["null_control_role"] == "count_preserving_slot_confound_control"
    assert build.null_viability_report["negative_control_claim_status"] == "CONFOUND_CONTROL_ONLY"
    assert "row-level TF family counts" in build.null_viability_report["preserved_signal"]
    assert build.labels[["lexA_count", "cpxR_count", "baeR_count"]].equals(
        labels[["lexA_count", "cpxR_count", "baeR_count"]]
    )
    assert build.labels["lexA_in_slot0"].sum() == labels["lexA_in_slot0"].sum()
    assert build.labels["lexA_in_slot0"].tolist() != labels["lexA_in_slot0"].tolist()
    assert (
        build.labels[["lexA_in_slot0", "lexA_in_slot1", "lexA_in_slot2"]].sum(axis=1) == build.labels["lexA_count"]
    ).all()
    assert (
        build.labels[["cpxR_or_baeR_in_slot0", "cpxR_or_baeR_in_slot1", "cpxR_or_baeR_in_slot2"]].sum(axis=1)
        == build.labels["cpxR_or_baeR_count"]
    ).all()


def test_family_content_null_fails_fast_when_declared_strata_are_not_exchangeable() -> None:
    labels = _content_labels().copy()
    labels["sigma35_variant"] = [f"unique-{idx}" for idx in range(len(labels))]
    config = TfbsNullConfig(
        fail_if_fraction_rows_in_singleton_strata_gt=0.0,
        fail_if_fraction_rows_in_tiny_strata_gt=0.0,
    )

    with pytest.raises(ValueError, match="exchangeability is too weak"):
        build_tfbs_family_content_matched_null(
            labels,
            seed=7,
            stratum_candidates=(("sigma35_variant", "spacer_length"),),
            config=config,
        )


def test_slot_geometry_null_rejects_non_slot_label_name() -> None:
    with pytest.raises(ValueError, match="slot-geometry null label_name"):
        build_tfbs_slot_geometry_count_matched_null(_slot_labels(), label_name="lexA_present", seed=7)


def test_write_tfbs_null_artifacts_records_table_hash_and_report(tmp_path) -> None:
    build = build_tfbs_family_content_matched_null(_content_labels(), seed=17, label_name="lexA_present")

    written = write_tfbs_null_artifacts(build, tmp_path)

    label_path = (
        tmp_path
        / "labels"
        / ("densegen_tfbs_learnability_family_content_matched_null_v1__lexA_present__seed17.parquet")
    )
    report_path = (
        tmp_path
        / "manifests"
        / ("densegen_tfbs_learnability_family_content_matched_null_v1__lexA_present__seed17.null_viability_report.json")
    )
    assert label_path.exists()
    assert report_path.exists()
    assert written.null_viability_report["null_label_table_hash"]
    assert written.null_viability_report["null_label_table_row_count"] == len(build.labels)
    assert pd.read_parquet(label_path)["null_version"].unique().tolist() == [
        TFBS_LEARNABILITY_FAMILY_CONTENT_NULL_VERSION
    ]


def _content_labels() -> pd.DataFrame:
    return build_tfbs_learnability_oracle(
        pd.DataFrame(
            {
                "id": [f"id-{idx}" for idx in range(6)],
                "sequence": [SEQ60] * 6,
                "densegen__used_tfbs_detail": [
                    _detail("LexA", "BaeR", "background"),
                    _detail("CpxR", "BaeR", "background"),
                    _detail("LexA", "LexA", "background"),
                    _detail("background", "BaeR", "CpxR"),
                    _detail("background", "background", "background"),
                    _detail("CpxR", "CpxR", "BaeR"),
                ],
            }
        )
    ).labels


def _slot_labels() -> pd.DataFrame:
    return build_tfbs_learnability_oracle(
        pd.DataFrame(
            {
                "id": [f"slot-{idx}" for idx in range(6)],
                "sequence": [SEQ60] * 6,
                "densegen__used_tfbs_detail": [
                    _detail("LexA", "BaeR", "background"),
                    _detail("BaeR", "LexA", "background"),
                    _detail("background", "LexA", "BaeR"),
                    _detail("LexA", "background", "BaeR"),
                    _detail("BaeR", "background", "LexA"),
                    _detail("background", "BaeR", "LexA"),
                ],
            }
        )
    ).labels


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
