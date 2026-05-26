from __future__ import annotations

from .helpers import (
    AXIS_CLASS_TO_LOGIC4,
    _detail,
    build_axis_oracle,
    label_family_manifest,
    make_permuted_labels,
    null_provenance_payload,
    pd,
    require_label_family_columns,
    suite_manifest_payload,
)


def test_label_family_manifest_records_active_and_passive_readouts() -> None:
    labels = build_axis_oracle(
        pd.DataFrame(
            [
                {
                    "id": "a",
                    "sequence": "AAAA",
                    "densegen__used_tfbs_detail": _detail("lexA_CTGTATAWAWWHACA", "cpxR"),
                    "densegen__plan": "ethanol_ciprofloxacin__sig35=f",
                }
            ]
        )
    )

    require_label_family_columns(
        labels,
        ["densegen_plan_logic4", "tf_family_presence", "tf_family_count", "densegen_plan_class"],
    )
    manifest = label_family_manifest(labels)

    assert manifest["active_label_family"] == "densegen_plan_logic4"
    assert manifest["active_label_families"] == ["densegen_plan_logic4", "tf_family_count"]
    assert manifest["passive_label_families"] == [
        "tf_family_presence",
        "densegen_plan_class",
    ]
    assert manifest["summaries"]["tf_family_presence"]["column_sums"]["tf_family__lexA__presence"] == 1
    assert manifest["summaries"]["tf_family_count"]["column_sums"]["tf_family__cpxR__count"] == 1
    assert manifest["summaries"]["densegen_plan_class"]["value_counts"]["ethanol_ciprofloxacin"] == 1


def test_null_provenance_records_seed_universe_and_balance() -> None:
    axis_classes = ["background_only", "ethanol_only", "cipro_only", "dual_axis_and"] * 2
    labels = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(8)],
            "logic4": [AXIS_CLASS_TO_LOGIC4[axis_class] for axis_class in axis_classes],
            "axis_class": axis_classes,
            "quality_flag": ["ok"] * 8,
        }
    )
    null_labels = make_permuted_labels(labels, seed=17)

    provenance = null_provenance_payload(labels, null_labels, seed=17)

    assert provenance["strategy"] == "global_quality_ok_permutation"
    assert provenance["seed"] == 17
    assert provenance["permutation_universe"]["row_count"] == 8
    assert provenance["class_balance_before"] == provenance["class_balance_after"]
    assert provenance["unchanged_assignment_count"] < 8


def test_default_suite_manifest_is_k12_three_seed_and_study_owned() -> None:
    manifest = suite_manifest_payload()

    assert manifest["suite_id"] == "densegen_motif_qa_k12_s3_v1"
    assert manifest["selection_k"] == 12
    assert manifest["initial_label_count"] == 12
    assert manifest["seeds"] == [7, 17, 29]
    assert manifest["active_label_family"] == "densegen_plan_logic4"
    assert manifest["active_label_families"] == ["densegen_plan_logic4", "tf_family_count"]
    assert manifest["passive_label_families"] == [
        "tf_family_presence",
        "densegen_plan_class",
    ]
    assert "OPAL notebooks" in manifest["notebook_boundary"]
