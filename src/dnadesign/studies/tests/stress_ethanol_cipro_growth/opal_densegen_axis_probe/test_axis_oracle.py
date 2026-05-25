from __future__ import annotations

from .helpers import (
    AXIS_CLASS_TO_LOGIC4,
    _detail,
    build_axis_oracle,
    build_train_ids,
    class_from_logic4,
    derive_axis_label,
    make_permuted_labels,
    parse_sigma35_variant,
    pd,
    predicted_axis_classes,
    pytest,
)


@pytest.mark.parametrize(
    ("detail", "expected_class", "expected_vec8"),
    [
        (_detail("background", "background", "background"), "background_only", [0, 0, 0, 0, 0, 0, 0, 0]),
        (_detail("cpxR", "background", "background"), "ethanol_only", [0, 1, 0, 1, 0, 1, 0, 1]),
        (_detail("lexA_CTGTATAWAWWHACA", "background", "background"), "cipro_only", [0, 0, 1, 1, 0, 0, 1, 1]),
        (_detail("baeR", "lexA_CTGTATAWAWWHACA", "background"), "dual_axis_and", [0, 0, 0, 1, 0, 0, 0, 1]),
    ],
)
def test_derive_axis_label_uses_part_detail_not_plan(
    detail: list[dict[str, object]], expected_class: str, expected_vec8: list[int]
) -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": detail,
            "densegen__plan": "background_only__sig35=f",
        }
    )

    assert label.axis_class == expected_class
    assert label.logic4 == AXIS_CLASS_TO_LOGIC4[expected_class]
    assert label.vec8 == expected_vec8
    assert label.densegen_plan_class is not None


def test_plan_axis_mismatch_is_flagged_without_coercing_label() -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": _detail("lexA_CTGTATAWAWWHACA", "background", "background"),
            "densegen__plan": "ethanol__sig35=f",
        }
    )

    assert label.axis_class == "cipro_only"
    assert label.quality_flag == "plan_axis_mismatch"


def test_missing_part_detail_excludes_row_even_when_plan_is_supported() -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": None,
            "densegen__plan": "ciprofloxacin__sig35=f",
        }
    )

    assert label.axis_class is None
    assert label.quality_flag == "missing_used_tfbs_detail"


def test_malformed_part_detail_excludes_row() -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": [{"regulator": "lexA_CTGTATAWAWWHACA"}],
            "densegen__plan": "ciprofloxacin__sig35=f",
        }
    )

    assert label.axis_class is None
    assert label.quality_flag == "malformed_used_tfbs_detail"


def test_unknown_tfbs_regulator_excludes_row() -> None:
    label = derive_axis_label(
        {
            "id": "candidate-1",
            "densegen__used_tfbs_detail": _detail("surpriseRegulator"),
            "densegen__plan": "background_only__sig35=f",
        }
    )

    assert label.axis_class is None
    assert label.quality_flag == "malformed_used_tfbs_detail"


def test_parse_sigma35_variant_from_densegen_plan_suffix() -> None:
    assert parse_sigma35_variant("ethanol_ciprofloxacin__sig35=d") == "d"
    assert parse_sigma35_variant("ethanol") is None


def test_vectorized_prediction_axis_classes_preserve_vec8_contract() -> None:
    values = [
        [0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.9, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    ]

    assert predicted_axis_classes(values) == ["cipro_only", "ethanol_only"]

    with pytest.raises(RuntimeError, match="vec8"):
        predicted_axis_classes([[0.0, 1.0, 0.0, 1.0]])


def test_build_axis_oracle_prefers_sidecar_detail_by_id() -> None:
    candidates = pd.DataFrame(
        [
            {"id": "a", "sequence": "AAAA", "densegen__used_tfbs_detail": None, "densegen__plan": "ethanol__sig35=f"},
            {
                "id": "b",
                "sequence": "CCCC",
                "densegen__used_tfbs_detail": _detail("background"),
                "densegen__plan": "background_only__sig35=e",
            },
        ]
    )
    densegen_sidecar = pd.DataFrame(
        [
            {
                "id": "a",
                "densegen__used_tfbs_detail": _detail("cpxR", "background"),
                "densegen__plan": "ethanol__sig35=f",
                "densegen__sampling_library_hash": "hash-a",
            }
        ]
    )

    labels = build_axis_oracle(candidates, densegen_sidecar=densegen_sidecar)

    row_a = labels.set_index("id").loc["a"]
    assert row_a["axis_class"] == "ethanol_only"
    assert row_a["densegen_plan_class"] == "ethanol"
    assert row_a["tf_family__cpxR__presence"] == 1
    assert row_a["tf_family__cpxR__count"] == 1
    assert row_a["tf_family__lexA__presence"] == 0
    assert row_a["quality_flag"] == "ok"
    assert row_a["sigma35_variant"] == "f"
    assert row_a["densegen__sampling_library_hash"] == "hash-a"


def test_build_axis_oracle_rejects_sidecar_duplicate_ids() -> None:
    candidates = pd.DataFrame(
        [{"id": "a", "sequence": "AAAA", "densegen__used_tfbs_detail": None, "densegen__plan": "ethanol__sig35=f"}]
    )
    densegen_sidecar = pd.DataFrame(
        [
            {"id": "a", "densegen__used_tfbs_detail": _detail("cpxR"), "densegen__plan": "ethanol__sig35=f"},
            {"id": "a", "densegen__used_tfbs_detail": _detail("cpxR"), "densegen__plan": "ethanol__sig35=f"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate id"):
        build_axis_oracle(candidates, densegen_sidecar=densegen_sidecar)


def test_build_axis_oracle_rejects_candidate_sidecar_conflicts() -> None:
    candidates = pd.DataFrame(
        [
            {
                "id": "a",
                "sequence": "AAAA",
                "densegen__used_tfbs_detail": _detail("cpxR"),
                "densegen__plan": "ethanol__sig35=f",
            }
        ]
    )
    densegen_sidecar = pd.DataFrame(
        [
            {
                "id": "a",
                "densegen__used_tfbs_detail": _detail("lexA_CTGTATAWAWWHACA"),
                "densegen__plan": "ethanol__sig35=f",
            }
        ]
    )

    with pytest.raises(ValueError, match="conflict"):
        build_axis_oracle(candidates, densegen_sidecar=densegen_sidecar)


def test_build_train_ids_is_stratified_and_reuses_positive_ids_for_null() -> None:
    rows = []
    for axis_class in AXIS_CLASS_TO_LOGIC4:
        for idx in range(4):
            rows.append(
                {
                    "id": f"{axis_class}-{idx}",
                    "axis_class": axis_class,
                    "quality_flag": "ok",
                    "sigma35_variant": "f" if idx < 2 else "e",
                }
            )
    labels = pd.DataFrame(rows)

    train_ids = build_train_ids(labels, budget=8, seed=7, split_id="random_id")

    selected = labels[labels["id"].isin(train_ids)]
    assert selected.groupby("axis_class").size().to_dict() == {
        "background_only": 2,
        "ethanol_only": 2,
        "cipro_only": 2,
        "dual_axis_and": 2,
    }


def test_build_train_ids_allows_small_nondivisible_initial_budget() -> None:
    rows = []
    for axis_class in AXIS_CLASS_TO_LOGIC4:
        for idx in range(4):
            rows.append(
                {
                    "id": f"{axis_class}-{idx}",
                    "axis_class": axis_class,
                    "quality_flag": "ok",
                    "sigma35_variant": "f" if idx < 2 else "e",
                }
            )
    labels = pd.DataFrame(rows)

    train_ids, metadata = build_train_ids(labels, budget=6, seed=7, split_id="random_id", return_metadata=True)

    selected = labels[labels["id"].isin(train_ids)]
    class_counts = selected.groupby("axis_class").size().to_dict()
    assert len(train_ids) == 6
    assert sorted(class_counts.values()) == [1, 1, 2, 2]
    assert set(class_counts) == set(AXIS_CLASS_TO_LOGIC4)
    assert metadata["class_budget"] == class_counts


def test_build_train_ids_rejects_budget_too_small_for_axis_coverage() -> None:
    labels = pd.DataFrame(
        [
            {"id": axis_class, "axis_class": axis_class, "quality_flag": "ok", "sigma35_variant": "f"}
            for axis_class in AXIS_CLASS_TO_LOGIC4
        ]
    )

    with pytest.raises(ValueError, match="seed every axis class"):
        build_train_ids(labels, budget=3, seed=7, split_id="random_id")


def test_build_train_ids_excludes_leave_sigma35_variant_pool() -> None:
    rows = []
    for axis_class in AXIS_CLASS_TO_LOGIC4:
        for variant in ("a", "b", "c"):
            for idx in range(2):
                rows.append(
                    {
                        "id": f"{axis_class}-{variant}-{idx}",
                        "axis_class": axis_class,
                        "quality_flag": "ok",
                        "sigma35_variant": variant,
                    }
                )
    labels = pd.DataFrame(rows)

    train_ids, metadata = build_train_ids(
        labels,
        budget=8,
        seed=7,
        split_id="leave_sigma35_variant",
        return_metadata=True,
    )

    selected = labels[labels["id"].isin(train_ids)]
    assert metadata["heldout_sigma35"] not in set(selected["sigma35_variant"])


def test_make_permuted_labels_preserves_distribution_and_changes_alignment() -> None:
    labels = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(8)],
            "vec8": [[idx % 2] * 8 for idx in range(8)],
            "axis_class": ["background_only", "ethanol_only", "cipro_only", "dual_axis_and"] * 2,
            "quality_flag": ["ok"] * 8,
        }
    )

    permuted = make_permuted_labels(labels, seed=7)

    assert sorted(map(tuple, permuted["vec8"])) == sorted(map(tuple, labels["vec8"]))
    assert not permuted.set_index("id")["vec8"].equals(labels.set_index("id")["vec8"])


def test_make_permuted_labels_keeps_non_ok_rows_unassigned() -> None:
    labels = pd.DataFrame(
        {
            "id": ["ok-a", "ok-b", "bad"],
            "vec8": [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                None,
            ],
            "axis_class": ["background_only", "ethanol_only", None],
            "quality_flag": ["ok", "ok", "missing_used_tfbs_detail"],
            "v00": [0.0, 0.0, pd.NA],
            "v10": [0.0, 1.0, pd.NA],
            "v01": [0.0, 0.0, pd.NA],
            "v11": [0.0, 1.0, pd.NA],
            "y00_star": [0.0, 0.0, pd.NA],
            "y10_star": [0.0, 1.0, pd.NA],
            "y01_star": [0.0, 0.0, pd.NA],
            "y11_star": [0.0, 1.0, pd.NA],
        }
    )

    permuted = make_permuted_labels(labels, seed=7)

    bad = permuted.set_index("id").loc["bad"]
    assert bad["vec8"] is None
    assert pd.isna(bad["v00"])
    assert pd.isna(bad["axis_class"])


def test_class_from_logic4_uses_nearest_canonical_vector() -> None:
    assert class_from_logic4([0.05, 0.10, 0.85, 0.90]) == "cipro_only"
    assert class_from_logic4([0.10, 0.20, 0.20, 0.75]) == "dual_axis_and"
