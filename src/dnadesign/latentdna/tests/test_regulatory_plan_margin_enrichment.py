"""RegulonDB regulator plan-margin enrichment contracts."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.enrichments.regulatory_plan_margin import (
    build_regulatory_plan_margin_artifacts,
)


def _rows() -> list[dict[str, object]]:
    base_synthetic = [
        ("bg_1", "background_only", [5.0, 0.0, 0.0, 0.0]),
        ("bg_2", "background_only", [5.0, 0.2, 0.0, 0.0]),
        ("eth_1", "ethanol", [0.0, 5.0, 0.0, 0.0]),
        ("eth_2", "ethanol", [0.0, 5.0, 0.2, 0.0]),
        ("cip_1", "ciprofloxacin", [0.0, 0.0, 5.0, 0.0]),
        ("cip_2", "ciprofloxacin", [0.0, 0.0, 5.0, 0.2]),
        ("dual_1", "ethanol_ciprofloxacin", [0.0, 0.0, 0.0, 5.0]),
        ("dual_2", "ethanol_ciprofloxacin", [0.2, 0.0, 0.0, 5.0]),
    ]
    native = [
        ("native_eth_1", "native", [0.0, 4.7, 0.0, 0.1], "rp_eth_1", "p_eth_1"),
        ("native_eth_2", "native", [0.1, 4.6, 0.0, 0.0], "rp_eth_2", "p_eth_2"),
        ("native_cipro_1", "native", [0.0, 0.0, 4.8, 0.2], "rp_cipro_1", "p_cipro_1"),
        ("native_dual_1", "native", [0.1, 0.1, 0.0, 4.9], "rp_dual_1", "p_dual_1"),
        ("native_bg_1", "native", [4.8, 0.1, 0.0, 0.0], "rp_bg_1", "p_bg_1"),
        ("native_bg_2", "native", [4.7, 0.0, 0.1, 0.0], "rp_bg_2", "p_bg_2"),
    ]
    rows: list[dict[str, object]] = []
    for row_id, design_family, _vector in base_synthetic:
        rows.append(
            {
                "alias_id": row_id,
                "design_family": design_family,
                "derived__parent_dataset": "densegen",
                "derived__parent_id": row_id,
                "regulondb__primary_promoter_id": None,
                "regulondb__primary_promoter_name": None,
            }
        )
    for row_id, design_family, _vector, parent_id, promoter_id in native:
        rows.append(
            {
                "alias_id": row_id,
                "design_family": design_family,
                "derived__parent_dataset": "usr_regulondb_native_promoters",
                "derived__parent_id": parent_id,
                "regulondb__primary_promoter_id": promoter_id,
                "regulondb__primary_promoter_name": promoter_id,
            }
        )
    return rows


def _matrix() -> np.ndarray:
    vectors = [
        [5.0, 0.0, 0.0, 0.0],
        [5.0, 0.2, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 5.0, 0.2, 0.0],
        [0.0, 0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0, 0.2],
        [0.0, 0.0, 0.0, 5.0],
        [0.2, 0.0, 0.0, 5.0],
        [0.0, 4.7, 0.0, 0.1],
        [0.1, 4.6, 0.0, 0.0],
        [0.0, 0.0, 4.8, 0.2],
        [0.1, 0.1, 0.0, 4.9],
        [4.8, 0.1, 0.0, 0.0],
        [4.7, 0.0, 0.1, 0.0],
    ]
    return np.asarray(vectors, dtype=np.float32)


def _relations() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"usr_id": "rp_eth_1", "regulator_abbrev": "CpxR"},
            {"usr_id": "rp_eth_1", "regulator_abbrev": "CpxR"},
            {"usr_id": "rp_eth_2", "regulator_abbrev": "BaeR"},
            {"usr_id": "rp_cipro_1", "regulator_abbrev": "LexA"},
            {"usr_id": "rp_dual_1", "regulator_abbrev": "LexA"},
            {"usr_id": "rp_bg_1", "regulator_abbrev": "CRP"},
            {"usr_id": "rp_bg_2", "regulator_abbrev": "CRP"},
        ]
    )


def test_regulatory_plan_margin_artifacts_dedupe_regulators_and_report_enrichment() -> None:
    artifacts = build_regulatory_plan_margin_artifacts(
        matrix=_matrix(),
        rows_table=pa.Table.from_pylist(_rows()),
        relations_table=_relations(),
        view_id="bidir_context",
        cohort_column="design_family",
        centroid_groups={
            "background": ["background_only"],
            "ethanol": ["ethanol"],
            "cipro": ["ciprofloxacin"],
            "dual": ["ethanol_ciprofloxacin"],
        },
        native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
        native_parent_column="derived__parent_id",
        relation_key="usr_id",
        regulator_column="regulator_abbrev",
        thresholds=[0.34],
        tail_modes=["margin_top_quantile", "margin_top_quantile_nearest_plan_only"],
        min_global_promoters=2,
        min_tail_hits=1,
        common_regulators=["CRP"],
        native_metadata_columns=[
            "alias_id",
            "regulondb__primary_promoter_id",
            "regulondb__primary_promoter_name",
        ],
        expected_output_rows=6,
    )

    score_rows = artifacts.scores_table.to_pylist()
    scores_by_parent = {row["native_parent_id"]: row for row in score_rows}
    assert artifacts.scores_table.num_rows == 6
    assert scores_by_parent["rp_eth_1"]["regulondb__primary_promoter_id"] == "p_eth_1"
    assert scores_by_parent["rp_eth_1"]["nearest_plan"] == "ethanol"
    assert scores_by_parent["rp_eth_1"]["regulator_degree"] == 1
    assert scores_by_parent["rp_dual_1"]["regulator_degree"] == 1
    assert scores_by_parent["rp_bg_1"]["regulator_degree"] == 1

    tail_rows = artifacts.tail_membership_table.to_pylist()
    ethanol_tail = {
        row["native_parent_id"]
        for row in tail_rows
        if row["plan"] == "ethanol" and row["tail_mode"] == "margin_top_quantile"
    }
    assert {"rp_eth_1", "rp_eth_2"}.issubset(ethanol_tail)

    enrichment_rows = artifacts.enrichment_table.to_pylist()
    by_key = {
        (row["regulator_abbrev"], row["plan"], row["threshold"], row["tail_mode"]): row for row in enrichment_rows
    }
    cpxr = by_key[("CpxR", "ethanol", 0.34, "margin_top_quantile")]
    assert cpxr["n_regulator_total"] == 1
    assert cpxr["n_regulator_tail"] == 1
    assert cpxr["passes_min_support"] is False
    assert cpxr["passes_min_tail_hits"] is True
    assert cpxr["p_value_method"] == "hypergeometric_survival"
    assert cpxr["fdr_method"] == "benjamini_hochberg"

    crp = by_key[("CRP", "background", 0.34, "margin_top_quantile")]
    assert crp["is_common_regulator"] is True
    assert crp["n_regulator_total"] == 2
    assert crp["q_value"] >= crp["p_value"]
    assert artifacts.stats["matched_regulators"] == 4


def test_regulatory_plan_margin_artifacts_support_configured_plan_ids() -> None:
    artifacts = build_regulatory_plan_margin_artifacts(
        matrix=_matrix(),
        rows_table=pa.Table.from_pylist(_rows()),
        relations_table=_relations(),
        view_id="bidir_context",
        cohort_column="design_family",
        centroid_groups={
            "axis_a": ["background_only"],
            "axis_b": ["ethanol"],
        },
        native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
        native_parent_column="derived__parent_id",
        relation_key="usr_id",
        regulator_column="regulator_abbrev",
        thresholds=[0.5],
        tail_modes=["margin_top_quantile"],
        min_global_promoters=1,
        min_tail_hits=1,
        common_regulators=[],
        expected_output_rows=6,
    )

    score_row = artifacts.scores_table.to_pylist()[0]
    assert artifacts.stats["plan_order"] == ["axis_a", "axis_b"]
    assert "sim_axis_a" in score_row
    assert "margin_axis_b" in score_row
    assert set(artifacts.enrichment_table.column("plan").to_pylist()) == {"axis_a", "axis_b"}


def test_regulatory_plan_margin_artifacts_fail_fast_on_duplicate_native_parent_ids() -> None:
    rows = _rows()
    rows[-1]["derived__parent_id"] = rows[-2]["derived__parent_id"]

    with pytest.raises(ContractViolationError, match="duplicate native parent ids"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(rows),
            relations_table=_relations(),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": ["background_only"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            thresholds=[0.34],
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            expected_output_rows=6,
        )


def test_regulatory_plan_margin_artifacts_fail_fast_when_no_regulators_match_native_rows() -> None:
    with pytest.raises(ContractViolationError, match="matched no regulatory associations"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(_rows()),
            relations_table=pa.Table.from_pylist([{"usr_id": "outside_scope", "regulator_abbrev": "LexA"}]),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": ["background_only"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            thresholds=[0.34],
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            expected_output_rows=6,
        )


def test_regulatory_plan_margin_artifacts_fail_fast_on_missing_required_relation_columns() -> None:
    with pytest.raises(ContractViolationError, match="source_release"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(_rows()),
            relations_table=_relations(),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": ["background_only"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            required_relation_columns=["source_release"],
            thresholds=[0.34],
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            expected_output_rows=6,
        )


def test_regulatory_plan_margin_artifacts_fail_fast_on_missing_native_metadata_columns() -> None:
    with pytest.raises(ContractViolationError, match="missing_metadata"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(_rows()),
            relations_table=_relations(),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": ["background_only"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            native_metadata_columns=["missing_metadata"],
            thresholds=[0.34],
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            expected_output_rows=6,
        )


def test_regulatory_plan_margin_artifacts_reject_ambiguous_native_filter() -> None:
    with pytest.raises(ContractViolationError, match="exactly one"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(_rows()),
            relations_table=_relations(),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": ["background_only"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            native_filter={
                "column": "derived__parent_dataset",
                "equals": "usr_regulondb_native_promoters",
                "in": ["usr_regulondb_native_promoters"],
            },
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            thresholds=[0.34],
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            expected_output_rows=6,
        )


def test_regulatory_plan_margin_artifacts_reject_string_collection_config() -> None:
    with pytest.raises(ContractViolationError, match="centroid group 'background'"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(_rows()),
            relations_table=_relations(),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": "background_only",
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            thresholds=[0.34],
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            expected_output_rows=6,
        )
    with pytest.raises(ContractViolationError, match="thresholds must be a sequence"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(_rows()),
            relations_table=_relations(),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": ["background_only"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            thresholds="0.34",
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            expected_output_rows=6,
        )


def test_regulatory_plan_margin_artifacts_reject_unsupported_fdr_method() -> None:
    with pytest.raises(ContractViolationError, match="fdr_method"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(_rows()),
            relations_table=_relations(),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": ["background_only"],
                "ethanol": ["ethanol"],
                "cipro": ["ciprofloxacin"],
                "dual": ["ethanol_ciprofloxacin"],
            },
            native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            thresholds=[0.34],
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            fdr_method="bonferroni",
            expected_output_rows=6,
        )


def test_regulatory_plan_margin_artifacts_reject_unsafe_plan_ids() -> None:
    with pytest.raises(ContractViolationError, match="plan ids"):
        build_regulatory_plan_margin_artifacts(
            matrix=_matrix(),
            rows_table=pa.Table.from_pylist(_rows()),
            relations_table=_relations(),
            view_id="bidir_context",
            cohort_column="design_family",
            centroid_groups={
                "background": ["background_only"],
                "cipro-response": ["ciprofloxacin"],
            },
            native_filter={"column": "derived__parent_dataset", "equals": "usr_regulondb_native_promoters"},
            native_parent_column="derived__parent_id",
            relation_key="usr_id",
            regulator_column="regulator_abbrev",
            thresholds=[0.34],
            tail_modes=["margin_top_quantile"],
            min_global_promoters=1,
            min_tail_hits=1,
            common_regulators=[],
            expected_output_rows=6,
        )
