"""Plan-margin feature-enrichment contract tests."""

from __future__ import annotations

import pyarrow as pa
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.enrichments.plan_margin_feature_enrichment import (
    build_plan_margin_feature_enrichment_artifact,
)


def _scores_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"native_parent_id": "p1", "margin_background": 0.1, "margin_ethanol": 0.7},
            {"native_parent_id": "p2", "margin_background": 0.4, "margin_ethanol": -0.2},
        ]
    )


def _tail_membership_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "native_parent_id": "p1",
                "plan": "ethanol",
                "threshold": 0.5,
                "tail_mode": "margin_top_quantile",
            },
            {
                "native_parent_id": "p2",
                "plan": "background",
                "threshold": 0.5,
                "tail_mode": "margin_top_quantile",
            },
        ]
    )


def _feature_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "usr_id": "p1",
                "feature_id": "F:1",
                "feature_label": "response to stress",
                "feature_namespace": "biological_process",
            }
        ]
    )


def test_plan_margin_feature_enrichment_requires_namespace_filter_pair() -> None:
    with pytest.raises(ContractViolationError, match="configured together"):
        build_plan_margin_feature_enrichment_artifact(
            scores_table=_scores_table(),
            tail_membership_table=_tail_membership_table(),
            feature_table=_feature_table(),
            subject_column="usr_id",
            feature_id_column="feature_id",
            feature_label_column="feature_label",
            feature_namespace_column=None,
            namespace_filter="biological_process",
            min_global_subjects=1,
            min_tail_hits=1,
        )


def test_plan_margin_feature_enrichment_emits_rank_backbone() -> None:
    artifact = build_plan_margin_feature_enrichment_artifact(
        scores_table=_scores_table(),
        tail_membership_table=_tail_membership_table(),
        feature_table=_feature_table(),
        subject_column="usr_id",
        feature_id_column="feature_id",
        feature_label_column="feature_label",
        feature_namespace_column="feature_namespace",
        namespace_filter="biological_process",
        min_global_subjects=1,
        min_tail_hits=1,
    )

    rank_rows = artifact.rank_tests_table.to_pylist()
    assert {row["plan"] for row in rank_rows} == {"background", "ethanol"}
    ethanol = next(row for row in rank_rows if row["plan"] == "ethanol")
    assert ethanol["feature_label"] == "response to stress"
    assert ethanol["rank_biserial"] > 0.0
    assert artifact.stats["rank_test_rows"] == 2


def test_plan_margin_feature_enrichment_rejects_score_plans_absent_from_tail_groups() -> None:
    with pytest.raises(ContractViolationError, match="absent from tail groups"):
        build_plan_margin_feature_enrichment_artifact(
            scores_table=pa.Table.from_pylist(
                [
                    {
                        "native_parent_id": "p1",
                        "margin_background": 0.1,
                        "margin_ethanol": 0.7,
                        "margin_extra": 0.2,
                    },
                    {
                        "native_parent_id": "p2",
                        "margin_background": 0.4,
                        "margin_ethanol": -0.2,
                        "margin_extra": -0.1,
                    },
                ]
            ),
            tail_membership_table=_tail_membership_table(),
            feature_table=_feature_table(),
            subject_column="usr_id",
            feature_id_column="feature_id",
            feature_label_column="feature_label",
            feature_namespace_column="feature_namespace",
            namespace_filter="biological_process",
            min_global_subjects=1,
            min_tail_hits=1,
        )
