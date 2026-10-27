"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_rank_association.py

Generic rank-association primitive contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.enrichments.rank_association import (
    RankAssociationConfig,
    rank_association_rows,
)


def test_rank_association_reports_mann_whitney_effect_size_and_fdr() -> None:
    rows = rank_association_rows(
        universe_ids=["a", "b", "c", "d", "e", "f"],
        score_by_subject={
            "a": {"axis_x": 6.0},
            "b": {"axis_x": 5.0},
            "c": {"axis_x": 4.0},
            "d": {"axis_x": 3.0},
            "e": {"axis_x": 2.0},
            "f": {"axis_x": 1.0},
        },
        features_by_subject={
            "a": {"hot"},
            "b": {"hot"},
            "c": {"hot"},
            "d": {"cold"},
            "e": {"cold"},
            "f": {"cold"},
        },
        feature_labels={"hot": "Hot", "cold": "Cold"},
        axis_ids=["axis_x"],
        config=RankAssociationConfig(min_feature_support=2, alternative="greater"),
        common_features={"cold"},
    )

    by_feature = {row["feature"]: row for row in rows}
    hot = by_feature["Hot"]
    assert hot["n_with"] == 3
    assert hot["n_without"] == 3
    assert hot["median_with"] == 5.0
    assert hot["median_without"] == 2.0
    assert hot["auc"] == pytest.approx(1.0)
    assert hot["rank_biserial"] == pytest.approx(1.0)
    assert hot["p_value_method"] == "scipy_mannwhitneyu_asymptotic"
    assert hot["p_value_alternative"] == "greater"
    assert hot["p_value"] < 0.1
    assert hot["q_value"] >= hot["p_value"]

    cold = by_feature["Cold"]
    assert cold["auc"] == pytest.approx(0.0)
    assert cold["rank_biserial"] == pytest.approx(-1.0)
    assert cold["is_common_feature"] is True
    assert "common_feature" in cold["notes"]


def test_rank_association_fails_fast_on_missing_scores_and_bad_config() -> None:
    with pytest.raises(ContractViolationError, match="missing subjects"):
        rank_association_rows(
            universe_ids=["a", "b"],
            score_by_subject={"a": {"axis_x": 1.0}},
            features_by_subject={"a": {"hot"}},
            feature_labels={"hot": "Hot"},
            axis_ids=["axis_x"],
            config=RankAssociationConfig(min_feature_support=1),
            common_features=set(),
        )

    with pytest.raises(ContractViolationError, match="alternative"):
        rank_association_rows(
            universe_ids=["a", "b"],
            score_by_subject={"a": {"axis_x": 1.0}, "b": {"axis_x": 0.0}},
            features_by_subject={"a": {"hot"}},
            feature_labels={"hot": "Hot"},
            axis_ids=["axis_x"],
            config=RankAssociationConfig(min_feature_support=1, alternative="sideways"),
            common_features=set(),
        )


def test_rank_association_rejects_string_collection_inputs() -> None:
    with pytest.raises(ContractViolationError, match="axis_ids"):
        rank_association_rows(
            universe_ids=["a", "b"],
            score_by_subject={"a": {"axis_x": 1.0}, "b": {"axis_x": 0.0}},
            features_by_subject={"a": {"hot"}},
            feature_labels={"hot": "Hot"},
            axis_ids="axis_x",
            config=RankAssociationConfig(min_feature_support=1),
            common_features=set(),
        )
