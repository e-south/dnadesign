"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_categorical_enrichment.py

Generic categorical enrichment primitive contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.enrichments.categorical_enrichment import (
    CategoricalEnrichmentConfig,
    CategoricalEnrichmentGroup,
    categorical_enrichment_rows,
)


def test_categorical_enrichment_deduplicates_subject_features_and_flags_common() -> None:
    rows = categorical_enrichment_rows(
        universe_ids=["a", "b", "c", "d"],
        features_by_subject={
            "a": {"lexa"},
            "b": {"lexa"},
            "c": {"crp"},
            "d": set(),
        },
        feature_labels={"lexa": "LexA", "crp": "CRP"},
        groups=[
            CategoricalEnrichmentGroup(
                labels={"axis": "cipro", "threshold": 0.5},
                members=frozenset({"a", "b"}),
            )
        ],
        config=CategoricalEnrichmentConfig(min_feature_support=2, min_group_hits=1),
        common_features={"crp"},
    )

    by_feature = {row["feature"]: row for row in rows}
    lexa = by_feature["LexA"]
    assert lexa["axis"] == "cipro"
    assert lexa["n_total"] == 4
    assert lexa["n_group"] == 2
    assert lexa["n_feature_total"] == 2
    assert lexa["n_feature_group"] == 2
    assert lexa["passes_min_feature_support"] is True
    assert lexa["passes_min_group_hits"] is True
    assert lexa["q_value"] >= lexa["p_value"]

    crp = by_feature["CRP"]
    assert crp["is_common_feature"] is True
    assert crp["passes_min_feature_support"] is False
    assert "common_feature" in crp["notes"]


def test_categorical_enrichment_rejects_group_members_outside_universe() -> None:
    with pytest.raises(ContractViolationError, match="outside universe"):
        categorical_enrichment_rows(
            universe_ids=["a"],
            features_by_subject={"a": {"lexa"}},
            feature_labels={"lexa": "LexA"},
            groups=[
                CategoricalEnrichmentGroup(
                    labels={"axis": "cipro"},
                    members=frozenset({"a", "outside"}),
                )
            ],
            config=CategoricalEnrichmentConfig(min_feature_support=1, min_group_hits=1),
            common_features=set(),
        )


def test_categorical_enrichment_rejects_feature_without_label() -> None:
    with pytest.raises(ContractViolationError, match="missing a display label"):
        categorical_enrichment_rows(
            universe_ids=["a"],
            features_by_subject={"a": {"unlabeled_feature"}},
            feature_labels={"lexa": "LexA"},
            groups=[
                CategoricalEnrichmentGroup(
                    labels={"axis": "cipro"},
                    members=frozenset({"a"}),
                )
            ],
            config=CategoricalEnrichmentConfig(min_feature_support=1, min_group_hits=1),
            common_features=set(),
        )


def test_categorical_enrichment_rejects_unsupported_fdr_method() -> None:
    with pytest.raises(ContractViolationError, match="fdr_method"):
        categorical_enrichment_rows(
            universe_ids=["a"],
            features_by_subject={"a": {"lexa"}},
            feature_labels={"lexa": "LexA"},
            groups=[
                CategoricalEnrichmentGroup(
                    labels={"axis": "cipro"},
                    members=frozenset({"a"}),
                )
            ],
            config=CategoricalEnrichmentConfig(
                min_feature_support=1,
                min_group_hits=1,
                fdr_method="bonferroni",
            ),
            common_features=set(),
        )


def test_categorical_enrichment_rejects_string_as_collection_inputs() -> None:
    with pytest.raises(ContractViolationError, match="universe_ids"):
        categorical_enrichment_rows(
            universe_ids="abc",
            features_by_subject={"a": {"lexa"}},
            feature_labels={"lexa": "LexA"},
            groups=[
                CategoricalEnrichmentGroup(
                    labels={"axis": "cipro"},
                    members=frozenset({"a"}),
                )
            ],
            config=CategoricalEnrichmentConfig(min_feature_support=1, min_group_hits=1),
            common_features=set(),
        )

    with pytest.raises(ContractViolationError, match="features for subject 'a'"):
        categorical_enrichment_rows(
            universe_ids=["a"],
            features_by_subject={"a": "lexa"},
            feature_labels={"lexa": "LexA"},
            groups=[
                CategoricalEnrichmentGroup(
                    labels={"axis": "cipro"},
                    members=frozenset({"a"}),
                )
            ],
            config=CategoricalEnrichmentConfig(min_feature_support=1, min_group_hits=1),
            common_features=set(),
        )

    with pytest.raises(ContractViolationError, match="common_features"):
        categorical_enrichment_rows(
            universe_ids=["a"],
            features_by_subject={"a": {"lexa"}},
            feature_labels={"lexa": "LexA"},
            groups=[
                CategoricalEnrichmentGroup(
                    labels={"axis": "cipro"},
                    members=frozenset({"a"}),
                )
            ],
            config=CategoricalEnrichmentConfig(min_feature_support=1, min_group_hits=1),
            common_features="crp",
        )

    with pytest.raises(ContractViolationError, match="group.members"):
        categorical_enrichment_rows(
            universe_ids=["a"],
            features_by_subject={"a": {"lexa"}},
            feature_labels={"lexa": "LexA"},
            groups=[
                CategoricalEnrichmentGroup(
                    labels={"axis": "cipro"},
                    members="a",  # type: ignore[arg-type]
                )
            ],
            config=CategoricalEnrichmentConfig(min_feature_support=1, min_group_hits=1),
            common_features=set(),
        )

    with pytest.raises(ContractViolationError, match="groups"):
        categorical_enrichment_rows(
            universe_ids=["a"],
            features_by_subject={"a": {"lexa"}},
            feature_labels={"lexa": "LexA"},
            groups="cipro",
            config=CategoricalEnrichmentConfig(min_feature_support=1, min_group_hits=1),
            common_features=set(),
        )


def test_categorical_enrichment_rejects_empty_groups() -> None:
    with pytest.raises(ContractViolationError, match="group members must not be empty"):
        categorical_enrichment_rows(
            universe_ids=["a"],
            features_by_subject={"a": {"lexa"}},
            feature_labels={"lexa": "LexA"},
            groups=[
                CategoricalEnrichmentGroup(
                    labels={"axis": "cipro"},
                    members=frozenset(),
                )
            ],
            config=CategoricalEnrichmentConfig(min_feature_support=1, min_group_hits=1),
            common_features=set(),
        )
