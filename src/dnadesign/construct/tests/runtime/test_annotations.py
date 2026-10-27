"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/test_annotations.py

Annotation parsing contracts for Construct realization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.construct.src.annotations.features import load_annotation_features
from dnadesign.construct.src.contracts.errors import ValidationError


def test_load_annotation_features_treats_missing_annotation_column_as_empty() -> None:
    assert load_annotation_features({"id": "row_a"}) == []
    assert load_annotation_features({"id": "row_a", "seq_annot__features": None}) == []


def test_load_annotation_features_rejects_malformed_feature_container() -> None:
    with pytest.raises(ValidationError, match="seq_annot__features must be a list"):
        load_annotation_features({"id": "row_a", "seq_annot__features": "not-a-list"})


def test_load_annotation_features_rejects_malformed_feature_entry() -> None:
    with pytest.raises(ValidationError, match="seq_annot__features\\[0\\] must be a mapping"):
        load_annotation_features({"id": "row_a", "seq_annot__features": ["not-a-mapping"]})


def test_load_annotation_features_rejects_malformed_interval_entry() -> None:
    with pytest.raises(ValidationError, match="intervals_0\\[0\\] must be a mapping"):
        load_annotation_features(
            {
                "id": "row_a",
                "seq_annot__features": [
                    {
                        "feature_id": "sig35",
                        "feature_type": "promoter",
                        "intervals_0": ["not-a-mapping"],
                    }
                ],
            }
        )


def test_load_annotation_features_rejects_inverted_feature_bounds() -> None:
    with pytest.raises(ValidationError, match="start_0 must be <= end_0"):
        load_annotation_features(
            {
                "id": "row_a",
                "seq_annot__features": [
                    {
                        "feature_id": "sig35",
                        "feature_type": "promoter",
                        "start_0": 8,
                        "end_0": 4,
                        "intervals_0": [{"start_0": 8, "end_0": 4, "partial": False}],
                    }
                ],
            }
        )
