"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/masks/test_rt_intervals.py

RT interval source contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks import load_manual_mask_authority_source
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks.rt_intervals import (
    EXPECTED_RT_INTERVAL_FEATURE_IDS,
    rt_interval_feature_ids_from_source,
    rt_interval_features_from_source,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def test_manual_mask_source_declares_exact_rt_interval_review_spans() -> None:
    authority_source = load_manual_mask_authority_source(repo_root())

    assert rt_interval_feature_ids_from_source(authority_source) == EXPECTED_RT_INTERVAL_FEATURE_IDS
    intervals = {feature.feature_id: feature for feature in rt_interval_features_from_source(authority_source)}
    assert {
        feature_id: (feature.canonical_start, feature.canonical_end) for feature_id, feature in intervals.items()
    } == {
        "rt1_interval": (33, 64),
        "rt2_interval": (65, 99),
        "rt3_interval": (111, 151),
        "rt4_interval": (159, 190),
        "rt5_interval": (192, 211),
        "rt6_interval": (212, 230),
        "rt7_interval": (231, 245),
    }
    assert {feature.policy for feature in intervals.values()} == {"review_label"}
