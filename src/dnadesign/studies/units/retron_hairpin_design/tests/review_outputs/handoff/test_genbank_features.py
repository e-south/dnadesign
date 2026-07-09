"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/review_outputs/handoff/test_genbank_features.py

Tests for Retron MSD GenBank feature-direction normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.retron_hairpin_design.compiler.exceptions import RetronMsdCompilerError
from dnadesign.studies.units.retron_hairpin_design.review_outputs.handoff.genbank_features import (
    rewrite_reverse_complement_features,
)


def test_genbank_feature_rewrite_fails_on_unknown_typed_role() -> None:
    features = [
        "FEATURES             Location/Qualifiers",
        "     misc_feature    complement(1..4)",
        '                     /label="Unexpected"',
        '                     /dnadesign_role="unexpected_role"',
    ]

    with pytest.raises(RetronMsdCompilerError, match="unknown dnadesign_role: unexpected_role"):
        rewrite_reverse_complement_features(features)


def test_genbank_feature_rewrite_preserves_untyped_source_features() -> None:
    features = [
        "FEATURES             Location/Qualifiers",
        "     source          1..4",
        '                     /mol_type="other DNA"',
    ]

    assert rewrite_reverse_complement_features(features) == features
