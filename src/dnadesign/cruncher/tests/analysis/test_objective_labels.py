"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_objective_labels.py

Characterization tests for shared objective-axis label semantics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.analysis.objective_labels import (
    objective_scalar_semantics,
    objective_scale_label,
)


def test_objective_scale_label_maps_known_scales() -> None:
    assert objective_scale_label({"score_scale": "llr"}) == "raw-LLR"
    assert objective_scale_label({"score_scale": "raw-llr"}) == "raw-LLR"
    assert objective_scale_label({"score_scale": "normalized-llr"}) == "norm-LLR"
    assert objective_scale_label({"score_scale": "logp"}) == "logp"
    assert objective_scale_label({"score_scale": "z"}) == "z"


def test_objective_scale_label_unknown_fallback_is_optional() -> None:
    assert objective_scale_label({"score_scale": "mystery"}) == "mystery"
    assert objective_scale_label({"score_scale": "mystery"}, unknown_fallback="norm-LLR") == "norm-LLR"


def test_objective_scalar_semantics_respects_combine_and_softmin() -> None:
    assert (
        objective_scalar_semantics({"combine": "sum", "score_scale": "normalized-llr"}) == "sum TF best-window norm-LLR"
    )
    assert (
        objective_scalar_semantics({"combine": "min", "score_scale": "normalized-llr", "softmin": {"enabled": True}})
        == "soft-min TF best-window norm-LLR"
    )
    assert objective_scalar_semantics({"combine": "min", "score_scale": "llr"}) == "min TF best-window raw-LLR"
