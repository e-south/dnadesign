"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_plot_meta.py

Regression tests for plot meta OPAL plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.opal.src.plots._mpl_utils import (
    COLORBLIND_PALETTE,
    categorical_style,
    compact_batch_label,
    math_label,
    observed_batch_marker_map,
    pretty_label,
    pretty_title,
)
from dnadesign.opal.src.plots.scatter_score_vs_rank import _rank_axis_label
from dnadesign.opal.src.registries.plots import describe_plot_kind, get_plot_meta


def test_sfxi_label_diagnostic_plot_meta_disallows_sampling() -> None:
    for name in ["sfxi_factorial_effects", "sfxi_support_diagnostics"]:
        meta = get_plot_meta(name)
        assert meta is not None
        assert "sample_n" not in meta.params
        assert "seed" not in meta.params


def test_sfxi_uncertainty_plot_meta_exposes_explicit_sampling() -> None:
    meta = get_plot_meta("sfxi_uncertainty")
    assert meta is not None
    assert "sample_n" in meta.params
    assert "seed" in meta.params
    assert "batch_size" in meta.params


def test_plot_meta_exposes_dropdown_capability_contract() -> None:
    metric = describe_plot_kind("metric_over_rounds")
    metric_capability = metric["capability"]
    assert metric_capability["objective_family"] == "generic"
    assert metric_capability["round_scope"] == "round_history"
    assert metric_capability["label_requirement"] == "none"
    assert metric_capability["tidy_available"] is True
    assert "band" in metric["params"]

    support = describe_plot_kind("sfxi_support_diagnostics")
    support_capability = support["capability"]
    assert support_capability["objective_family"] == "sfxi"
    assert support_capability["label_requirement"] == "required"
    assert support_capability["requires_labels"] is True


def test_plot_style_helpers_prettify_labels_and_cycle_accessible_categories() -> None:
    assert pretty_label("view__selection_score") == "Selected objective score"
    assert pretty_label("obj__logic_fidelity", raw=True) == "Logic fidelity (obj__logic_fidelity)"
    assert pretty_title("RF top-N Score = -MSE(y_hat, [0, 0, 1, 1]) over rounds") == (
        "RF top-N Score = -MSE(y_hat, [0, 0, 1, 1]) over rounds"
    )
    first = categorical_style(0)
    second = categorical_style(1)
    assert first["color"] == COLORBLIND_PALETTE[0]
    assert second["color"] == COLORBLIND_PALETTE[1]
    assert first["marker"] != second["marker"]
    assert "F_{\\ell}" in math_label("logic_fidelity")
    assert pretty_title("Higher RMF scores receive better round-0 ranks") == (
        "Higher RMF scores receive better round 0 ranks"
    )
    assert pretty_title("MSRB score by active view rank") == "MSRB score by active view rank"
    assert compact_batch_label("pre_round0_response_corpus_4_8h_v1") == "Pre-round 0"
    assert compact_batch_label("batch_12_reader_replicate_v2") == "Batch 12"


def test_observed_batch_markers_never_silently_repeat() -> None:
    batch_ids = tuple(f"batch_{index}" for index in range(12))
    marker_map = observed_batch_marker_map(batch_ids)

    assert tuple(marker_map) == batch_ids
    assert len(set(marker_map.values())) == len(batch_ids)

    import pytest

    with pytest.raises(ValueError, match="filter to at most 12 batches"):
        observed_batch_marker_map((*batch_ids, "batch_12"))

    extended_universe = (*batch_ids, "batch_12")
    with pytest.raises(ValueError, match="marker universe contains 13 batches; at most 12 are supported"):
        observed_batch_marker_map(
            ("batch_0", "batch_12"),
            universe_batch_ids=extended_universe,
        )


def test_score_rank_plot_accepts_an_explicit_directional_rank_label() -> None:
    assert (
        _rank_axis_label(
            x_field="view__rank_competition",
            rank_mode="competition",
            rank_label="Selection rank (lower is better)",
        )
        == "Selection rank (lower is better)"
    )
    metadata = describe_plot_kind("scatter_score_vs_rank")
    assert metadata["premise"]
    assert metadata["decision_value"]
    assert metadata["rationale"]
    assert "rank one appears at the right" in metadata["alt_text"]
    assert metadata["non_claim_boundary"]
    assert "y_axis" in metadata["params"]
