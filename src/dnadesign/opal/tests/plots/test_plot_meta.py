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
    math_label,
    pretty_label,
    pretty_title,
)
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
    assert pretty_label("pred__score_selected") == "Selected objective score"
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
