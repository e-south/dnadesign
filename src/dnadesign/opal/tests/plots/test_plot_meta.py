"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/plots/test_plot_meta.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.opal.src.registries.plots import describe_plot_kind, get_plot_meta


def test_sfxi_diagnostic_plot_meta_disallows_sampling() -> None:
    for name in ["sfxi_factorial_effects", "sfxi_support_diagnostics", "sfxi_uncertainty"]:
        meta = get_plot_meta(name)
        assert meta is not None
        assert "sample_n" not in meta.params
        assert "seed" not in meta.params


def test_plot_meta_exposes_dropdown_capability_contract() -> None:
    metric = describe_plot_kind("metric_over_rounds")
    metric_capability = metric["capability"]
    assert metric_capability["objective_family"] == "generic"
    assert metric_capability["round_scope"] == "round_history"
    assert metric_capability["label_requirement"] == "none"
    assert metric_capability["tidy_available"] is True

    support = describe_plot_kind("sfxi_support_diagnostics")
    support_capability = support["capability"]
    assert support_capability["objective_family"] == "sfxi"
    assert support_capability["label_requirement"] == "required"
    assert support_capability["requires_labels"] is True
