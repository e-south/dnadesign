"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/plots/test_plot_meta.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.opal.src.registries.plots import get_plot_meta


def test_sfxi_diagnostic_plot_meta_disallows_sampling() -> None:
    for name in ["sfxi_factorial_effects", "sfxi_support_diagnostics", "sfxi_uncertainty"]:
        meta = get_plot_meta(name)
        assert meta is not None
        assert "sample_n" not in meta.params
        assert "seed" not in meta.params
