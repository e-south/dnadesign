"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_plot_run_import_contract.py

Import-contract checks for run plotting modules to prevent plot_run.py monolith regression.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import dnadesign.densegen.src.viz.plot_run as plot_run
import dnadesign.densegen.src.viz.plot_run_panels as plot_run_panels


def test_plot_run_panel_builders_are_extracted_modules() -> None:
    assert plot_run.plot_tfbs_usage is plot_run_panels.plot_tfbs_usage
    assert plot_run._build_tfbs_usage_breakdown_figure is plot_run_panels._build_tfbs_usage_breakdown_figure
    assert (
        plot_run._build_run_health_compression_ratio_figure
        is plot_run_panels._build_run_health_compression_ratio_figure
    )
    assert (
        plot_run._build_run_health_tfbs_length_by_regulator_figure
        is plot_run_panels._build_run_health_tfbs_length_by_regulator_figure
    )
