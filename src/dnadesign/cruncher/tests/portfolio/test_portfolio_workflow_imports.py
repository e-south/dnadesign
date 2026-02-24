"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/portfolio/test_portfolio_workflow_imports.py

Import contracts for Portfolio workflow module.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import sys


def test_portfolio_workflow_defers_matplotlib_import() -> None:
    module_name = "dnadesign.cruncher.app.portfolio_workflow"
    pyplot_module = "matplotlib.pyplot"

    sys.modules.pop(module_name, None)
    sys.modules.pop(pyplot_module, None)

    importlib.import_module(module_name)

    assert pyplot_module not in sys.modules


def test_portfolio_preflight_helpers_are_extracted() -> None:
    import dnadesign.cruncher.app.portfolio_preflight as preflight
    import dnadesign.cruncher.app.portfolio_workflow as workflow

    assert workflow._preflight_source_readiness is preflight._preflight_source_readiness
    assert workflow._collect_source_readiness is preflight._collect_source_readiness
    assert workflow._raise_aggregate_only_preflight is preflight._raise_aggregate_only_preflight
    assert workflow._resolve_source_label is preflight._resolve_source_label
    assert workflow._render_prepare_runbook_command is preflight._render_prepare_runbook_command


def test_portfolio_output_helpers_are_extracted() -> None:
    import dnadesign.cruncher.app.portfolio_materialization as outputs
    import dnadesign.cruncher.app.portfolio_workflow as workflow

    assert workflow._materialize_portfolio_outputs is outputs._materialize_portfolio_outputs
    assert workflow._select_portfolio_showcase_elites is outputs._select_portfolio_showcase_elites
    assert workflow._write_tradeoff_plot is outputs._write_tradeoff_plot
