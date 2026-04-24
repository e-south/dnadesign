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
from pathlib import Path


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


def test_portfolio_source_load_helpers_are_extracted() -> None:
    import dnadesign.cruncher.app.portfolio_source_load as source_load
    import dnadesign.cruncher.app.portfolio_workflow as workflow

    assert workflow._load_analysis_summary is source_load._load_analysis_summary
    assert workflow._load_export_elites_windows_and_consensus is source_load._load_export_elites_windows_and_consensus
    assert workflow._mean_pairwise_hamming_bp is source_load._mean_pairwise_hamming_bp


def test_portfolio_workflow_imports_execution_helpers_from_helper_module() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "portfolio_workflow.py"
    content = workflow_path.read_text()

    helper_import = "from dnadesign.cruncher.app.portfolio_execution import ("

    assert helper_import in content
    assert "_run_prepare_then_aggregate as _run_prepare_then_aggregate_helper" in content
    assert "_run_aggregate_only as _run_aggregate_only_helper" in content
    assert "_aggregate_source_into_state as _aggregate_source_into_state_helper" in content
