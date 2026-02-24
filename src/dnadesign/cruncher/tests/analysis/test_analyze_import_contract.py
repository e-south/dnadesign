"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_analyze_import_contract.py

Guards analyze import order so Matplotlib cache setup happens before plot modules load.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_analyze_workflow_has_no_top_level_plot_imports() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    tree = ast.parse(workflow_path.read_text(), filename=str(workflow_path))

    offending: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("dnadesign.cruncher.analysis.plots")
        ):
            offending.append(f"{node.module}:{node.lineno}")

    assert not offending, "Move plot imports inside run_analyze after ensure_mpl_cache:\n" + "\n".join(offending)


def test_run_analyze_resolves_runs_before_cache_setup_and_plot_imports() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    content = workflow_path.read_text()

    runs_call = "runs = _resolve_run_names("
    cache_call = "ensure_mpl_cache(catalog_root)"
    plot_import = "from dnadesign.cruncher.analysis.plots.elites_nn_distance import plot_elites_nn_distance"

    runs_idx = content.find(runs_call)
    cache_idx = content.find(cache_call)
    plot_idx = content.find(plot_import)

    assert runs_idx >= 0, "Expected run_analyze to resolve run names."
    assert cache_idx >= 0, "Expected run_analyze to initialize Matplotlib cache."
    assert plot_idx >= 0, "Expected run_analyze to import plot modules lazily."
    assert runs_idx < cache_idx, "run_analyze must resolve run names before cache setup."
    assert cache_idx < plot_idx, "Matplotlib cache must be initialized before loading plot modules."


def test_analyze_cli_defers_mpl_cache_to_workflow() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    cli_path = cruncher_root / "src" / "cli" / "commands" / "analyze.py"
    content = cli_path.read_text()

    workflow_import = "from dnadesign.cruncher.app.analyze_workflow import run_analyze"

    import_idx = content.find(workflow_import)

    assert import_idx >= 0, "Expected analyze CLI to import run_analyze lazily."
    assert "ensure_mpl_cache(" not in content, "Analyze CLI should delegate Matplotlib cache setup to run_analyze."


def test_analyze_workflow_delegates_score_space_helpers_to_module() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    content = workflow_path.read_text()

    helper_import = "from dnadesign.cruncher.app.analyze_score_space import ("
    score_space_context_def = "def _resolve_score_space_context("
    trajectory_projection_def = "def _project_trajectory_views_with_cleanup("
    projection_inputs_def = "def _resolve_objective_projection_inputs("

    assert helper_import in content, "Expected analyze_workflow to import score-space helpers from analyze_score_space."
    assert score_space_context_def not in content, "analyze_workflow should not define _resolve_score_space_context."
    assert trajectory_projection_def not in content, (
        "analyze_workflow should not define _project_trajectory_views_with_cleanup."
    )
    assert projection_inputs_def not in content, (
        "analyze_workflow should not define _resolve_objective_projection_inputs."
    )
