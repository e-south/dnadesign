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


def test_analyze_plot_resolver_has_no_top_level_plot_imports() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    resolver_path = cruncher_root / "src" / "app" / "analyze" / "plot_resolver.py"
    tree = ast.parse(resolver_path.read_text(), filename=str(resolver_path))

    offending: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("dnadesign.cruncher.analysis.plots")
        ):
            offending.append(f"{node.module}:{node.lineno}")

    assert not offending, (
        "plot_resolver should keep plot imports inside resolve_analysis_plot_functions:\n" + "\n".join(offending)
    )


def test_run_analyze_resolves_runs_before_cache_setup_and_plot_imports() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    content = workflow_path.read_text()

    runs_call = "runs = _resolve_run_names("
    cache_call = "ensure_mpl_cache(catalog_root)"
    resolver_call = "plotters = resolve_analysis_plot_functions()"

    runs_idx = content.find(runs_call)
    cache_idx = content.find(cache_call)
    resolver_idx = content.find(resolver_call)

    assert runs_idx >= 0, "Expected run_analyze to resolve run names."
    assert cache_idx >= 0, "Expected run_analyze to initialize Matplotlib cache."
    assert resolver_idx >= 0, "Expected run_analyze to resolve plot functions lazily."
    assert runs_idx < cache_idx, "run_analyze must resolve run names before cache setup."
    assert cache_idx < resolver_idx, "Matplotlib cache must be initialized before loading plot modules."


def test_analyze_workflow_delegates_plot_import_resolution_to_module() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    content = workflow_path.read_text()

    assert "from dnadesign.cruncher.app.analyze.plot_resolver import" in content
    assert "resolve_analysis_plot_functions" in content
    assert "plotters = resolve_analysis_plot_functions()" in content

    direct_plot_imports = (
        "from dnadesign.cruncher.analysis.plots.elites_nn_distance import plot_elites_nn_distance",
        "from dnadesign.cruncher.analysis.plots.elites_showcase import plot_elites_showcase",
        "from dnadesign.cruncher.analysis.plots.fimo_concordance import plot_optimizer_vs_fimo",
        "from dnadesign.cruncher.analysis.plots.health_panel import plot_health_panel",
        "from dnadesign.cruncher.analysis.plots.trajectory_score_space_plot import plot_elite_score_space_context",
        "from dnadesign.cruncher.analysis.plots.trajectory_sweep import plot_chain_trajectory_sweep",
    )
    for text in direct_plot_imports:
        assert text not in content, "analyze_workflow should not directly import plot implementations."


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

    run_context_import = "from dnadesign.cruncher.app.analyze.run_context import"
    resolver_call = "resolve_analysis_run_context("
    direct_score_space_import = "from dnadesign.cruncher.app.analyze_score_space import _resolve_score_space_context"

    assert run_context_import in content, "Expected analyze_workflow to delegate score-space resolution to run_context."
    assert resolver_call in content, "Expected analyze_workflow to resolve run context through run_context module."
    assert direct_score_space_import not in content, "analyze_workflow should not import score-space helpers directly."


def test_analyze_workflow_delegates_plot_rendering_helpers_to_module() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    content = workflow_path.read_text()

    helper_import = "from dnadesign.cruncher.app.analyze.rendering import render_analysis_plots"
    delegated_calls = (
        "_prepare_analysis_plot_dir(",
        "_render_trajectory_analysis_plots(",
        "_render_trajectory_video_plot(",
        "_render_static_analysis_plots(",
        "_render_fimo_analysis_plot(",
    )

    assert helper_import in content, (
        "Expected analyze_workflow to import plot orchestration from app.analyze.rendering."
    )
    assert "render_analysis_plots(" in content, "analyze_workflow should delegate plot orchestration."
    for delegated_call in delegated_calls:
        assert delegated_call not in content, f"analyze_workflow should delegate {delegated_call}."


def test_analyze_plotting_module_delegates_rendering_to_submodules() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    plotting_path = cruncher_root / "src" / "app" / "analyze" / "plotting.py"
    content = plotting_path.read_text()

    assert "from dnadesign.cruncher.app.analyze.plotting_trajectory import (" in content
    assert "from dnadesign.cruncher.app.analyze.plotting_static import (" in content
    assert "from dnadesign.cruncher.app.analyze.plotting_registry import _prepare_analysis_plot_dir" in content

    delegated_defs = (
        "def _record_analysis_plot(",
        "def _render_trajectory_analysis_plots(",
        "def _render_trajectory_video_plot(",
        "def _render_static_analysis_plots(",
        "def _render_fimo_analysis_plot(",
    )
    for delegated_def in delegated_defs:
        assert delegated_def not in content, f"plotting.py should not define {delegated_def}."


def test_analyze_workflow_delegates_output_publication_to_module() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    content = workflow_path.read_text()

    assert "from dnadesign.cruncher.app.analyze.publish import publish_analysis_outputs" in content
    assert "publish_analysis_outputs(" in content

    non_orchestration_calls = (
        "build_table_entries(",
        "build_table_artifacts(",
        "build_analysis_manifests(",
        "build_report_payload(",
        "write_report_json(",
        "write_report_md(",
        "build_summary_payload(",
    )
    for call_text in non_orchestration_calls:
        assert call_text not in content, f"analyze_workflow should delegate {call_text}."


def test_analyze_workflow_delegates_table_and_metric_computation_to_run_context_module() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    content = workflow_path.read_text()

    assert "from dnadesign.cruncher.app.analyze.run_context import" in content
    assert "resolve_analysis_run_context" in content
    assert "compute_analysis_tables_and_metrics(" not in content


def test_analyze_workflow_delegates_run_execution_context_to_module() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "analyze_workflow.py"
    content = workflow_path.read_text()

    assert "from dnadesign.cruncher.app.analyze.execution import (" in content
    assert "AnalysisRunExecutionContext" in content
    assert "resolve_analysis_run_execution_context" in content
    assert "execution = resolve_analysis_run_execution_context(" in content

    inline_context_calls = (
        "run_dir = _resolve_run_dir(",
        "manifest = load_manifest(run_dir)",
        "optimizer_stats = _resolve_optimizer_stats(",
        "_verify_manifest_lockfile(",
        "pwms, used_cfg = load_pwms_from_config(",
        "sample_meta = _resolve_sample_meta(",
        "_load_run_artifacts_for_analysis(",
    )
    for call_text in inline_context_calls:
        assert call_text not in content, f"analyze_workflow should delegate {call_text} to execution context module."
