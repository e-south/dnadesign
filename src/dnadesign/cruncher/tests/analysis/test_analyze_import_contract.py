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


def test_analyze_cli_initializes_mpl_cache_before_loading_workflow() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    cli_path = cruncher_root / "src" / "cli" / "commands" / "analyze.py"
    content = cli_path.read_text()

    cache_call = "ensure_mpl_cache(resolve_catalog_root(config_path, cfg.catalog.catalog_root))"
    workflow_import = "from dnadesign.cruncher.app.analyze_workflow import run_analyze"

    cache_idx = content.find(cache_call)
    import_idx = content.find(workflow_import)

    assert cache_idx >= 0, "Expected analyze CLI to initialize Matplotlib cache."
    assert import_idx >= 0, "Expected analyze CLI to import run_analyze lazily."
    assert cache_idx < import_idx, "Matplotlib cache must be initialized before importing analyze_workflow."
