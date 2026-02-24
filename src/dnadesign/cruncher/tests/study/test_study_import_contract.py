"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_import_contract.py

Guards study import order so Matplotlib cache setup happens before plot modules
load.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_study_summary_has_no_top_level_plot_imports() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    summary_path = cruncher_root / "src" / "app" / "study_summary.py"
    tree = ast.parse(summary_path.read_text(), filename=str(summary_path))

    offending: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "dnadesign.cruncher.study.plots":
            offending.append(f"{node.module}:{node.lineno}")

    assert not offending, "Move study plot imports inside summarize_study_run after ensure_mpl_cache:\n" + "\n".join(
        offending
    )


def test_study_workflow_delegates_summarize_to_study_summary_module() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    workflow_path = cruncher_root / "src" / "app" / "study_workflow.py"
    content = workflow_path.read_text()

    summary_import = "from dnadesign.cruncher.app.study_summary import summarize_study_run"

    import_idx = content.find(summary_import)
    summarize_def_idx = content.find("def summarize_study_run(")

    assert import_idx >= 0, "Expected study_workflow to import summarize entrypoints from study_summary."
    assert summarize_def_idx < 0, "study_workflow should not define summarize_study_run directly."


def test_summarize_study_run_sets_cache_before_lazy_plot_imports() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    summary_path = cruncher_root / "src" / "app" / "study_summary.py"
    content = summary_path.read_text()

    cache_call = "ensure_mpl_cache(study_meta_dir(study_run_dir))"
    plot_import = "from dnadesign.cruncher.study import plots as study_plots"
    mmr_call = "plot_mmr_diversity_tradeoff_fn=study_plots.plot_mmr_diversity_tradeoff"
    length_call = "plot_sequence_length_tradeoff_fn=study_plots.plot_sequence_length_tradeoff"

    cache_idx = content.find(cache_call)
    import_idx = content.find(plot_import)
    mmr_idx = content.find(mmr_call)
    length_idx = content.find(length_call)

    assert cache_idx >= 0, "Expected summarize_study_run to initialize Matplotlib cache."
    assert import_idx >= 0, "Expected summarize_study_run to lazily import study plot modules."
    assert mmr_idx >= 0, "Expected summarize_study_run to pass lazily-imported MMR plot function."
    assert length_idx >= 0, "Expected summarize_study_run to pass lazily-imported length plot function."
    assert cache_idx < import_idx, "Matplotlib cache must be initialized before loading study plot modules."
    assert import_idx < mmr_idx, "MMR plot dependency should be wired after the lazy import."
    assert import_idx < length_idx, "Length plot dependency should be wired after the lazy import."
