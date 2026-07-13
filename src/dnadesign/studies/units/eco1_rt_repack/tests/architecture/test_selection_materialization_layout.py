"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_selection_materialization_layout.py

Panel-selection materialization layout tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_SHARED_MATERIALIZATION_ROOT_FILES = {
    "__init__.py",
    "rendering.py",
    "rt_annotation_context.py",
    "rt_plot_annotations.py",
}
_SAE_WINDOW_SUMMARY_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "io.py",
    "models.py",
    "pipeline.py",
    "vectors.py",
    "windows.py",
}
_SELECTION_READINESS_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "chemistry_balance.py",
    "cli.py",
    "constants.py",
    "handoff_readiness.py",
    "io.py",
    "local_structure.py",
    "local_structure_plot.py",
    "local_structure_regions.py",
    "local_structure_sensitivity.py",
    "manifest.py",
    "models.py",
    "mutation_distance.py",
    "mutation_distance_plot.py",
    "panel.py",
    "panel_contract.py",
    "panel_ranking.py",
    "panel_rows.py",
    "panel_trace.py",
    "pipeline.py",
    "plot_support.py",
    "plots.py",
    "region_msa_support.py",
    "region_msa_support_plot.py",
    "review_axis_contracts.py",
    "review_axes.py",
    "regional_plots.py",
    "sankey_plot.py",
    "sequence_export.py",
    "selection_policy_context.py",
    "selection_summary.py",
    "triage.py",
    "visual_inventory.py",
}


def test_sae_window_summary_materializer_keeps_vector_math_separate_from_pipeline() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/sae_window_summary"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_SAE_WINDOW_SUMMARY_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    vectors_text = (source_root / "vectors.py").read_text(encoding="utf-8")
    io_text = (source_root / "io.py").read_text(encoding="utf-8")
    assert "argparse" not in pipeline_text
    assert "polars" not in pipeline_text
    assert "pyarrow" not in pipeline_text
    assert "pl.scan_parquet" in vectors_text
    assert "pyarrow.parquet" in io_text


def test_shared_materialization_helpers_are_neutral() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/shared"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_SHARED_MATERIALIZATION_ROOT_FILES)
    for path in source_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "selection_readiness" not in text
        assert "review_deliverables" not in text


def test_selection_readiness_materializer_keeps_decision_logic_decomposed() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/selection_readiness"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_SELECTION_READINESS_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    triage_text = (source_root / "triage.py").read_text(encoding="utf-8")
    panel_text = (source_root / "panel.py").read_text(encoding="utf-8")
    plots_text = (source_root / "plots.py").read_text(encoding="utf-8")
    review_axes_text = (source_root / "review_axes.py").read_text(encoding="utf-8")
    sequence_export_text = (source_root / "sequence_export.py").read_text(encoding="utf-8")
    selection_summary_text = (source_root / "selection_summary.py").read_text(encoding="utf-8")
    for path in source_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "operations.materialization.review_deliverables" not in text
        assert "..review_deliverables" not in text
    assert "argparse" not in pipeline_text
    assert "csv.DictWriter" not in pipeline_text
    assert "hashlib" not in pipeline_text
    assert "matplotlib" not in pipeline_text
    assert "pyarrow" not in pipeline_text
    assert "build_triage_rows" in triage_text
    assert "build_selected_panel_rows" in panel_text
    assert "write_selection_readiness_plots" in plots_text
    assert "load_fasta_records" in review_axes_text
    assert "csv.DictWriter" in sequence_export_text
    assert "hashlib.sha256" in sequence_export_text
    assert "build_selection_summary" in selection_summary_text
    assert "processivity_score" not in panel_text
    assert "activity_score" not in panel_text


def test_selection_readiness_facade_does_not_import_plotting_stack() -> None:
    code = """
    import sys
    import dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness as module

    assert module.__all__ == [
        "MaterializedSelectionReadiness",
        "NA_FACING_CHEMISTRY_METRICS",
        "materialize_selection_readiness",
    ]
    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules
    """
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)
