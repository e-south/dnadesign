"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_selection_materialization_layout.py

Panel-selection materialization layout tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
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
    "cli.py",
    "constants.py",
    "feasibility.py",
    "io.py",
    "models.py",
    "panel.py",
    "pipeline.py",
    "plot_support.py",
    "plots.py",
    "review_axes.py",
    "triage.py",
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


def test_selection_readiness_materializer_keeps_decision_logic_decomposed() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/selection_readiness"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_SELECTION_READINESS_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    feasibility_text = (source_root / "feasibility.py").read_text(encoding="utf-8")
    triage_text = (source_root / "triage.py").read_text(encoding="utf-8")
    panel_text = (source_root / "panel.py").read_text(encoding="utf-8")
    plots_text = (source_root / "plots.py").read_text(encoding="utf-8")
    review_axes_text = (source_root / "review_axes.py").read_text(encoding="utf-8")
    assert "argparse" not in pipeline_text
    assert "matplotlib" not in pipeline_text
    assert "pyarrow" not in pipeline_text
    assert "build_feasibility_rows" in feasibility_text
    assert "build_triage_rows" in triage_text
    assert "build_selection_panel_rows" in panel_text
    assert "write_selection_readiness_plots" in plots_text
    assert "load_fasta_records" in review_axes_text
    assert "processivity_score" not in panel_text
    assert "activity_score" not in panel_text
