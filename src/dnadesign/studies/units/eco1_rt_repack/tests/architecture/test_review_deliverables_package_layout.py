"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_review_deliverables_package_layout.py

Review-deliverables package-layout regression tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "biohub_esmc_model_provenance.py",
    "biohub_esmc_sae_activation_pattern.py",
    "biohub_esmc_sae_audit.py",
    "biohub_esmc_sae_heatmap_manifest.py",
    "biohub_esmc_sae_interpretation.py",
    "biohub_esmc_sae_interpretation_shared.py",
    "biohub_esmc_sae_tables.py",
    "biohub_esmc_sequence_preference.py",
    "biohub_esmc_sequence_preference_model_agreement.py",
    "biohub_esmc_sequence_preference_plot.py",
    "cli.py",
    "constants.py",
    "esmc_model_check.py",
    "esmc_model_check_metadata.py",
    "manifest.py",
    "mask_rows.py",
    "mask_tracks.py",
    "mask_structure_browser.py",
    "models.py",
    "msa_panel.py",
    "msa_panel_annotations.py",
    "msa_panel_data.py",
    "msa_panel_layout.py",
    "notebook.py",
    "notebook_runtime.py",
    "notebook_sae_features.py",
    "notebook_sequences.py",
    "notebook_selection_panel.py",
    "notebook_selection_summary.py",
    "notebook_structure_browser.py",
    "notebook_structure_dashboard.py",
    "notebook_structure_rows.py",
    "notebook_visuals.py",
    "pipeline.py",
    "plain_titles.py",
    "sae_structure_browser.py",
    "selection_readiness.py",
    "structure_browser.py",
    "structure_browser_common.py",
    "structure_sequences.py",
}


def test_review_deliverables_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/review_deliverables"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "argparse" not in pipeline_text
    assert "matplotlib" not in pipeline_text
    assert "pyarrow" not in pipeline_text
    assert "write_msa_plurality_mask_panel" in pipeline_text
    assert "write_linear_mask_tracks" not in pipeline_text
    assert "write_design_class_mask_overview" not in pipeline_text
    assert "write_proteinmpnn_diversity_panels" not in pipeline_text
    assert "write_esmc_model_check_panels" in pipeline_text
    assert "write_biohub_esmc_sae_interpretation_panels" in pipeline_text
    assert "write_interactive_structure_browser_manifest" in pipeline_text
    assert "linked_selection_readiness_rows" in pipeline_text
    assert "read_mask_residues" in pipeline_text


def test_review_deliverables_facade_does_not_import_plotting_stack() -> None:
    code = """
    import sys
    import dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables as module

    assert module.__all__ == ["MaterializedReviewDeliverables", "materialize_review_deliverables"]
    assert "matplotlib" not in sys.modules
    assert "matplotlib.pyplot" not in sys.modules
    """
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)
