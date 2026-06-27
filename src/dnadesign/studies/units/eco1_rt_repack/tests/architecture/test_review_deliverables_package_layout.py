"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_review_deliverables_package_layout.py

Review-deliverables package-layout regression tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "manifest.py",
    "mask_rows.py",
    "mask_tracks.py",
    "models.py",
    "msa_panel.py",
    "notebook.py",
    "pipeline.py",
    "proteinmpnn_diversity.py",
    "rendering.py",
}


def test_review_deliverables_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/review_deliverables"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "argparse" not in pipeline_text
    assert "matplotlib" not in pipeline_text
    assert "pyarrow" not in pipeline_text
    assert "write_msa_plurality_mask_panel" in pipeline_text
    assert "write_linear_mask_tracks" in pipeline_text
    assert "write_proteinmpnn_diversity_panels" in pipeline_text
    assert "read_mask_residues" in pipeline_text
