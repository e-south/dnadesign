"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_biohub_materialization_layout.py

Biohub materialization-package layout regression tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_BIOHUB_ESMC_SAE_PROFILE_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "pipeline.py",
    "resume.py",
    "run_contract.py",
    "selection.py",
}


def test_biohub_esmc_sae_profile_materializer_uses_thread_biohub_adapter() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/biohub_esmc_sae_profile"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_BIOHUB_ESMC_SAE_PROFILE_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    resume_text = (source_root / "resume.py").read_text(encoding="utf-8")
    selection_text = (source_root / "selection.py").read_text(encoding="utf-8")
    assert "argparse" not in pipeline_text
    assert "pyarrow" not in pipeline_text
    assert "dnadesign.thread.adapters.biohub_esmc" in pipeline_text
    assert "Bearer " not in pipeline_text
    assert "pyarrow.parquet" in resume_text
    assert "pyarrow.parquet" in selection_text
    assert 'row.get("status")' in selection_text
    assert '"accepted"' in selection_text
