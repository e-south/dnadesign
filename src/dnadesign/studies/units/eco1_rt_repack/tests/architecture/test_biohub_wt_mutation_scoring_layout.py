"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_biohub_wt_mutation_scoring_layout.py

Architecture checks for Eco1 WT ESMC mutation scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_BIOHUB_ESMC_WT_MUTATION_SCORING_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "mask_join.py",
    "pipeline.py",
    "plot_context.py",
    "resume.py",
    "selection.py",
}


def test_biohub_esmc_wt_mutation_scoring_materializer_uses_permuter_grid() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/biohub_esmc_wt_mutation_scoring"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_BIOHUB_ESMC_WT_MUTATION_SCORING_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    resume_text = (source_root / "resume.py").read_text(encoding="utf-8")
    selection_text = (source_root / "selection.py").read_text(encoding="utf-8")
    mask_join_text = (source_root / "mask_join.py").read_text(encoding="utf-8")
    plot_context_text = (source_root / "plot_context.py").read_text(encoding="utf-8")
    assert "argparse" not in pipeline_text
    assert "pyarrow" not in pipeline_text
    assert "ProteinDmsRequest" in pipeline_text
    assert "build_masked_marginal_jobs" in pipeline_text
    assert "sequence_logits_for_sequence" in pipeline_text
    assert "Bearer " not in pipeline_text
    assert "pyarrow.parquet" in resume_text
    assert "select_fold_accepted_biohub_esmc_sequences" in selection_text
    assert "thread.mask_set" not in pipeline_text
    assert "thread.mask_set" in mask_join_text
    assert "rt_interval_review_label" in plot_context_text
