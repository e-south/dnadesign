"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_materialization_package_layout.py

Materialization-package layout regression tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_CLI_MATERIALIZATION_PACKAGES = {
    "atlas_semantic_profile",
    "biohub_esmc_sae_profile",
    "biohub_esmc_wt_mutation_scoring",
    "candidate_table",
    "contact_geometry",
    "contact_risk",
    "foldcheck_review",
    "foldcheck_report",
    "foldcheck_request",
    "manual_mask_authority",
    "mask_set",
    "proteinmpnn_request",
    "proteinmpnn_sample_ingest",
    "review_deliverables",
    "structure_preprocessing",
    "thread_plan",
}
_CONTACT_GEOMETRY_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "models.py",
    "paths.py",
    "pipeline.py",
    "rows.py",
    "structure_io.py",
    "writer.py",
}
_PROTEINMPNN_REQUEST_ROOT_FILES = {"__init__.py", "__main__.py", "cli.py", "constants.py", "models.py", "pipeline.py"}
_FOLDCHECK_REQUEST_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "models.py",
    "pipeline.py",
    "sequences.py",
}
_FOLDCHECK_REPORT_ROOT_FILES = {"__init__.py", "__main__.py", "cli.py", "constants.py", "pipeline.py"}
_FOLDCHECK_REVIEW_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "chimerax.py",
    "cli.py",
    "constants.py",
    "models.py",
    "notebook.py",
    "pdb_alignment.py",
    "pipeline.py",
    "plots.py",
    "ranking.py",
    "selection.py",
    "structure_overlay.py",
    "structures.py",
    "visuals.py",
}
_ATLAS_SEMANTIC_PROFILE_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "pipeline.py",
    "resume.py",
    "selection.py",
}
_BIOHUB_ESMC_SAE_PROFILE_ROOT_FILES = {
    "__init__.py",
    "__main__.py",
    "cli.py",
    "constants.py",
    "pipeline.py",
    "resume.py",
    "selection.py",
}


def test_cli_materializers_keep_cli_parsing_out_of_pipelines() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization"

    for package in sorted(_CLI_MATERIALIZATION_PACKAGES):
        package_root = source_root / package
        assert (package_root / "cli.py").is_file()
        pipeline_text = (package_root / "pipeline.py").read_text(encoding="utf-8")
        assert "argparse" not in pipeline_text
        assert "def main(" not in pipeline_text


def test_contact_geometry_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/contact_geometry"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_CONTACT_GEOMETRY_ROOT_FILES)
    assert "Bio.PDB" not in (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "pyarrow as pa" not in (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "np.stack" not in (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "write_geometry_profile" in (source_root / "writer.py").read_text(encoding="utf-8")
    assert "distance_matrix" in (source_root / "rows.py").read_text(encoding="utf-8")
    assert "MMCIFParser" in (source_root / "structure_io.py").read_text(encoding="utf-8")


def test_proteinmpnn_request_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/proteinmpnn_request"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_PROTEINMPNN_REQUEST_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "pyarrow" not in pipeline_text
    assert "hashlib" not in pipeline_text
    assert "protein_mpnn_run.py" not in pipeline_text
    assert "dnadesign.thread.adapters.proteinmpnn" in pipeline_text
    assert "build_request_manifest" in pipeline_text
    assert "export_chain_backbone" in pipeline_text


def test_foldcheck_request_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/foldcheck_request"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_FOLDCHECK_REQUEST_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "pyarrow" not in pipeline_text
    assert "re.compile" not in pipeline_text
    assert "dnadesign.thread.foldcheck" in pipeline_text
    assert "build_foldcheck_sequence_records" in pipeline_text


def test_foldcheck_report_materializer_uses_thread_colabfold_adapter() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/foldcheck_report"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_FOLDCHECK_REPORT_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    assert "pyarrow" not in pipeline_text
    assert "np." not in pipeline_text
    assert "dnadesign.thread.adapters.colabfold" in pipeline_text
    assert "dnadesign.thread.foldcheck" in pipeline_text


def test_foldcheck_review_materializer_uses_semantic_modules() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/foldcheck_review"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_FOLDCHECK_REVIEW_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    ranking_text = (source_root / "ranking.py").read_text(encoding="utf-8")
    structures_text = (source_root / "structures.py").read_text(encoding="utf-8")
    assert "argparse" not in pipeline_text
    assert "pyarrow" not in pipeline_text
    assert "ca_rmsd" not in pipeline_text
    assert "write_foldcheck_ranking" in ranking_text
    assert "stage_structure_panel" in structures_text
    assert "write_chimerax_script" in pipeline_text


def test_atlas_semantic_profile_materializer_uses_thread_atlas_adapter() -> None:
    source_root = repo_root() / _PACKAGE_ROOT / "operations/materialization/atlas_semantic_profile"

    assert sorted(path.name for path in source_root.glob("*.py")) == sorted(_ATLAS_SEMANTIC_PROFILE_ROOT_FILES)
    pipeline_text = (source_root / "pipeline.py").read_text(encoding="utf-8")
    resume_text = (source_root / "resume.py").read_text(encoding="utf-8")
    selection_text = (source_root / "selection.py").read_text(encoding="utf-8")
    assert "pyarrow" not in pipeline_text
    assert "dnadesign.thread.adapters.esm_atlas" in pipeline_text
    assert "dnadesign.thread.structure_predictions" in pipeline_text
    assert "fold_on_miss=allow_fold_on_miss" in pipeline_text
    assert "pyarrow.parquet" in resume_text
    assert "pyarrow.parquet" in selection_text
    assert 'row.get("status", "")' in selection_text
    assert '"accepted"' in selection_text


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
