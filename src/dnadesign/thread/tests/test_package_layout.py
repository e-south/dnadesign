"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/test_package_layout.py

Package-layout regression tests for generic thread workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

_THREAD_ROOT_FILES = {"__init__.py"}
_THREAD_DIRECTORIES = {
    "adapters",
    "assets",
    "candidates",
    "docs",
    "foldcheck",
    "structure_predictions",
    "structure_views",
    "tests",
}
_PROTEINMPNN_FILES = {
    "__init__.py",
    "execution.py",
    "execution_preflight.py",
    "hashing.py",
    "manifest.py",
    "models.py",
    "positions.py",
    "samples.py",
    "sidecars.py",
    "structure.py",
    "validation.py",
}
_COLABFOLD_FILES = {"__init__.py", "index.py", "manifest.py", "metrics.py", "outputs.py"}
_BIOHUB_ESMC_FILES = {
    "__init__.py",
    "auth.py",
    "client.py",
    "encoded.py",
    "feature_descriptions.py",
    "hashes.py",
    "models.py",
    "normalize.py",
    "tables.py",
}
_ESM_ATLAS_FILES = {
    "__init__.py",
    "client.py",
    "hashes.py",
    "models.py",
    "normalize.py",
    "structure_predictions.py",
    "tables.py",
}
_FOLDCHECK_FILES = {"__init__.py", "hashes.py", "models.py", "report.py", "request.py", "subset.py"}
_STRUCTURE_PREDICTION_FILES = {"__init__.py", "hashes.py", "models.py", "registry.py"}
_STRUCTURE_VIEW_FILES = {
    "__init__.py",
    "html.py",
    "models.py",
    "nucleic_geometry.py",
    "styles.py",
}


def test_thread_root_is_small_public_tool_surface() -> None:
    root = _repo_root() / "src/dnadesign/thread"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_THREAD_ROOT_FILES)
    assert sorted(path.name for path in root.iterdir() if path.is_dir() and path.name != "__pycache__") == sorted(
        _THREAD_DIRECTORIES
    )


def test_proteinmpnn_adapter_owns_generic_request_mechanics() -> None:
    root = _repo_root() / "src/dnadesign/thread/adapters/proteinmpnn"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_PROTEINMPNN_FILES)
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "eco1" not in text
        assert "ec86" not in text
        assert "mestre" not in text
        assert "wang" not in text
    assert "ProteinMPNN" in (root / "validation.py").read_text(encoding="utf-8")
    assert "resolve_manifest_sidecar_path" in (root / "sidecars.py").read_text(encoding="utf-8")


def test_colabfold_adapter_owns_generic_result_normalization() -> None:
    root = _repo_root() / "src/dnadesign/thread/adapters/colabfold"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_COLABFOLD_FILES)
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "eco1" not in text
        assert "ec86" not in text
        assert "mestre" not in text
        assert "wang" not in text
    assert "colabfold" in (root / "outputs.py").read_text(encoding="utf-8").lower()


def test_biohub_esmc_adapter_owns_generic_query_time_sae_normalization() -> None:
    root = _repo_root() / "src/dnadesign/thread/adapters/biohub_esmc"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_BIOHUB_ESMC_FILES)
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "eco1" not in text
        assert "ec86" not in text
        assert "mestre" not in text
        assert "wang" not in text
    assert "authorization" not in (root / "tables.py").read_text(encoding="utf-8").lower()
    assert "thread.biohub_esmc.sae_profile" in (root / "tables.py").read_text(encoding="utf-8")


def test_esm_atlas_adapter_owns_generic_semantic_annotation() -> None:
    root = _repo_root() / "src/dnadesign/thread/adapters/esm_atlas"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_ESM_ATLAS_FILES)
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "eco1" not in text
        assert "ec86" not in text
        assert "mestre" not in text
        assert "wang" not in text
    assert "folded_on_demand" in (root / "normalize.py").read_text(encoding="utf-8")
    assert "thread.esm_atlas.semantic_profile" in (root / "tables.py").read_text(encoding="utf-8")


def test_foldcheck_package_owns_generic_fold_report_contracts() -> None:
    root = _repo_root() / "src/dnadesign/thread/foldcheck"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_FOLDCHECK_FILES)
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "eco1" not in text
        assert "ec86" not in text
        assert "mestre" not in text
        assert "wang" not in text
    assert "thread.foldcheck_report" in (root / "report.py").read_text(encoding="utf-8")
    assert "thread.foldcheck_external_run_manifest" in (root / "subset.py").read_text(encoding="utf-8")


def test_structure_predictions_package_owns_generic_structure_registry_contracts() -> None:
    root = _repo_root() / "src/dnadesign/thread/structure_predictions"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_STRUCTURE_PREDICTION_FILES)
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "eco1" not in text
        assert "ec86" not in text
        assert "mestre" not in text
        assert "wang" not in text
    assert "thread.structure_predictions.registry" in (root / "registry.py").read_text(encoding="utf-8")


def test_structure_views_package_owns_generic_browser_view_contracts() -> None:
    root = _repo_root() / "src/dnadesign/thread/structure_views"

    assert sorted(path.name for path in root.glob("*.py")) == sorted(_STRUCTURE_VIEW_FILES)
    assert sorted(path.name for path in (root / "backends").glob("*.py")) == ["__init__.py", "py3dmol.py"]
    for path in [*root.glob("*.py"), *(root / "backends").glob("*.py")]:
        text = path.read_text(encoding="utf-8").lower()
        assert "eco1" not in text
        assert "ec86" not in text
        assert "mestre" not in text
        assert "wang" not in text
    assert "StructureViewSpec" in (root / "models.py").read_text(encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
