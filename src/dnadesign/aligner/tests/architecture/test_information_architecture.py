"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/tests/architecture/test_information_architecture.py

Module support for dnadesign.aligner.tests.architecture.test_information_architecture.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ALIGNER_ROOT = Path(__file__).resolve().parents[2]


def test_aligner_root_is_entrypoint_only() -> None:
    allowed_root_files = {
        "__init__.py",
        "README.md",
    }
    root_py_files = {path.name for path in ALIGNER_ROOT.glob("*.py")}
    source_files = {path.name for path in ALIGNER_ROOT.iterdir() if path.is_file() and not path.name.startswith(".")}

    assert root_py_files <= {"__init__.py"}
    assert source_files <= allowed_root_files


def test_semantic_packages_exist() -> None:
    expected_packages = [
        ALIGNER_ROOT / "pairwise",
        ALIGNER_ROOT / "msa",
        ALIGNER_ROOT / "msa" / "backends",
        ALIGNER_ROOT / "msa" / "bundles",
        ALIGNER_ROOT / "msa" / "visualization",
        ALIGNER_ROOT / "msa" / "visualization" / "contracts",
        ALIGNER_ROOT / "msa" / "visualization" / "materialization",
        ALIGNER_ROOT / "msa" / "visualization" / "renderers",
        ALIGNER_ROOT / "tests" / "pairwise",
        ALIGNER_ROOT / "tests" / "msa",
        ALIGNER_ROOT / "tests" / "msa" / "visualization",
        ALIGNER_ROOT / "tests" / "architecture",
    ]

    for package in expected_packages:
        assert package.is_dir(), f"missing semantic package {package}"


def test_no_legacy_flat_pairwise_modules_remain() -> None:
    legacy_modules = {
        "align.py",
        "cache.py",
        "matrix.py",
        "metrics.py",
        "utils.py",
        "example.py",
    }

    existing = {path.name for path in ALIGNER_ROOT.iterdir()}
    assert existing.isdisjoint(legacy_modules)


def test_msa_backend_process_execution_has_one_contract_surface() -> None:
    backend_root = ALIGNER_ROOT / "msa" / "backends"

    assert (backend_root / "execution.py").is_file()
    for backend_module_name in ("mafft.py", "clustalo.py"):
        text = (backend_root / backend_module_name).read_text(encoding="utf-8")
        assert "run_staged_backend_alignment" in text
        assert "write_bundle_manifest" not in text
        assert "perf_counter" not in text
        assert "uuid4" not in text
        assert "hashlib" not in text
