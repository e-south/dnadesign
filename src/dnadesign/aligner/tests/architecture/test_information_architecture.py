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
        ALIGNER_ROOT / "tests" / "pairwise",
        ALIGNER_ROOT / "tests" / "msa",
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
