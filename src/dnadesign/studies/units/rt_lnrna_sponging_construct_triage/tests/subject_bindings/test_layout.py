"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/subject_bindings/test_layout.py

Architecture boundaries for the RT-lnRNA subject-binding package.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2] / "subject_bindings"


def test_subject_binding_package_has_semantic_modules() -> None:
    observed = {path.name for path in _package_root().glob("*.py")}
    assert observed == {
        "__init__.py",
        "authorities.py",
        "contracts.py",
        "loader.py",
        "projection.py",
        "query.py",
        "registry.py",
        "sources.py",
        "subjects.py",
        "validation.py",
    }


def test_loader_is_thin_public_orchestration() -> None:
    loader_path = _package_root() / "loader.py"
    source = loader_path.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 150
    tree = ast.parse(source)
    relative_imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level == 1}
    assert relative_imports == {"contracts", "registry"}
