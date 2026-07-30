"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/architecture/test_lazy_public_facades.py

Static-symbol coverage tests for lazy public package facades.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_LAZY_PUBLIC_FACADES = (
    "dnadesign.baserender",
    "dnadesign.baserender.src",
    "dnadesign.baserender.src.adapters",
    "dnadesign.baserender.src.public",
    "dnadesign.contracts",
    "dnadesign.contracts.folding",
    "dnadesign.contracts.sequence",
    "dnadesign.contracts.visual",
    "dnadesign.folding",
    "dnadesign.folding.src",
    "dnadesign.notify",
)


def _type_checking_imports(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    imported: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name) or node.test.id != "TYPE_CHECKING":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                imported.update(alias.asname or alias.name for alias in child.names)
    return imported


@pytest.mark.parametrize("module_name", _LAZY_PUBLIC_FACADES)
def test_lazy_public_facade_declares_every_export_for_static_analysis(module_name: str) -> None:
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__ or "")

    assert module_path.is_file()
    assert set(module.__all__) <= _type_checking_imports(module_path)
