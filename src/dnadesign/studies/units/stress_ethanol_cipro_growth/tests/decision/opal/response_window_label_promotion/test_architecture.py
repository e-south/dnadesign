"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_window_label_promotion/test_architecture.py

Architecture guards for the study-owned OPAL label publisher.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion"
)


def _module_body_line_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0] == '"""':
        closing_line = next(index for index, line in enumerate(lines[1:], start=1) if line == '"""')
        return len(lines) - closing_line - 1
    return len(lines)


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


@pytest.mark.parametrize("module_path", sorted(PACKAGE_ROOT.glob("*.py")), ids=lambda path: path.name)
def test_label_promotion_modules_stay_bounded(module_path: Path) -> None:
    limit = 180 if module_path.name == "publisher.py" else 220
    assert _module_body_line_count(module_path) <= limit


@pytest.mark.parametrize("module_path", sorted(PACKAGE_ROOT.glob("*.py")), ids=lambda path: path.name)
def test_label_promotion_respects_package_boundaries(module_path: Path) -> None:
    offenders = [
        f"{module_path.name}:{line}:{module}"
        for line, module in _imported_modules(module_path)
        if module == "reader"
        or module.startswith("reader.")
        or "response_metastudy" in module.split(".")
        or module == "dnadesign.opal.src"
        or module.startswith("dnadesign.opal.src.")
    ]
    assert offenders == []


def test_label_promotion_uses_a_public_opal_surface() -> None:
    opal_imports = []
    for module_path in PACKAGE_ROOT.glob("*.py"):
        opal_imports.extend(
            module
            for _, module in _imported_modules(module_path)
            if module == "dnadesign.opal" or module.startswith("dnadesign.opal.")
        )
    assert opal_imports
    assert all(
        module == "dnadesign.opal" or module == "dnadesign.opal.api" or module.startswith("dnadesign.opal.api.")
        for module in opal_imports
    )
