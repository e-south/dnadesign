"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_core_import_contract.py

Contract checks for core import boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[2] / "src" / "core"
BANNED_IMPORT_PREFIXES = (
    "dnadesign.cruncher.artifacts",
    "dnadesign.cruncher.cli",
    "os",
    "pathlib",
    "shutil",
    "glob",
    "tempfile",
)


def _iter_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _is_banned(module: str) -> bool:
    for banned in BANNED_IMPORT_PREFIXES:
        if module == banned or module.startswith(f"{banned}."):
            return True
    return False


def test_core_imports_stay_pure() -> None:
    violations: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*.py")):
        for module in _iter_imports(path):
            if _is_banned(module):
                violations.append(f"{path.relative_to(CORE_ROOT.parent)} -> {module}")
    assert not violations, "Core imports must stay pure:\n" + "\n".join(violations)
