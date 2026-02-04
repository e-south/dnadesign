"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_core_import_contract.py

Enforce core purity by banning filesystem and UX imports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

BANNED_PREFIXES = (
    "dnadesign.cruncher.artifacts",
    "dnadesign.cruncher.cli",
    "pathlib",
    "os",
    "shutil",
    "tempfile",
    "tqdm",
)


def _core_root() -> Path:
    tests_dir = Path(__file__).resolve().parent
    return tests_dir.parent / "src" / "core"


def _iter_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                continue
            if node.module:
                imports.append(node.module)
    return imports


def test_core_has_no_io_or_cli_imports() -> None:
    core_root = _core_root()
    offenders: list[str] = []
    for path in sorted(core_root.rglob("*.py")):
        for name in _iter_imports(path):
            if name.startswith(BANNED_PREFIXES):
                offenders.append(f"{path.relative_to(core_root)}: {name}")
    assert not offenders, "\n".join(offenders)
