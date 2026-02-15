"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_baserender_import_boundary.py

Enforce Cruncher->Baserender import boundary (public API only).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_runtime_modules_do_not_import_baserender_internals() -> None:
    runtime_root = _package_root() / "src"
    disallowed = "dnadesign.baserender" + ".src."
    violations: list[str] = []
    for path in sorted(runtime_root.rglob("*.py")):
        text = path.read_text()
        if disallowed in text:
            violations.append(str(path))
    assert not violations, f"Found disallowed deep baserender imports: {violations}"


def test_elites_showcase_uses_public_baserender_imports() -> None:
    file_path = _package_root() / "src" / "analysis" / "plots" / "elites_showcase.py"
    text = file_path.read_text()
    assert "from dnadesign.baserender import " in text
