"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/package/test_public_import_boundary.py

Notify runtime import-boundary tests for sibling tool public APIs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _runtime_root() -> Path:
    current = Path(__file__).resolve()
    return next(
        parent
        for parent in current.parents
        if (parent / "core" / "contracts.py").exists() and (parent / "runtime").is_dir()
    )


def test_notify_runtime_does_not_import_densegen_or_usr_internal_paths() -> None:
    disallowed = ("dnadesign.densegen.src.", "dnadesign.usr.src.")
    violations: list[str] = []
    for path in sorted(_runtime_root().rglob("*.py")):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in disallowed):
            violations.append(str(path))
    assert not violations, f"Found disallowed internal imports in notify runtime: {violations}"
