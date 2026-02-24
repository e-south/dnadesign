"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py

Cluster runtime import-boundary tests for USR public APIs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _runtime_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


def test_cluster_runtime_does_not_import_usr_internal_paths() -> None:
    disallowed = "dnadesign.usr.src."
    violations: list[str] = []
    for path in sorted(_runtime_root().rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if disallowed in text:
            violations.append(str(path))
    assert not violations, f"Found disallowed USR internal imports in cluster runtime: {violations}"
