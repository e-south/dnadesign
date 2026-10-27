"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/package/test_public_api_contracts.py

Package-surface contracts for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.latentdna.src import workspaces


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_workspace_package_keeps_narrow_explicit_surface() -> None:
    assert workspaces.__all__ == [
        "CoordinateSpaceError",
        "WorkspaceContext",
        "WorkspaceValidationError",
        "load_workspace_config",
    ]


def test_latentdna_src_has_no_execution_helper_barrel() -> None:
    assert not (_repo_root() / "src" / "dnadesign" / "latentdna" / "src" / "api.py").exists()
