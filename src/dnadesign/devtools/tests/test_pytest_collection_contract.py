"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_pytest_collection_contract.py

Contracts for repo-root pytest collection boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_pytest_config_excludes_generated_artifact_roots_from_recursion() -> None:
    pyproject = _repo_root() / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    pytest_options = config["tool"]["pytest"]["ini_options"]
    norecursedirs = set(pytest_options["norecursedirs"])

    assert "outputs" in norecursedirs
    assert "batch_results" in norecursedirs
    assert "runs" in norecursedirs
    assert ".pytest_cache" in norecursedirs


def test_pytest_root_collection_scope_stays_bound_to_repo_sources() -> None:
    pyproject = _repo_root() / "pyproject.toml"
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    pytest_options = config["tool"]["pytest"]["ini_options"]

    assert pytest_options["testpaths"] == ["src/dnadesign"]
