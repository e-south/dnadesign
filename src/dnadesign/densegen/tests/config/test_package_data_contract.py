"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_package_data_contract.py

Package-data contracts for DenseGen packaged workspace templates.

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


def test_densegen_package_data_includes_public_workspace_resources() -> None:
    repo_root = _repo_root()
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    paths = project["tool"]["setuptools"]["package-data"]["dnadesign.densegen"]
    required = {
        "workspaces/_shared/workspace_runbook_flow.sh",
        "workspaces/demo_tfbs_baseline/config.yaml",
        "workspaces/demo_tfbs_baseline/inputs/sites.csv",
        "workspaces/demo_sampling_baseline/config.yaml",
        "workspaces/demo_dense_array_showcase/config.yaml",
    }
    assert required <= set(paths)
    assert all(not any(token in path for token in "*?[") for path in paths)
    assert all((repo_root / "src/dnadesign/densegen" / path).is_file() for path in paths)


def test_declared_templates_are_packaged() -> None:
    from dnadesign.densegen.src.cli.workspace_sources import PACKAGED_WORKSPACE_IDS

    project = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    paths = project["tool"]["setuptools"]["package-data"]["dnadesign.densegen"]
    assert all(f"workspaces/{workspace}/config.yaml" in paths for workspace in PACKAGED_WORKSPACE_IDS)
