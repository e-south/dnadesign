"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_package_data_contract.py

Package-data contracts for DenseGen packaged workspace templates.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_densegen_package_data_uses_extensible_workspace_globs() -> None:
    pyproject = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
    assert '"dnadesign.densegen" = [' in pyproject
    assert '"workspaces/_shared/*.sh"' in pyproject
    assert '"workspaces/*/*.md"' in pyproject
    assert '"workspaces/*/*.sh"' in pyproject
    assert '"workspaces/*/*.yaml"' in pyproject
    assert '"workspaces/*/inputs/*"' in pyproject
    assert '"workspaces/demo_tfbs_baseline/*.yaml"' not in pyproject
    assert '"workspaces/demo_sampling_baseline/*.yaml"' not in pyproject
    assert '"workspaces/study_constitutive_sigma_panel/*.yaml"' not in pyproject
    assert '"workspaces/study_stress_ethanol_cipro/*.yaml"' not in pyproject
