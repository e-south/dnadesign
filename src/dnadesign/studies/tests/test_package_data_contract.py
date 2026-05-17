"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_package_data_contract.py

Package-data contracts for studies status adapter registries.

Module Author(s): Eric J. South
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


def test_studies_package_data_includes_status_adapter_ops_status_registries() -> None:
    pyproject = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")

    assert '"dnadesign.studies" = [' in pyproject
    assert '"status_adapters/*/ops/status.registry.yaml"' in pyproject
    assert '"stress_ethanol_cipro_growth/opal_batch0/sampling.yaml"' in pyproject
    assert '"*/status.registry.yaml"' not in pyproject
