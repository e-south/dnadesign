"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_package_data_contract.py

Package-data contracts for concrete study status registries.

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


def test_studies_package_data_includes_concrete_runtime_resources() -> None:
    pyproject = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")

    assert '"dnadesign.studies" = [' in pyproject
    assert '"units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_template.py.txt"' in pyproject
    assert '"units/retron_hairpin_design/status/ops/status.registry.yaml"' in pyproject
    assert '"units/stress_ethanol_cipro_growth/operations/status/ops/status.registry.yaml"' in pyproject
    assert '"units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml"' in pyproject
    assert '"units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/configs/*.yaml"' in pyproject
    assert '"units/stress_ethanol_cipro_growth/response_window_observations/config/*.yaml"' in pyproject
    assert '"units/stress_ethanol_cipro_growth/response_window_observations/config/evidence/*.json"' in pyproject
    assert '"units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/config/*.yaml"' in pyproject
    assert '"studies/retron_hairpin_design/status/ops/status.registry.yaml"' not in pyproject
    assert '"*/status.registry.yaml"' not in pyproject
