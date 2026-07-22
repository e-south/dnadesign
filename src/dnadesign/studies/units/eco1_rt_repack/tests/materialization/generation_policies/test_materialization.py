"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies/test_materialization.py

Materialization test surface contract for Eco1 generation policies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def test_generation_policy_materialization_tests_are_split_by_contract() -> None:
    test_root = Path(__file__).parent

    assert (test_root / "test_policy_manifests.py").is_file()
    assert (test_root / "test_request_materialization.py").is_file()
    assert (test_root / "test_candidate_outputs.py").is_file()
