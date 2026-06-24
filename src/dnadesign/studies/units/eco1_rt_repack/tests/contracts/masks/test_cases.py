"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/masks/test_cases.py

Conservative mask-case contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks import (
    validate_conservative_mask_cases_payload,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import load_yaml


def test_conservative_mask_cases_keep_required_fail_fast_gates() -> None:
    cases = load_yaml("docs/studies/eco1_rt_repack/operations/contract/fixtures/thread/conservative_mask_cases.yaml")
    cases["cases"] = [case for case in cases["cases"] if case["id"] != "reject_missing_contact_threshold"]

    report = validate_conservative_mask_cases_payload(cases)

    assert report.passed is False
    check_ids = {issue.check_id for issue in report.issues}
    assert "eco1_rt.mask_cases.missing_required_case" in check_ids
