"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_triage_contract.py

Candidate triage contract tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.triage import (
    _hard_gate_status,
)


def test_review_band_without_other_blockers_is_manual_reserve() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "review_band"},
        feasibility={"feasibility_status": "feasible"},
    )

    assert status == "needs_review"
    assert reasons == ["fold_review_class_requires_manual_review"]


def test_review_band_does_not_override_hard_blockers() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 1},
        fold={"foldcheck_status": "accepted", "review_class": "review_band"},
        feasibility={"feasibility_status": "feasible"},
    )

    assert status == "ineligible"
    assert "protected_mutation_violation" in reasons
    assert "fold_review_class_requires_manual_review" in reasons
