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


def test_review_band_without_other_blockers_is_ineligible() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "review_band"},
        feasibility={"feasibility_status": "feasible"},
        local_structure_review={"local_structure_gate_status": "passed"},
    )

    assert status == "ineligible"
    assert reasons == ["fold_review_class_not_allowed"]


def test_review_band_does_not_override_hard_blockers() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 1},
        fold={"foldcheck_status": "accepted", "review_class": "review_band"},
        feasibility={"feasibility_status": "feasible"},
        local_structure_review={"local_structure_gate_status": "passed"},
    )

    assert status == "ineligible"
    assert "protected_mutation_violation" in reasons
    assert "fold_review_class_not_allowed" in reasons


def test_catalytic_or_direct_contact_mutation_fails_hard_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        feasibility={"feasibility_status": "feasible"},
        review_axes={"catalytic_or_direct_contact_mutation_count": 1},
        local_structure_review={"local_structure_gate_status": "passed"},
    )

    assert status == "ineligible"
    assert "catalytic_or_direct_contact_mutation" in reasons


def test_missing_local_structure_review_fails_hard_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        feasibility={"feasibility_status": "feasible"},
        local_structure_review=None,
    )

    assert status == "missing_inputs"
    assert reasons == ["missing_local_structure_review"]


def test_unavailable_local_structure_review_fails_hard_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        feasibility={"feasibility_status": "feasible"},
        local_structure_review={
            "local_structure_gate_status": "unavailable",
            "local_structure_gate_failure_reasons_json": '["thumb_contact_track_context:model_structure_missing"]',
        },
    )

    assert status == "missing_inputs"
    assert reasons == ["local_structure_gate_unavailable"]


def test_local_structure_threshold_excess_fails_hard_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        feasibility={"feasibility_status": "feasible"},
        local_structure_review={
            "local_structure_gate_status": "threshold_exceeded",
            "local_structure_gate_failure_reasons_json": '["thumb_contact_track_context:local_ca_rmsd 3.20 > 3.00"]',
        },
    )

    assert status == "ineligible"
    assert reasons == ["local_structure_threshold_exceeded"]
