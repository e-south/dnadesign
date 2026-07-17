"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_wang_alpha1_review_contract.py

Wang alpha-1 annotation contract tests for Eco1 RT selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.triage import (
    _selection_candidate_fields,
    _wang_alpha1_review_fields,
)


def test_wang_alpha1_r13_mutation_is_annotation_not_selection_gate() -> None:
    fields = _selection_candidate_fields(
        {
            "hard_gate_status": "eligible",
            "wang_alpha1_r13_mutation_count": 1,
            "nucleic_acid_facing_acidic_gain_count": 0,
            "proximal_review_unobserved_mutation_count": 0,
        }
    )

    assert fields["selection_contract_pass"] is True
    assert fields["wang_alpha1_r13_review_status"] == "substituted"
    assert "wang_alpha1_r13_mutation" not in fields["selection_contract_failure_reasons_json"]


def test_wang_alpha1_r13_wild_type_passes_when_other_contract_checks_pass() -> None:
    fields = _selection_candidate_fields(
        {
            "hard_gate_status": "eligible",
            "wang_alpha1_r13_mutation_count": 0,
            "nucleic_acid_facing_acidic_gain_count": 0,
            "proximal_review_unobserved_mutation_count": 0,
        }
    )

    assert fields["selection_contract_pass"] is True
    assert fields["wang_alpha1_r13_review_status"] == "retained_wt"


def test_wang_alpha1_review_records_exact_contact_substitutions_without_claiming_monomeric_state() -> None:
    fields = _wang_alpha1_review_fields(["F10E", "R13K", "A20G"])

    assert fields == {
        "wang_alpha1_f10_substitution": "F10E",
        "wang_alpha1_r13_substitution": "R13K",
        "wang_alpha1_cross_protomer_contact_mutation_count": 2,
        "wang_r13a_interface_disruption_evidence_match": False,
        "rt_msdna_oligomeric_state_review_status": "not_established",
    }


def test_wang_alpha1_review_marks_only_the_tested_r13a_substitution_as_an_evidence_match() -> None:
    fields = _wang_alpha1_review_fields(["R13A"])

    assert fields["wang_alpha1_f10_substitution"] == "WT"
    assert fields["wang_alpha1_r13_substitution"] == "R13A"
    assert fields["wang_r13a_interface_disruption_evidence_match"] is True
    assert fields["rt_msdna_oligomeric_state_review_status"] == "wang_r13a_interface_disruption_match"
