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


def _passed_local_structure_review(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "local_structure_gate_status": "passed",
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom": 1.0,
        "local_structure_retron_x_naxxh_context_ca_rmsd_angstrom": 1.0,
        "local_structure_retron_y_vtg_context_ca_rmsd_angstrom": 1.0,
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom": 1.0,
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom": 1.0,
        "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom": 1.0,
    }
    values.update(overrides)
    return values


def test_review_band_is_excluded_by_strong_fold_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "review_band"},
        local_structure_review=_passed_local_structure_review(),
    )

    assert status == "ineligible"
    assert reasons == ["fold_review_class_not_strong"]


def test_good_fold_is_excluded_by_strong_fold_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "good_fold_preserved"},
        local_structure_review=_passed_local_structure_review(),
    )

    assert status == "ineligible"
    assert reasons == ["fold_review_class_not_strong"]


def test_strong_fold_without_other_blockers_is_eligible() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        local_structure_review=_passed_local_structure_review(),
    )

    assert status == "eligible"
    assert reasons == []


def test_review_label_does_not_override_hard_blockers() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 1},
        fold={"foldcheck_status": "accepted", "review_class": "review_band"},
        local_structure_review=_passed_local_structure_review(),
    )

    assert status == "ineligible"
    assert reasons == ["fold_review_class_not_strong", "protected_mutation_violation"]


def test_catalytic_or_direct_contact_mutation_fails_hard_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        review_axes={"catalytic_or_direct_contact_mutation_count": 1},
        local_structure_review=_passed_local_structure_review(),
    )

    assert status == "ineligible"
    assert "catalytic_or_direct_contact_mutation" in reasons


def test_thumb_contact_track_mutation_fails_ordinary_panel_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        review_axes={"thumb_contact_track_mutation_count": 1},
        local_structure_review=_passed_local_structure_review(),
    )

    assert status == "ineligible"
    assert "thumb_contact_track_mutation" in reasons


def test_near_region_directional_chemistry_does_not_fail_preservation_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        review_axes={
            "nucleic_acid_facing_charge_delta": -1,
            "nucleic_acid_facing_acidic_gain_count": 2,
            "nucleic_acid_facing_basic_gain_count": 0,
        },
        local_structure_review=_passed_local_structure_review(),
    )

    assert status == "eligible"
    assert "nucleic_acid_facing_chemistry_incompatible" not in reasons


def test_missing_local_structure_review_fails_hard_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        local_structure_review=None,
    )

    assert status == "missing_inputs"
    assert reasons == ["missing_local_structure_review"]


def test_unavailable_local_structure_review_fails_hard_gate() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
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
        local_structure_review={
            "local_structure_gate_status": "threshold_exceeded",
            "local_structure_gate_failure_reasons_json": '["thumb_contact_track_context:local_ca_rmsd 2.70 > 2.50"]',
        },
    )

    assert status == "ineligible"
    assert reasons == ["local_structure_threshold_exceeded"]


def test_hard_gate_uses_declared_local_structure_status_without_extra_rmsd_overlay() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        local_structure_review=_passed_local_structure_review(
            local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom=3.2
        ),
    )

    assert status == "eligible"
    assert reasons == []


def test_hard_gate_does_not_require_region_rmsd_fields_after_declared_pass() -> None:
    status, reasons = _hard_gate_status(
        candidate={"status": "accepted", "protected_mutation_count": 0},
        fold={"foldcheck_status": "accepted", "review_class": "strong_fold_preserved"},
        local_structure_review={"local_structure_gate_status": "passed"},
    )

    assert status == "eligible"
    assert reasons == []
