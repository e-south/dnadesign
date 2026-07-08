"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_panel_contract_fixtures.py

Panel-contract fixtures for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    PRIMARY_PANEL_SIZE,
)

PRIMARY_CLASS = "eco1_rt_clade9_plurality25_contact10a_v1"
BOUNDARY_CLASS = "eco1_rt_clade9_plurality25_contact5a_v1"


def panel_rows(classes: list[str]) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "design_class_id": design_class_id,
            "selection_candidate_tier": "primary_panel_candidate",
        }
        for index, design_class_id in enumerate(classes, start=1)
    ]


def candidate_row(
    candidate_id: str,
    *,
    design_class_id: str = PRIMARY_CLASS,
    tier: str = "primary_panel_candidate",
    na_facing_mutation_count: int,
    proximal_unobserved_mutation_count: int = 0,
    proximal_rare_or_unobserved_mutation_count: int = 0,
    acidic_gain_count: int = 0,
    basic_loss_count: int = 0,
    proline_glycine_gain_count: int = 0,
    msa_fraction: float = 1.0,
    msa_frequency: float = 0.5,
    chemistry_warning_count: int = 0,
    mutation_count_total: int = 100,
    c_terminal_rmsd: float = 1.0,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "sequence_hash": f"sha256:{candidate_id:0<64}"[:71],
        "design_class_id": design_class_id,
        "selection_candidate_tier": tier,
        "primary_panel_candidate": tier == "primary_panel_candidate",
        "primary_panel_failure_reasons_json": "[]"
        if tier == "primary_panel_candidate"
        else '["near_retained_dna_rna_acidic_gain"]',
        "fold_review_class": "strong_fold_preserved",
        "hard_gate_status": "eligible",
        "feasibility_status": "feasible",
        "proximal_review_unobserved_mutation_count": proximal_unobserved_mutation_count,
        "proximal_review_rare_or_unobserved_mutation_count": proximal_rare_or_unobserved_mutation_count,
        "selection_support_profile_id": "ec86_clade9_conservation_v1",
        "selection_support_alt_observed_fraction": msa_fraction,
        "selection_support_alt_frequency_mean": msa_frequency,
        "selection_support_unobserved_mutation_count": 0,
        "nucleic_acid_facing_mutation_count": na_facing_mutation_count,
        "nucleic_acid_facing_charge_delta": 1,
        "nucleic_acid_facing_acidic_gain_count": acidic_gain_count,
        "nucleic_acid_facing_basic_gain_count": 0,
        "nucleic_acid_facing_basic_loss_count": basic_loss_count,
        "nucleic_acid_facing_proline_glycine_gain_count": proline_glycine_gain_count,
        "nucleic_acid_facing_chemistry_warning_count": chemistry_warning_count,
        "nucleic_acid_facing_chemistry_compatible": True,
        "nucleic_acid_facing_chemistry_gate_status": "passed",
        "near_retained_dna_rna_acidic_gain_review_status": "passed",
        "proximal_msa_support_review_status": "passed",
        "catalytic_or_direct_contact_mutation_count": 0,
        "thumb_contact_track_mutation_count": 0,
        "c_terminal_primer_rna_recognition_mutation_count": 0,
        "distal_scaffold_mutation_count": 3,
        "local_structure_gate_status": "passed",
        "local_structure_threshold_policy_id": "fixture_threshold_policy",
        "local_structure_threshold_failed_region_count": 0,
        "local_structure_max_ca_rmsd_angstrom": c_terminal_rmsd,
        "local_structure_catalytic_initiation_context_ca_rmsd_angstrom": 1.0,
        "local_structure_thumb_contact_track_context_ca_rmsd_angstrom": 1.0,
        "local_structure_c_terminal_primer_rna_recognition_context_ca_rmsd_angstrom": c_terminal_rmsd,
        "local_structure_near_retained_dna_rna_annulus_ca_rmsd_angstrom": 1.0,
        "mean_plddt": 90.0,
        "wt_runtime_ca_rmsd": 1.0,
        "cryoem_mapped_ca_rmsd": 2.0,
        "mutation_count_total": mutation_count_total,
        "sae_window_status": "wt_like_not_used_for_selection",
    }


def primary_candidate_rows(count: int = PRIMARY_PANEL_SIZE) -> list[dict[str, object]]:
    return [
        candidate_row(
            f"primary_{index}",
            design_class_id=PRIMARY_CLASS,
            na_facing_mutation_count=10 + index,
            chemistry_warning_count=1,
            mutation_count_total=30 + index,
        )
        for index in range(count)
    ]
