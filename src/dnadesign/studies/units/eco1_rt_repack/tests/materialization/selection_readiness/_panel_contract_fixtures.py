"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/_panel_contract_fixtures.py

Panel-contract fixtures for Eco1 RT selection-readiness tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)

PRIMARY_POLICY = COMBINED_NEAR_PLUS_DISTAL_POLICY_ID
COMPARISON_POLICY = NEAR_DNA_RNA_ACID_FREE_POLICY_ID


def panel_rows(policy_ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "candidate_id": f"candidate_{index}",
            "policy_id": policy_id,
            "selection_contract_pass": True,
            "nucleic_acid_facing_mutation_count": 1,
        }
        for index, policy_id in enumerate(policy_ids, start=1)
    ]


def candidate_row(
    candidate_id: str,
    *,
    policy_id: str = PRIMARY_POLICY,
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
    wang_alpha1_mutation_count: int = 0,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "sequence_hash": f"sha256:{candidate_id:0<64}"[:71],
        "policy_id": policy_id,
        "primary_policy_id": policy_id,
        "source_policy_ids": [policy_id],
        "selection_contract_pass": True,
        "selection_contract_failure_reasons_json": "[]",
        "wang_alpha1_r13_mutation_count": 0,
        "wang_alpha1_r13_review_status": "retained_wt",
        "wang_alpha1_mutation_count": wang_alpha1_mutation_count,
        "fold_review_class": "strong_fold_preserved",
        "hard_gate_status": "eligible",
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
        "near_retained_dna_rna_acidic_gain_review_status": "passed",
        "proximal_msa_support_review_status": "passed",
        "catalytic_or_direct_contact_mutation_count": 0,
        "thumb_contact_track_mutation_count": 0,
        "c_terminal_primer_rna_recognition_mutation_count": 0,
        "distal_scaffold_mutation_count": 3,
        "local_structure_gate_status": "passed",
        "local_structure_threshold_policy_id": "fixture_threshold_policy",
        "local_structure_threshold_failed_region_count": 0,
        "local_structure_max_gated_ca_rmsd_angstrom": c_terminal_rmsd,
        "local_structure_max_all_region_ca_rmsd_angstrom": c_terminal_rmsd,
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


def comparison_candidates() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return three contract-pass candidates for each comparison policy."""

    policy_specs = (
        (DISTAL_SCAFFOLD_POLICY_ID, "distal", 0),
        (NEAR_DNA_RNA_ACID_FREE_POLICY_ID, "near", 8),
        (COMBINED_NEAR_PLUS_DISTAL_POLICY_ID, "combined", 8),
    )
    triage_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    mutation_position = 10
    for policy_id, prefix, peripheral_count in policy_specs:
        for index in range(1, 4):
            candidate_id = f"{prefix}_{index}"
            triage_rows.append(
                candidate_row(
                    candidate_id,
                    policy_id=policy_id,
                    na_facing_mutation_count=peripheral_count,
                    mutation_count_total=20 + index,
                )
            )
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "policy_id": policy_id,
                    "sequence": "A" * (63 - index) + "C" * index,
                    "canonical_mutations": [
                        f"A{mutation_position}G",
                        f"L{mutation_position + 1}V",
                    ],
                }
            )
            mutation_position += 2
    return triage_rows, candidate_rows


def comparison_input_hashes() -> dict[str, str | None]:
    """Return deterministic input hashes for selected-panel tests."""

    return {
        "candidate_triage_table": "sha256:triage",
        "foldcheck_review": "sha256:fold",
        "sae_window_summary": None,
    }
