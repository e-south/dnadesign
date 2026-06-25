"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/constants.py

Study-local constants for Eco1 RT repack contract validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

_STUDY_ID = "eco1_rt_repack"
_DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
_CONTRACT_ROOT = _DOCS_ROOT / "operations/contract"
_PLANNED_THREAD_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
_ALLOWED_PHASES = (
    "phase0_scaffold",
    "phase1_thread_contract",
    "phase2_real_backend_ingest",
    "phase3_downstream_promotion",
)
_PHASE_RANK = {phase: index for index, phase in enumerate(_ALLOWED_PHASES)}
_CONTRACT_STATES = {"scaffold", "fixture", "materialized", "accepted", "rejected"}
_PENDING_VALUES = {"pending", "pending_review", "pending_source_selection", "not started"}
_EXPECTED_ARTIFACT_ORDER = (
    "backbone_bundle",
    "residue_map",
    "conservation_profile",
    "contact_profile",
    "mask_set",
    "thread_plan",
    "proteinmpnn_request",
    "sample_table",
    "candidate_table",
    "foldcheck_report",
    "feasibility_report",
    "candidate_handoff",
)
_SHARED_ARTIFACT_FIELDS = {
    "schema_id",
    "schema_version",
    "artifact_id",
    "status",
    "created_by",
    "created_at",
    "upstream_artifact_hashes",
}
_REQUIRED_ARTIFACT_INVARIANTS = {
    "fallback_policy_must_be_explicit_no_fallback_for_sampling": (
        "thread.artifact_chain.missing_no_fallback_invariant"
    ),
    "fixture_artifacts_cannot_satisfy_materialized_handoff": (
        "thread.artifact_chain.missing_fixture_boundary_invariant"
    ),
    "no_candidate_handoff_without_foldcheck_and_feasibility_reports": (
        "thread.artifact_chain.missing_handoff_gate_invariant"
    ),
}
_REQUIRED_MASK_CASES = {
    "reject_pending_structure_authority_phase1",
    "reject_missing_contact_threshold",
    "reject_missing_conservation_profile_hash",
    "reject_fixture_foldcheck_for_materialized_handoff",
    "reject_candidate_handoff_without_hash_closure",
    "reject_downstream_construct_subject_preclaim",
    "accept_mapped_nonprotected_mutable_residue",
    "accept_effector_interface_unfixed_when_not_retained",
}
_REQUIRED_NUMBERING_POLICY_FIELDS = {
    "policy_id",
    "selected_structure_source_id",
    "reference_sequence_hash",
    "residue_numbering_origin",
    "canonical_position_basis",
    "structure_position_basis",
    "design_position_basis",
    "source_map_ref",
    "source_map_sha256",
    "coverage",
    "required_mapping_columns",
    "residue_map_artifact",
}
_REQUIRED_RESIDUE_MAP_COLUMNS = {
    "canonical_position",
    "wt_aa",
    "structure_chain_id",
    "structure_residue_id",
    "pdb_insertion_code",
    "cds_codon_index",
    "design_position",
    "mapping_status",
    "mapping_issue",
}
_REQUIRED_CONTACT_PROFILE_COLUMNS = {
    "canonical_position",
    "retained_context_id",
    "nearest_context_atom_distance_angstrom",
    "contact_threshold_angstrom",
    "passes_contact_mask",
    "source_hash",
    "wt_aa",
    "structure_chain_id",
    "structure_residue_id",
    "mapping_status",
}
_REQUIRED_CONTACT_GEOMETRY_PROFILE_COLUMNS = {
    "canonical_position",
    "wt_aa",
    "structure_chain_id",
    "structure_residue_id",
    "mapping_status",
    "nearest_context_atom_distance_angstrom",
    "nearest_sidechain_context_distance_angstrom",
    "nearest_backbone_context_distance_angstrom",
    "nearest_dna_distance_angstrom",
    "nearest_rna_distance_angstrom",
    "nearest_context_chain_id",
    "nearest_context_molecule_type",
    "nearest_context_atom_name",
    "sidechain_atom_status",
    "contact_atom_count_within_4a",
    "contact_atom_count_within_6a",
    "contact_atom_count_within_8a",
    "contact_atom_count_within_10a",
    "contact_atom_count_within_12a",
    "contact_atom_count_within_15a",
    "contact_atom_count_within_20a",
    "retained_context_chain_count_within_8a",
    "retained_context_chain_count_within_12a",
    "retained_context_chain_count_within_15a",
    "retained_context_chain_count_within_20a",
}
_REQUIRED_CONSERVATION_PROFILE_COLUMNS = {
    "canonical_position",
    "profile_id",
    "wt_aa",
    "msa_column",
    "non_gap_count",
    "wt_count",
    "wt_frequency",
    "plurality_aa",
    "wt_is_plurality",
    "conservation_threshold",
    "min_non_gap_count",
    "passes_conservation_mask",
    "source_hash",
    "target_sequence_hash",
    "mapping_status",
}
_REQUIRED_MASK_SET_COLUMNS = {
    "canonical_position",
    "wt_aa",
    "design_position",
    "mapping_status",
    "has_backbone_coordinates",
    "min_distance_to_retained_dna_rna_angstrom",
    "direct_contact_threshold_angstrom",
    "direct_retained_dna_rna_contact_5a",
    "motif_protected",
    "wang_ec86_direct_contact_prior",
    "evolutionarily_conserved_clade9_25pct_plurality",
    "wt_plurality_frequency",
    "wt_plurality_aa",
    "conservation_profile_ids",
    "manual_mask_reason",
    "wang_ec86_direct_contact_reason",
    "rt_interval_review_label",
    "protected",
    "non_fixed",
    "non_fixed_missing_backbone",
    "protection_reasons",
    "conflict_status",
    "conflict_reason",
}
_REQUIRED_CONSERVATION_PROFILE_IDS = {"ec86_clade9_conservation_v1", "ec86_iia3_cluster42_1_conservation_v1"}
_REQUIRED_CONSERVATION_PROVIDER_IDS = {"ncbi_protein_efetch", "bv_brc_feature_protein_fasta"}
_CONSERVATION_GAP_DENOMINATOR_POLICY = "non_gap_count"
_CONSERVATION_PLURALITY_RULE = "wt_aa_must_equal_plurality_aa"
_CONSERVATION_TARGET_POLICY = "ec86kit_reference_sequence_must_be_target_row"
_TARGET_MISMATCH_POLICY = "reject_as_target_without_declared_substitution"
_PROVIDER_FAILURE_POLICY = "explicit_exclude_or_fail"
_FORBIDDEN_CONSERVATION_DENOMINATOR_RULES = {"mestre_s1_all_retron_rt_records_context"}
