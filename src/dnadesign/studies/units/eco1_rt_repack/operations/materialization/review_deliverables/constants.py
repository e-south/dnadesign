"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/constants.py

Constants for Eco1 review-deliverable materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
DELIVERABLE_DIR_NAME = "review_deliverables"
MANIFEST_FILE_NAME = "review_deliverable_manifest.yaml"
NOTEBOOK_FILE_NAME = "eco1_review_deliverables.py"
NOTEBOOKS_DIR_NAME = "notebooks"
MSA_PANEL_DIR_NAME = "msa_plurality_mask_panel"
MASK_CONTEXT_DIR_NAME = "mask_structure_context"
PROTEINMPNN_DIR_NAME = "proteinmpnn_candidate_diversity"
FOLDCHECK_REVIEW_DIR_NAME = "foldcheck_review"
WT_MODEL_CONSTRAINT_DIR_NAME = "wt_model_constraint_audit"
BIOHUB_ESMC_SAE_INTERPRETATION_DIR_NAME = "biohub_esmc_sae_interpretation"
STRUCTURE_BROWSER_DIR_NAME = "structure_browser"
CONSERVATION_CLADE9_PROFILE_ID = "ec86_clade9_conservation_v1"
CONSERVATION_SUBTYPE_PROFILE_ID = "ec86_iia3_cluster42_1_conservation_v1"
CONSERVATION_PROFILE_ID = CONSERVATION_CLADE9_PROFILE_ID
SCHEMA_ID = "eco1_rt.review_deliverables"
REFERENCE_BACKBONE_RELATIVE_PATH = "proteinmpnn_request/chain_a_backbone.pdb"
CONSERVATION_PROFILE_FILE_NAME = "conservation_profile.parquet"
MASK_SET_FILE_NAME = "mask_set.yaml"
CANDIDATE_TABLE_FILE_NAME = "candidate_table.parquet"
FOLDCHECK_REPORT_FILE_NAME = "foldcheck_report.parquet"
FOLDCHECK_REVIEW_MANIFEST_RELATIVE_PATH = "foldcheck_review/review_visual_manifest.yaml"
FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH = "foldcheck_review/foldcheck_candidate_ranking.parquet"
FOLDCHECK_FULL_STRUCTURE_SET_RELATIVE_PATH = "foldcheck_review/foldcheck_full_structure_set.yaml"
BIOHUB_ESMC_MUTATION_SCORING_RELATIVE_PATH = "biohub_esmc/mutation_scoring"
BIOHUB_ESMC_WT_SUBSTITUTION_LLR_RELATIVE_PATH = "biohub_esmc/mutation_scoring/wt_substitution_llr.parquet"
BIOHUB_ESMC_SAE_PROFILE_FILE_NAME = "biohub_esmc_sae_profile.parquet"
BIOHUB_ESMC_PROTEIN_FEATURES_FILE_NAME = "biohub_esmc_protein_features.parquet"
BIOHUB_ESMC_RESIDUE_FEATURES_FILE_NAME = "biohub_esmc_residue_features.parquet"
BIOHUB_ESMC_FEATURE_CATALOG_FILE_NAME = "biohub_esmc_feature_catalog.parquet"
BIOHUB_ESMC_REQUEST_MANIFEST_FILE_NAME = "biohub_esmc_request_manifest.yaml"
ALIGNED_FASTA_RELATIVE_PATH = f"conservation_alignments/{CONSERVATION_CLADE9_PROFILE_ID}.aligned.fasta"
SUBTYPE_ALIGNED_FASTA_RELATIVE_PATH = f"conservation_alignments/{CONSERVATION_SUBTYPE_PROFILE_ID}.aligned.fasta"
CONSERVATION_SOURCE_MANIFEST_RELATIVE_PATH = (
    f"conservation_sources/{CONSERVATION_CLADE9_PROFILE_ID}.source_manifest.yaml"
)
SUBTYPE_CONSERVATION_SOURCE_MANIFEST_RELATIVE_PATH = (
    f"conservation_sources/{CONSERVATION_SUBTYPE_PROFILE_ID}.source_manifest.yaml"
)

SECTION_CONSTRAINT_EVIDENCE = "constraint_evidence_for_design_mask"
SECTION_DESIGNS_AND_FOLD_TRIAGE = "proteinmpnn_designs_and_fold_triage"
SECTION_ESMC_FEATURE_REVIEW = "esmc_feature_review"
SECTION_FEASIBILITY_AND_HANDOFF = "feasibility_and_handoff"
