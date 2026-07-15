"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/constants.py

Constants for Eco1 fold-check review materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
REVIEW_DIR_NAME = "foldcheck_review"
RANKING_FILE_NAME = "foldcheck_candidate_ranking.parquet"
STRUCTURE_PANEL_FILE_NAME = "foldcheck_structure_panel.yaml"
FULL_STRUCTURE_SET_FILE_NAME = "foldcheck_full_structure_set.yaml"
ATLAS_SUBSET_FILE_NAME = "atlas_subset_manifest.yaml"
STRUCTURES_DIR_NAME = "structures"
FULL_STRUCTURE_SET_DIR_NAME = "full_fold_set"
CHIMERAX_DIR_NAME = "chimerax"
CHIMERAX_SCRIPT_NAME = "ec86_fold_panel.cxc"
FULL_CHIMERAX_SCRIPT_NAME = "ec86_full_fold_set.cxc"
PLOTS_DIR_NAME = "plots"
NOTEBOOKS_DIR_NAME = "notebooks"
VISUAL_MANIFEST_FILE_NAME = "review_visual_manifest.yaml"
REVIEW_NOTEBOOK_FILE_NAME = "eco1_foldcheck_review.py"
REFERENCE_BACKBONE_RELATIVE_PATH = "proteinmpnn_request/chain_a_backbone.pdb"
RESIDUE_MAP_FILE_NAME = "residue_map.parquet"
CANDIDATE_TABLE_FILE_NAME = "candidate_table.parquet"
FOLDCHECK_REPORT_FILE_NAME = "foldcheck_report.parquet"
BIOHUB_ESMC_PROFILE_FILE_NAME = "biohub_esmc_sae_profile.parquet"
FOLDCHECK_REQUEST_MANIFEST_RELATIVE_PATH = "foldcheck_request/foldcheck_request_manifest.yaml"
FOLDCHECK_RANKING_SCHEMA_ID = "eco1_rt.foldcheck_candidate_ranking"
STRUCTURE_PANEL_SCHEMA_ID = "eco1_rt.foldcheck_structure_panel"
FULL_STRUCTURE_SET_SCHEMA_ID = "eco1_rt.foldcheck_full_structure_set"
ATLAS_SUBSET_SCHEMA_ID = "eco1_rt.atlas_subset_manifest"
VISUAL_MANIFEST_SCHEMA_ID = "eco1_rt.foldcheck_review_visual_manifest"
WT_SEQUENCE_ID = "wild_type"
STRONG_FOLD_REVIEW_CLASS = "strong_fold_preserved"
STRONG_FOLD_MIN_MEAN_PLDDT = 91.5
STRONG_FOLD_MAX_WT_RUNTIME_CA_RMSD_ANGSTROM = 1.25
