"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/materialization_expected.py

Expected review-deliverable ids for Eco1 materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
)

EXPECTED_RENDERED_DELIVERABLE_IDS = {
    "msa_plurality_mask_panel",
    "msa_subtype_plurality_panel",
    "design_class_mask_overview",
    "proteinmpnn_score_mutation_burden",
    "proteinmpnn_residue_frequency_heatmap",
    "expanded_proteinmpnn_fold_validation",
    "foldcheck_review_review_class_counts",
    "mask_structure_context_script",
    "mask_structure_context_orientation_template",
    "mask_structure_browser_manifest",
    "biohub_esmc_sae_structure_browser_manifest",
    "msa_plurality_vs_esmc_entropy",
    "msa_plurality_vs_best_alt_llr",
    "msa_esmc_constraint_tracks",
    *CURRENT_SELECTION_PLOT_IDS,
    "selection_funnel_summary",
    "selection_panel_table",
    "selection_handoff_sequences",
    "selection_handoff_readiness",
    "selected_panel_structure_browser_manifest",
}

EXPECTED_LINKED_MODEL_CHECK_DELIVERABLE_IDS = {
    "wt_esmc_entropy_by_position",
    "wt_esmc_fraction_negative_alternate_llr",
    "wt_esmc_substitution_llr_heatmap",
}
