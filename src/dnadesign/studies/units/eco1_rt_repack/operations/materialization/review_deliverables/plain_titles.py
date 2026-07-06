"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/plain_titles.py

Plain-language notebook titles for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_PLAIN_TITLES,
)

PLAIN_DELIVERABLE_TITLES = {
    "msa_plurality_mask_panel": "Clade 9 conservation defines the baseline mask",
    "msa_subtype_plurality_panel": "The Eco1 subtype MSA gives a closer conservation view",
    "design_class_mask_overview": "Fixed residues combine motif, conservation, and substrate rules",
    "mask_structure_context_script": "ChimeraX can reproduce the active mask view",
    "mask_structure_context_orientation_template": "The ChimeraX template preserves mask colors",
    "mask_structure_context_png": "The EC86 structure shows active fixed residues",
    "mask_structure_browser_manifest": "The EC86 structure maps each fixed-mask rule",
    "proteinmpnn_score_mutation_burden": "ProteinMPNN scores describe proposal spread",
    "proteinmpnn_residue_frequency_heatmap": "ProteinMPNN samples amino acids within each fixed mask",
    "expanded_proteinmpnn_fold_validation": "Expanded designs preserve the RT fold",
    "foldcheck_review_review_class_counts": "Each fixed mask keeps foldable candidates",
    "foldcheck_review_biohub_esmc_sae_coverage": "Biohub ESMC returns SAE coverage for accepted rows",
    "foldcheck_review_structure_overlay_panel": "ColabFold structures align to the EC86 reference",
    "biohub_esmc_sequence_scoring_manifest": "ESMC scoring records WT-context inputs",
    "biohub_esmc_variant_llr_scores": "ESMC LLR scores are recorded for each candidate",
    "biohub_esmc_candidate_preference_vs_wt": "Candidate substitutions shift ESMC LLR",
    "biohub_esmc_6b_sequence_scoring_manifest": "6B ESMC scoring records WT-context inputs",
    "biohub_esmc_6b_variant_llr_scores": "6B ESMC LLR scores are recorded for each candidate",
    "biohub_esmc_6b_candidate_preference_vs_wt": "6B ESMC LLR scores provide review-only context",
    "biohub_esmc_candidate_preference_model_agreement_table": "300M and 6B ESMC score-order changes are recorded",
    "biohub_esmc_candidate_preference_model_agreement": "300M and 6B ESMC scores differ by candidate",
    "interactive_structure_browser_manifest": "Folded candidates can be inspected one at a time",
    "selected_panel_structure_browser_manifest": "Selected structures can be inspected one at a time",
    "wt_esmc_entropy_by_position": "ESMC entropy varies by WT residue",
    "wt_esmc_fraction_negative_alternate_llr": "ESMC disfavors different alternates by residue",
    "wt_esmc_substitution_llr_heatmap": "ESMC scores WT-context substitutions",
    "msa_plurality_vs_esmc_entropy": "Clade 9 plurality tracks lower ESMC entropy",
    "msa_plurality_vs_best_alt_llr": "Highest alternate LLR marks MSA-model disagreement",
    "msa_esmc_constraint_tracks": "MSA and ESMC signals align with mask classes",
    "biohub_esmc_protein_top_sae_features": "Strongest SAE features are ordered by activation",
    "biohub_esmc_wt_top_sae_feature_activation_pattern": "WT-active SAE features localize by residue",
    "biohub_esmc_sae_feature_activation_heatmap": "Selected SAE features activate across RT variants",
    "biohub_esmc_sae_structure_browser_manifest": "SAE activation regions can be inspected on structure",
    "selection_funnel_summary": "Panel selection keeps activity claims separate",
    "selection_panel_table": "Six Eco1 RT variants form a protein review panel",
    "selection_handoff_sequences": "Selected protein sequences keep handoff scope explicit",
    **SELECTION_PLOT_PLAIN_TITLES,
}


def apply_plain_titles(deliverables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deliverable rows with notebook-facing plain-language titles."""

    rows: list[dict[str, Any]] = []
    for row in deliverables:
        normalized = dict(row)
        title = PLAIN_DELIVERABLE_TITLES.get(str(row.get("deliverable_id") or ""))
        if title:
            normalized["title"] = title
        rows.append(normalized)
    return rows
