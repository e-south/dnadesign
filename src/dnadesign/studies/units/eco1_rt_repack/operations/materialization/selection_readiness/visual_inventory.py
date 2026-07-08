"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/visual_inventory.py

Selection-readiness visual inventory for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionPlot:
    """Notebook-facing selection plot identity and plain title."""

    plot_id: str
    file_name: str
    plain_title: str
    selection_role: str
    funnel_stage_id: str = ""
    notebook_group: str = "context_checks"
    not_a_selector_reason: str = ""


CURRENT_SELECTION_PLOTS = (
    SelectionPlot(
        plot_id="selection_design_class_contrast",
        file_name="selection_design_class_contrast.svg",
        plain_title="Design-class mask policies",
        selection_role="input_stratification",
        funnel_stage_id="candidate_pool",
        notebook_group="core_funnel",
        not_a_selector_reason=("Explains input mask policies; design class is context and is not a final-panel quota."),
    ),
    SelectionPlot(
        plot_id="selection_primary_panel_sankey",
        file_name="selection_primary_panel_sankey.svg",
        plain_title="Primary panel funnel",
        selection_role="hard_gate_and_global_rank",
        funnel_stage_id="global_conservative_diverse_selection",
        notebook_group="core_funnel",
    ),
    SelectionPlot(
        plot_id="selection_local_structure_stratification",
        file_name="selection_local_structure_stratification.svg",
        plain_title="Local RMSD gates",
        selection_role="hard_gate",
        funnel_stage_id="local_structure_gate",
        notebook_group="core_funnel",
    ),
    SelectionPlot(
        plot_id="selection_local_structure_threshold_sensitivity",
        file_name="selection_local_structure_threshold_sensitivity.svg",
        plain_title="Local RMSD threshold sensitivity",
        selection_role="gate_audit",
        funnel_stage_id="local_structure_gate",
        notebook_group="context_checks",
        not_a_selector_reason=(
            "Audits threshold sensitivity; only the declared local RMSD threshold gate filters candidates."
        ),
    ),
    SelectionPlot(
        plot_id="selection_local_structure_by_region",
        file_name="selection_local_structure_by_region.svg",
        plain_title="Selected local RMSD by region",
        selection_role="selected_panel_audit",
        funnel_stage_id="local_structure_gate",
        notebook_group="core_funnel",
        not_a_selector_reason=(
            "Audits selected rows after the local-structure gate; the declared threshold table is the filter."
        ),
    ),
    SelectionPlot(
        plot_id="selection_regional_mutation_burden",
        file_name="selection_regional_mutation_burden.svg",
        plain_title="Selected mutation burden by region",
        selection_role="tie_break_context",
        funnel_stage_id="global_conservative_diverse_selection",
        notebook_group="selection_rationale",
        not_a_selector_reason=(
            "Contributes to post-gate selection review only; it is not activity evidence or an independent gate."
        ),
    ),
    SelectionPlot(
        plot_id="selection_na_facing_chemistry_balance",
        file_name="selection_na_facing_chemistry_balance.svg",
        plain_title="Chemistry changes near retained DNA/RNA",
        selection_role="gate_audit_and_tie_break_context",
        funnel_stage_id="chemistry_support_gate",
        notebook_group="selection_rationale",
        not_a_selector_reason=(
            "Audits the near-region chemistry gate and post-gate chemistry fields; it is not activity evidence."
        ),
    ),
    SelectionPlot(
        plot_id="selection_regionwise_msa_support",
        file_name="selection_regionwise_msa_support.svg",
        plain_title="MSA support is reviewed by mutation region",
        selection_role="gate_audit_and_tie_break_context",
        funnel_stage_id="chemistry_support_gate",
        notebook_group="selection_rationale",
        not_a_selector_reason=(
            "Audits the proximal support gate and regional support fields; it is not activity evidence."
        ),
    ),
    SelectionPlot(
        plot_id="selection_selected_substitutions_across_rt",
        file_name="selection_selected_substitutions_across_rt.svg",
        plain_title="Selected substitutions map to RT regions",
        selection_role="selected_panel_audit",
        funnel_stage_id="global_conservative_diverse_selection",
        notebook_group="selection_rationale",
        not_a_selector_reason=("Audits selected-row mutation locations; it does not add a hidden selection rule."),
    ),
    SelectionPlot(
        plot_id="selection_design_class_gate_counts",
        file_name="selection_design_class_gate_counts.svg",
        plain_title="Each design class retains protein-gate candidates",
        selection_role="context_check",
        funnel_stage_id="preservation_gate",
        notebook_group="context_checks",
        not_a_selector_reason=(
            "Context plot for per-class gate counts; the manifest funnel table is the selector record."
        ),
    ),
    SelectionPlot(
        plot_id="selection_premise_alignment",
        file_name="selection_premise_alignment.svg",
        plain_title="Selected candidates show core/contact, local, and regional checks",
        selection_role="selected_panel_audit",
        funnel_stage_id="global_conservative_diverse_selection",
        notebook_group="context_checks",
        not_a_selector_reason="Summary checklist for selected rows; it does not filter candidates.",
    ),
    SelectionPlot(
        plot_id="selection_six_sequence_distance",
        file_name="selection_six_sequence_distance.svg",
        plain_title="Selected mutation-set dissimilarity",
        selection_role="global_rank_audit",
        funnel_stage_id="global_conservative_diverse_selection",
        notebook_group="selection_rationale",
        not_a_selector_reason=(
            "Audits the mutation-set dissimilarity used during global panel selection; it is not functional evidence."
        ),
    ),
)

CURRENT_SELECTION_PLOT_IDS = tuple(plot.plot_id for plot in CURRENT_SELECTION_PLOTS)
CURRENT_SELECTION_PLOT_FILE_NAMES = tuple(plot.file_name for plot in CURRENT_SELECTION_PLOTS)
SELECTION_PLOT_PLAIN_TITLES = {plot.plot_id: plot.plain_title for plot in CURRENT_SELECTION_PLOTS}
SELECTION_PLOT_METADATA = {
    plot.plot_id: {
        "selection_role": plot.selection_role,
        "funnel_stage_id": plot.funnel_stage_id,
        "notebook_group": plot.notebook_group,
        "notebook_group_label": {
            "core_funnel": "Funnel",
            "selection_rationale": "Selection review",
            "context_checks": "Context",
        }[plot.notebook_group],
        "not_a_selector_reason": plot.not_a_selector_reason,
    }
    for plot in CURRENT_SELECTION_PLOTS
}

RETIRED_SELECTION_PLOT_IDS = (
    "selection_panel_review_axes",
    "selection_panel_sequence_differences",
    "selection_panel_mutation_geography_chemistry",
    "selection_population_stratification",
    "selection_class_local_elimination_trace",
    "selection_class_local_percentiles",
)
RETIRED_SELECTION_PLOT_FILE_NAMES = tuple(f"{plot_id}.svg" for plot_id in RETIRED_SELECTION_PLOT_IDS)
