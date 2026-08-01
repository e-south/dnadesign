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
    role: str = "manuscript_facing"


CURRENT_SELECTION_PLOTS = (
    SelectionPlot(
        plot_id="selection_hypothesis_panel_flow",
        file_name="selection_hypothesis_panel_flow.svg",
        plain_title="Sequence generation and structural review produce the selected panel",
        selection_role="funnel_summary",
        notebook_group="core_funnel",
        not_a_selector_reason="Summarizes the declared filters and ranking; the plot does not select rows.",
    ),
    SelectionPlot(
        plot_id="selection_local_structure_stratification",
        file_name="selection_local_structure_stratification.svg",
        plain_title="Local geometry screen",
        selection_role="hard_gate",
        funnel_stage_id="local_geometry_screen",
        notebook_group="core_funnel",
    ),
    SelectionPlot(
        plot_id="selection_local_structure_by_region",
        file_name="selection_local_structure_by_region.svg",
        plain_title="Selected local RMSD by region",
        selection_role="selected_panel_audit",
        funnel_stage_id="local_geometry_screen",
        notebook_group="core_funnel",
        not_a_selector_reason=(
            "Audits selected rows after the local-structure gate; the declared threshold table is the filter."
        ),
    ),
    SelectionPlot(
        plot_id="selection_selected_substitutions_across_rt",
        file_name="selection_selected_substitutions_across_rt.svg",
        plain_title="Selected substitutions map to RT regions",
        selection_role="selected_panel_audit",
        funnel_stage_id="selected_panel",
        notebook_group="selection_rationale",
        not_a_selector_reason=("Audits selected-row mutation locations; it does not add a hidden selection rule."),
    ),
    SelectionPlot(
        plot_id="selection_distal_pair_n_terminal_comparison",
        file_name="selection_distal_pair_n_terminal_comparison.svg",
        plain_title="WT, D01, and D02 N-terminal sequence and charge-proxy comparison",
        selection_role="selected_distal_pair_sequence_context",
        funnel_stage_id="selected_panel",
        notebook_group="context_checks",
        not_a_selector_reason=(
            "Compares canonical selected sequences after panel selection; the charge proxy does not select rows or "
            "identify causal residues."
        ),
        role="review_only",
    ),
    SelectionPlot(
        plot_id="selection_regional_mutation_burden",
        file_name="selection_regional_mutation_burden.svg",
        plain_title="Selected mutation burden by region",
        selection_role="tie_break_context",
        funnel_stage_id="selected_panel",
        notebook_group="selection_rationale",
        not_a_selector_reason=(
            "Contributes to post-gate selection review only; it is not activity evidence or an independent gate."
        ),
    ),
    SelectionPlot(
        plot_id="selection_na_facing_chemistry_balance",
        file_name="selection_na_facing_chemistry_balance.svg",
        plain_title="Selected charge changes near retained DNA/RNA",
        selection_role="generation_contract_audit_and_tie_break_context",
        funnel_stage_id="",
        notebook_group="selection_rationale",
        not_a_selector_reason=(
            "Audits the near-region chemistry contract; it is not activity evidence or a separate displayed gate."
        ),
    ),
    SelectionPlot(
        plot_id="selection_regionwise_msa_support",
        file_name="selection_regionwise_msa_support.svg",
        plain_title="Clade-9 MSA support for selected substitutions",
        selection_role="gate_audit_and_tie_break_context",
        funnel_stage_id="",
        notebook_group="selection_rationale",
        not_a_selector_reason=(
            "Audits proximal and regional support fields; it is not activity evidence or a separate displayed gate."
        ),
    ),
    SelectionPlot(
        plot_id="selection_mutation_set_dissimilarity",
        file_name="selection_mutation_set_dissimilarity.svg",
        plain_title="Selected mutation-position distances",
        selection_role="within_policy_and_panel_distance_audit",
        funnel_stage_id="selected_panel",
        notebook_group="core_funnel",
        not_a_selector_reason=(
            "Audits within-group mutation-set dissimilarity; cross-group values are descriptive because the open "
            "position sets differ."
        ),
    ),
    SelectionPlot(
        plot_id="selection_pairwise_sequence_differences",
        file_name="selection_pairwise_sequence_differences.svg",
        plain_title="Pairwise amino-acid differences among selected RT sequences",
        selection_role="selected_panel_sequence_distance_audit",
        funnel_stage_id="selected_panel",
        notebook_group="core_funnel",
        not_a_selector_reason=(
            "Reports final amino-acid differences; selection used within-group mutation-position and exact-"
            "substitution Jaccard distances."
        ),
    ),
    SelectionPlot(
        plot_id="selection_local_structure_threshold_sensitivity",
        file_name="selection_local_structure_threshold_sensitivity.svg",
        plain_title="Local RMSD threshold sensitivity",
        selection_role="gate_audit",
        funnel_stage_id="local_geometry_screen",
        notebook_group="context_checks",
        not_a_selector_reason=(
            "Audits threshold sensitivity; only the declared local RMSD threshold gate filters candidates."
        ),
        role="review_only",
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
        "role": plot.role,
    }
    for plot in CURRENT_SELECTION_PLOTS
}
