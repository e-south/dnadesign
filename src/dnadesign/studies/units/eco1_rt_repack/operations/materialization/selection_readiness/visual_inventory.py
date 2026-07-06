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


CURRENT_SELECTION_PLOTS = (
    SelectionPlot(
        plot_id="selection_design_class_gate_counts",
        file_name="selection_design_class_gate_counts.svg",
        plain_title="Each mask class contributes fold-preserved candidates",
    ),
    SelectionPlot(
        plot_id="selection_population_stratification",
        file_name="selection_population_stratification.svg",
        plain_title="Selected candidates sit within the full design pool",
    ),
    SelectionPlot(
        plot_id="selection_class_local_percentiles",
        file_name="selection_class_local_percentiles.svg",
        plain_title="Each selected row is reviewed within its own mask class",
    ),
    SelectionPlot(
        plot_id="selection_six_sequence_distance",
        file_name="selection_six_sequence_distance.svg",
        plain_title="The selected six sample distinct sequence neighborhoods",
    ),
    SelectionPlot(
        plot_id="selection_selected_substitutions_across_rt",
        file_name="selection_selected_substitutions_across_rt.svg",
        plain_title="Selected substitutions map to RT regions",
    ),
    SelectionPlot(
        plot_id="selection_regional_mutation_burden",
        file_name="selection_regional_mutation_burden.svg",
        plain_title="Selected candidates differ in which RT regions carry mutations",
    ),
    SelectionPlot(
        plot_id="selection_na_facing_chemistry_balance",
        file_name="selection_na_facing_chemistry_balance.svg",
        plain_title="Chemistry changes near DNA/RNA or thumb-track are review context",
    ),
)

CURRENT_SELECTION_PLOT_IDS = tuple(plot.plot_id for plot in CURRENT_SELECTION_PLOTS)
CURRENT_SELECTION_PLOT_FILE_NAMES = tuple(plot.file_name for plot in CURRENT_SELECTION_PLOTS)
SELECTION_PLOT_PLAIN_TITLES = {plot.plot_id: plot.plain_title for plot in CURRENT_SELECTION_PLOTS}

RETIRED_SELECTION_PLOT_IDS = (
    "selection_panel_review_axes",
    "selection_panel_sequence_differences",
    "selection_panel_mutation_geography_chemistry",
)
RETIRED_SELECTION_PLOT_FILE_NAMES = tuple(f"{plot_id}.svg" for plot_id in RETIRED_SELECTION_PLOT_IDS)
