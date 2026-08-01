"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/plots.py

Panel-selection SVG plots for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

from ..shared.rt_annotation_context import RTAnnotationContext
from .chemistry_balance import write_na_facing_chemistry_balance_plot
from .local_structure_plot import (
    write_local_structure_by_region_plot,
    write_local_structure_stratification_plot,
    write_local_structure_threshold_sensitivity_plot,
)
from .mutation_distance_plot import (
    write_selected_mutation_dissimilarity_plot,
)
from .n_terminal_pair_plot import write_n_terminal_pair_comparison_plot
from .region_msa_support_plot import write_regionwise_msa_support_plot
from .regional_plots import (
    write_regional_mutation_burden_plot,
    write_selected_substitutions_across_rt_plot,
)
from .sankey_plot import write_hypothesis_panel_flow_plot
from .sequence_distance_plot import write_selected_sequence_distance_plot

matplotlib.use("Agg")


def write_selection_readiness_plots(
    *,
    plot_root: Path,
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    canonical_sequences_by_id: dict[str, str],
    mask_residues: list[dict[str, object]],
    local_structure_rows: list[dict[str, object]],
    local_structure_threshold_sensitivity_rows: list[dict[str, object]],
    region_msa_support_rows: list[dict[str, object]],
    hypothesis_panel_selection_trace_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
    rt_annotation_context: RTAnnotationContext | None = None,
) -> list[dict[str, Any]]:
    """Write panel-selection plots and return manifest rows."""

    plot_root.mkdir(parents=True, exist_ok=True)
    return [
        write_hypothesis_panel_flow_plot(
            plot_root,
            hypothesis_panel_selection_trace_rows=hypothesis_panel_selection_trace_rows,
            input_hashes=input_hashes,
        ),
        write_local_structure_stratification_plot(
            plot_root,
            triage_rows=triage_rows,
            panel_rows=panel_rows,
            local_structure_rows=local_structure_rows,
            input_hashes=input_hashes,
        ),
        write_local_structure_by_region_plot(
            plot_root,
            panel_rows=panel_rows,
            local_structure_rows=local_structure_rows,
            input_hashes=input_hashes,
        ),
        write_selected_substitutions_across_rt_plot(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
            rt_annotation_context=rt_annotation_context,
        ),
        write_n_terminal_pair_comparison_plot(
            plot_root,
            panel_rows=panel_rows,
            canonical_sequences_by_id=canonical_sequences_by_id,
            input_hashes=input_hashes,
        ),
        write_regional_mutation_burden_plot(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            mask_residues=mask_residues,
            input_hashes=input_hashes,
        ),
        write_na_facing_chemistry_balance_plot(
            plot_root,
            panel_rows=panel_rows,
            triage_rows=triage_rows,
            input_hashes=input_hashes,
        ),
        write_regionwise_msa_support_plot(
            plot_root,
            panel_rows=panel_rows,
            region_msa_support_rows=region_msa_support_rows,
            input_hashes=input_hashes,
        ),
        write_selected_mutation_dissimilarity_plot(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            triage_rows=triage_rows,
            input_hashes=input_hashes,
        ),
        write_selected_sequence_distance_plot(
            plot_root,
            panel_rows=panel_rows,
            candidate_rows=candidate_rows,
            input_hashes=input_hashes,
        ),
        write_local_structure_threshold_sensitivity_plot(
            plot_root,
            threshold_sensitivity_rows=local_structure_threshold_sensitivity_rows,
            input_hashes=input_hashes,
        ),
    ]
