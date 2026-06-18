"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/slot_diagnostics/plots/contracts.py

Contracts for Stage B slot-diagnostic plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_slot_diagnostic_plots.v1"
SLOT_DIAGNOSTIC_PLOT_MANIFEST_FILENAME = "tfbs_stage_b_slot_diagnostic_plot_manifest.json"
SLOT_DIAGNOSTIC_INTERPRETATION_BOUNDARY = (
    "These plots diagnose whether slot-label enrichment is explained by target-family count. "
    "They are diagnostic evidence surfaces, not clean negative-control claims by themselves."
)
SLOT_DIAGNOSTIC_STYLE_CONTRACT = {
    "axis_style": "stress_ethanol_cipro_growth.tfbs_review_axis.v1",
    "axes_facecolor": "white",
    "grid": "light_gray_background_grid_lines",
    "visible_spines": ["left", "bottom"],
    "tick_style": "styled_outward_ticks",
    "font_scale": "larger_review_tick_and_axis_labels",
    "square_axes": "where_data_shape_supports_it",
    "trajectory_layout": "single_row_square_panels_for_label_trajectories",
    "legend_layout": "single row below the plot",
}

SlotDiagnosticRenderer = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path], None]
