"""Contracts for Stage B realized-label review plots."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_review_plots.v1"
REALIZED_REVIEW_PLOT_MANIFEST_FILENAME = "tfbs_stage_b_realized_label_plot_manifest.json"
REALIZED_REVIEW_INTERPRETATION_BOUNDARY = (
    "These plots use selected label values from DenseGen and profile-specific control label tables. For shuffled "
    "controls, control values are control-label values; post hoc DenseGen-truth checks must be named separately. "
    "These are selected-batch enrichment review surfaces, not acquisition-score traces or monotonic model-learning "
    "curves."
)
REALIZED_REVIEW_STYLE_CONTRACT = {
    "axis_style": "stress_ethanol_cipro_growth.tfbs_review_axis.v1",
    "axes_facecolor": "white",
    "grid": "light_gray_background_grid_lines",
    "visible_spines": ["left", "bottom"],
    "tick_style": "styled_outward_ticks",
    "font_scale": "unified_review_body_font_for_ticks_axes_subtitle_legend",
    "square_axes": "where_data_shape_supports_it",
    "trajectory_axes": "square",
    "trajectory_reference_lines": ["baseline", "same_batch_top_k_reference"],
}

RealizedReviewRenderer = Callable[[pd.DataFrame, pd.DataFrame, Path, str], None]
