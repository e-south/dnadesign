"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/learning_loop_baselines/plots/contracts.py

Plot contracts for TFBS learning-loop baseline reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

FROZEN_REPLAY_STYLE_CONTRACT = {
    "axis_style": "stress_ethanol_cipro_growth.tfbs_review_axis.v1",
    "axes_facecolor": "white",
    "grid": "light_gray_background_grid_lines",
    "visible_spines": ["left", "bottom"],
    "tick_style": "styled_outward_ticks",
    "font_scale": "unified_review_body_font_for_ticks_axes_subtitle_legend",
    "interval_kind": "sample_sd_across_seed_runs",
    "interval_is_confidence_interval": False,
    "subplot_layout": "single_row_square_panels_for_label_trajectories",
    "bar_summary_axes": "square",
    "trajectory_reference_series": ["pool_average", "same_budget_known_label_ranking"],
    "categorical_encoding": {
        "color": "label_source",
        "line_style": "active_retraining_vs_frozen_round0_vs_reference",
        "marker_shape": "active_retraining_vs_frozen_round0_vs_reference",
        "palette": "colorblind_review_palette",
    },
    "label_order": "composition_regulator_order_or_placement_left_middle_right",
}
