"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_b_review_plot_layout.py

Regression tests for TFBS stage b review plot studies units stress.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from .helpers import _dark_edge_pixel_count
from .probe_modules import probe_module

realized_visual_spec = probe_module("tfbs.stage_b.notebook_visuals.specs").realized_visual_spec
realized_review_renderer = probe_module("tfbs.stage_b.review.plots.renderers").realized_review_renderer


def test_stage_b_review_long_plot_titles_stay_inside_square_canvas(tmp_path: Path) -> None:
    """Guard long TFBS labels, legends, baseline, and top-K references against clipping."""

    for label_name in ("baeR_count_fraction", "cpxR_or_baeR_in_slot2"):
        trajectory = pd.DataFrame(
            {
                "label_name": [label_name] * 6,
                "oracle_role": ["positive"] * 3 + ["matched_null"] * 3,
                "null_control_role": ["count_fixed_shuffled_slot_negative_control"] * 6,
                "round": [0, 1, 23, 0, 1, 23],
                "same_batch_top_lift_ratio": [4.8, 4.8, 4.8, 3.1, 3.1, 3.1],
                "selected_true_lift_ratio": [1.1, 2.0, 4.3, 1.0, 1.4, 1.8],
                "seed_true_lift_ratio": [0.9, 0.9, 0.9, 1.0, 1.0, 1.0],
            }
        )
        pair_summary = pd.DataFrame(
            {
                "label_name": [label_name],
                "final_positive_minus_null_lift_ratio": [2.5],
                "trapezoid_auc_positive_minus_null_lift_ratio": [1.9],
            }
        )

        for kind in ("realized_label_lift_trajectory", "positive_null_lift_summary"):
            path = tmp_path / f"{label_name}__{kind}.png"
            renderer = realized_review_renderer(realized_visual_spec(kind))
            renderer(trajectory, pair_summary, path, label_name)

            image = Image.open(path).convert("RGB")
            assert image.size == (1152, 1152)
            assert _dark_edge_pixel_count(image) == 0
