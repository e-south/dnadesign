"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_opt_trajectory_story_plot.py

Validate story trajectory plot semantics and metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.plots.opt_trajectory import plot_opt_trajectory_story


def test_opt_trajectory_story_plot_includes_expected_legend_and_consensus(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep": [0, 1, 0, 1],
            "phase": ["tune", "draw", "tune", "draw"],
            "is_cold_chain": [0, 0, 1, 1],
            "objective_scalar": [0.1, 0.2, 0.15, 0.25],
            "x": [0.10, 0.30, 0.20, 0.40],
            "y": [0.20, 0.25, 0.30, 0.45],
            "x_metric": ["score_lexA"] * 4,
            "y_metric": ["score_cpxR"] * 4,
        }
    )
    baseline_df = pd.DataFrame(
        {
            "score_lexA": [0.05, 0.10, 0.15, 0.20],
            "score_cpxR": [0.03, 0.08, 0.13, 0.18],
        }
    )
    selected_df = pd.DataFrame(
        {
            "x": [0.42, 0.44],
            "y": [0.46, 0.48],
        }
    )
    consensus_anchors = [
        {"tf": "lexA", "x": 0.95, "y": 0.20, "label": "lexA consensus (max)"},
        {"tf": "cpxR", "x": 0.25, "y": 0.96, "label": "cpxR consensus (max)"},
    ]

    out_path = tmp_path / "plot__opt_trajectory_story.png"
    metadata = plot_opt_trajectory_story(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        selected_df=selected_df,
        consensus_anchors=consensus_anchors,
        tf_names=["lexA", "cpxR"],
        out_path=out_path,
        score_scale="normalized-llr",
        dpi=72,
        png_compress_level=1,
    )

    assert out_path.exists()
    legend_labels = metadata["legend_labels"]
    assert any(label.startswith("random baseline") for label in legend_labels)
    assert "best-so-far" in legend_labels
    assert "selected top-k" in legend_labels
    assert "all chains (context)" not in legend_labels
    assert "cold tune→draw" not in legend_labels
    anchors = metadata["consensus_anchors"]
    assert len(anchors) == 2
    assert {item["tf"] for item in anchors} == {"lexA", "cpxR"}
