"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_trajectory_score_space_panel_boundaries.py

Characterization tests for trajectory score-space panel rendering boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from dnadesign.cruncher.analysis.plots.trajectory_score_space_panel import (
    _render_score_space_panel,
    _resolve_grid_pairs,
    _resolve_score_scale,
)


def test_resolve_score_scale_accepts_supported_values() -> None:
    assert _resolve_score_scale("llr") == ("raw_llr_", "raw LLR")
    assert _resolve_score_scale("normalized-llr") == ("norm_llr_", "normalized LLR")

    with pytest.raises(ValueError, match="scatter_scale"):
        _resolve_score_scale("bad-scale")


def test_resolve_grid_pairs_auto_builds_unique_pairs() -> None:
    assert _resolve_grid_pairs(tf_list=["lexA", "cpxR", "marA"], tf_pairs_grid=None) == [
        ("lexA", "cpxR"),
        ("lexA", "marA"),
        ("cpxR", "marA"),
    ]

    with pytest.raises(ValueError, match="at least two TFs"):
        _resolve_grid_pairs(tf_list=["lexA"], tf_pairs_grid=None)


def test_render_score_space_panel_honors_edge_only_axis_labels() -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
            "raw_llr_lexA": [0.1, 0.2],
            "raw_llr_cpxR": [0.2, 0.3],
        }
    )
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    try:
        elite_stats, legend_labels = _render_score_space_panel(
            ax=ax,
            panel_traj_df=trajectory_df,
            panel_baseline_df=pd.DataFrame(),
            panel_elites_df=None,
            x_col="raw_llr_lexA",
            y_col="raw_llr_cpxR",
            x_label="lexA best-window raw LLR",
            y_label="cpxR best-window raw LLR",
            title="Selected elites in TF score space",
            panel_anchors=None,
            panel_y_tf="cpxR",
            show_legend=False,
            show_x_label=False,
            show_y_label=False,
            retain_elites=True,
            legend_fontsize=12,
            anchor_annotation_fontsize=11,
        )
    finally:
        plt.close(fig)

    assert elite_stats["total"] == 0
    assert legend_labels == []
    assert ax.get_xlabel() == ""
    assert ax.get_ylabel() == ""
