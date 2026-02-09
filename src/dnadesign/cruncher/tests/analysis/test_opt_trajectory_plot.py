"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_opt_trajectory_plot.py

Validate optimization trajectory scatter and sweep plot semantics and metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.cruncher.analysis.plots.opt_trajectory import plot_opt_trajectory, plot_opt_trajectory_sweep


def test_opt_trajectory_scatter_requires_scale_columns(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "particle_id": [0, 0],
            "sweep_idx": [0, 1],
            "slot_id": [0, 0],
        }
    )
    baseline_df = pd.DataFrame({"raw_llr_lexA": [0.1], "raw_llr_cpxR": [0.2]})
    out_path = tmp_path / "plot__opt_trajectory.png"

    with pytest.raises(ValueError, match="raw_llr_lexA"):
        plot_opt_trajectory(
            trajectory_df=trajectory_df,
            baseline_df=baseline_df,
            tf_pair=("lexA", "cpxR"),
            scatter_scale="llr",
            consensus_anchors=None,
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )


def test_opt_trajectory_scatter_renders_background_particle_and_consensus(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "particle_id": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "slot_id": [0, 1, 1, 1, 0, 0],
            "raw_llr_lexA": [0.10, 0.30, 0.45, 0.08, 0.22, 0.35],
            "raw_llr_cpxR": [0.12, 0.33, 0.47, 0.09, 0.25, 0.37],
            "norm_llr_lexA": [0.05, 0.20, 0.35, 0.03, 0.15, 0.28],
            "norm_llr_cpxR": [0.07, 0.22, 0.36, 0.04, 0.17, 0.29],
            "raw_llr_objective": [0.10, 0.30, 0.45, 0.08, 0.22, 0.35],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.02, 0.04, 0.06, 0.08],
            "raw_llr_cpxR": [0.03, 0.05, 0.07, 0.09],
            "norm_llr_lexA": [0.01, 0.03, 0.05, 0.07],
            "norm_llr_cpxR": [0.02, 0.04, 0.06, 0.08],
        }
    )
    consensus_anchors = [
        {"tf": "lexA", "label": "lexA consensus (max)", "x": 0.95, "y": 0.20},
        {"tf": "cpxR", "label": "cpxR consensus (max)", "x": 0.20, "y": 0.96},
    ]
    out_path = tmp_path / "plot__opt_trajectory.png"
    metadata = plot_opt_trajectory(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=consensus_anchors,
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        slot_overlay=True,
    )

    assert out_path.exists()
    labels = metadata["legend_labels"]
    assert any(label.startswith("random baseline") for label in labels)
    assert any(label.startswith("particle lineage (id=0..") for label in labels)
    assert "consensus anchors" in labels
    assert metadata["particle_count"] == 2
    assert metadata["mode"] == "particle_scatter"


def test_opt_trajectory_sweep_requires_selected_y_column(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "particle_id": [0, 0],
            "sweep_idx": [0, 1],
        }
    )
    out_path = tmp_path / "plot__opt_trajectory_sweep.png"

    with pytest.raises(ValueError, match="raw_llr_objective"):
        plot_opt_trajectory_sweep(
            trajectory_df=trajectory_df,
            y_column="raw_llr_objective",
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )


def test_opt_trajectory_sweep_requires_cold_slot_annotation(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "particle_id": [0, 1],
            "sweep_idx": [0, 0],
            "slot_id": [0, 1],
            "raw_llr_objective": [0.20, 0.30],
        }
    )
    out_path = tmp_path / "plot__opt_trajectory_sweep.png"

    with pytest.raises(ValueError, match="is_cold_chain"):
        plot_opt_trajectory_sweep(
            trajectory_df=trajectory_df,
            y_column="raw_llr_objective",
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )


def test_opt_trajectory_sweep_renders_particle_raw_llr_by_sweep(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "particle_id": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "slot_id": [0, 1, 1, 1, 0, 0],
            "is_cold_chain": [1, 0, 0, 0, 1, 1],
            "raw_llr_objective": [0.40, 0.32, 0.28, 0.35, 0.55, 0.40],
            "raw_llr_lexA": [0.50, 0.40, 0.45, 0.40, 0.70, 0.40],
            "raw_llr_cpxR": [0.40, 0.32, 0.28, 0.35, 0.55, 0.58],
            "phase": ["tune", "draw", "draw", "tune", "draw", "draw"],
        }
    )
    out_path = tmp_path / "plot__opt_trajectory_sweep.png"
    metadata = plot_opt_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="raw_llr_objective",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        stride=1,
        slot_overlay=True,
    )

    assert out_path.exists()
    legend_labels = metadata["legend_labels"]
    assert "cold-slot progression" in legend_labels
    assert "lineage handoff (slot swap)" in legend_labels
    assert "bottleneck TF: lexA" in legend_labels
    assert "bottleneck TF: cpxR" in legend_labels
    assert metadata["mode"] == "cold_slot_sweep"
    assert metadata["cold_point_count"] == 3
    assert metadata["cold_handoff_count"] == 1
