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

from dnadesign.cruncher.analysis.plots.opt_trajectory import (
    plot_chain_trajectory_scatter,
    plot_chain_trajectory_sweep,
)


def test_chain_trajectory_scatter_requires_scale_columns(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
        }
    )
    baseline_df = pd.DataFrame({"raw_llr_lexA": [0.1], "raw_llr_cpxR": [0.2]})
    out_path = tmp_path / "plot__chain_trajectory_scatter.png"

    with pytest.raises(ValueError, match="raw_llr_lexA"):
        plot_chain_trajectory_scatter(
            trajectory_df=trajectory_df,
            baseline_df=baseline_df,
            tf_pair=("lexA", "cpxR"),
            scatter_scale="llr",
            consensus_anchors=None,
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )


def test_chain_trajectory_scatter_renders_background_chain_and_consensus(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
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
    out_path = tmp_path / "plot__chain_trajectory_scatter.png"
    metadata = plot_chain_trajectory_scatter(
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
    assert any(label.startswith("chain 0") for label in labels)
    assert any(label.startswith("chain 1") for label in labels)
    assert "consensus anchors" in labels
    assert metadata["chain_count"] == 2
    assert metadata["mode"] == "chain_scatter"


def test_chain_trajectory_sweep_requires_selected_y_column(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
        }
    )
    out_path = tmp_path / "plot__chain_trajectory_sweep.png"

    with pytest.raises(ValueError, match="raw_llr_objective"):
        plot_chain_trajectory_sweep(
            trajectory_df=trajectory_df,
            y_column="raw_llr_objective",
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )


def test_chain_trajectory_sweep_renders_raw_llr_by_sweep(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "raw_llr_objective": [0.40, 0.32, 0.28, 0.35, 0.55, 0.40],
            "phase": ["tune", "draw", "draw", "tune", "draw", "draw"],
        }
    )
    out_path = tmp_path / "plot__chain_trajectory_sweep.png"
    metadata = plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="raw_llr_objective",
        y_mode="all",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        stride=1,
        slot_overlay=True,
    )

    assert out_path.exists()
    legend_labels = metadata["legend_labels"]
    assert "chain 0" in legend_labels
    assert "chain 1" in legend_labels
    assert metadata["mode"] == "chain_sweep"
    assert metadata["chain_count"] == 2
    assert metadata["y_mode"] == "all"


def test_chain_trajectory_sweep_best_so_far_mode_is_supported(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "raw_llr_objective": [0.40, 0.32, 0.28, 0.35, 0.55, 0.40],
            "phase": ["tune", "draw", "draw", "tune", "draw", "draw"],
        }
    )
    out_path = tmp_path / "plot__chain_trajectory_sweep_best.png"
    metadata = plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="raw_llr_objective",
        y_mode="best_so_far",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
    )

    assert out_path.exists()
    assert metadata["y_mode"] == "best_so_far"
    assert metadata["chain_count"] == 2


def test_chain_trajectory_sweep_rejects_unknown_mode(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
            "raw_llr_objective": [0.4, 0.5],
        }
    )
    out_path = tmp_path / "plot__chain_trajectory_sweep_invalid.png"
    with pytest.raises(ValueError, match="y_mode"):
        plot_chain_trajectory_sweep(
            trajectory_df=trajectory_df,
            y_column="raw_llr_objective",
            y_mode="invalid",
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )
