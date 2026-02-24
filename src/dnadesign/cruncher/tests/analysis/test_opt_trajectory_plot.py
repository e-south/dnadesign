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
    plot_chain_trajectory_sweep,
    plot_elite_score_space_context,
)


def test_chain_trajectory_scatter_requires_scale_columns(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
        }
    )
    baseline_df = pd.DataFrame({"raw_llr_lexA": [0.1], "raw_llr_cpxR": [0.2]})
    out_path = tmp_path / "chain_trajectory_scatter.png"

    with pytest.raises(ValueError, match="raw_llr_lexA"):
        plot_elite_score_space_context(
            trajectory_df=trajectory_df,
            baseline_df=baseline_df,
            tf_pair=("lexA", "cpxR"),
            scatter_scale="llr",
            consensus_anchors=None,
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )


def test_chain_trajectory_scatter_renders_without_random_baseline(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep_idx": [0, 1, 0, 1],
            "raw_llr_lexA": [0.10, 0.20, 0.12, 0.24],
            "raw_llr_cpxR": [0.11, 0.21, 0.13, 0.25],
            "norm_llr_lexA": [0.04, 0.14, 0.06, 0.18],
            "norm_llr_cpxR": [0.05, 0.15, 0.07, 0.19],
            "objective_scalar": [0.10, 0.20, 0.12, 0.24],
            "raw_llr_objective": [0.10, 0.20, 0.12, 0.24],
        }
    )
    out_path = tmp_path / "chain_trajectory_scatter_no_baseline.png"

    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=pd.DataFrame(),
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
    )

    assert out_path.exists()
    labels = metadata["legend_labels"]
    assert not any(label.startswith("Random baseline") for label in labels)


def test_chain_trajectory_scatter_renders_background_and_consensus_without_chain_legend(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "raw_llr_lexA": [0.10, 0.30, 0.45, 0.08, 0.22, 0.35],
            "raw_llr_cpxR": [0.12, 0.33, 0.47, 0.09, 0.25, 0.37],
            "norm_llr_lexA": [0.05, 0.20, 0.35, 0.03, 0.15, 0.28],
            "norm_llr_cpxR": [0.07, 0.22, 0.36, 0.04, 0.17, 0.29],
            "objective_scalar": [0.10, 0.30, 0.45, 0.08, 0.22, 0.35],
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
        {"tf": "lexA", "label": "lexA consensus", "x": 0.95, "y": 0.20},
        {"tf": "cpxR", "label": "cpxR consensus", "x": 0.20, "y": 0.96},
    ]
    out_path = tmp_path / "chain_trajectory_scatter.png"
    elites_df = pd.DataFrame(
        {
            "id": ["elite-1", "elite-2"],
            "raw_llr_lexA": [0.45, 0.35],
            "raw_llr_cpxR": [0.47, 0.37],
            "norm_llr_lexA": [0.35, 0.28],
            "norm_llr_cpxR": [0.36, 0.29],
            "rank": [1, 2],
        }
    )
    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=consensus_anchors,
        objective_caption="Joint objective: maximize min_TF(score_TF).",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
    )

    assert out_path.exists()
    labels = metadata["legend_labels"]
    assert any(label.startswith("Random baseline") for label in labels)
    assert not any(label.startswith("Chain ") for label in labels)
    assert "TF consensus anchors" in labels
    assert metadata["mode"] == "elite_score_space_context"
    assert metadata["legend_fontsize"] == 12
    assert metadata["anchor_annotation_fontsize"] == 11
    assert metadata["elite_points_plotted"] == 2
    assert "maximize min_TF(score_TF)" in metadata["objective_caption"]


def test_chain_trajectory_scatter_reports_context_only_metadata(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 0, 1, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 3, 0, 1, 2, 3],
            "raw_llr_lexA": [0.10, 0.15, 0.14, 0.20, 0.20, 0.18, 0.25, 0.24],
            "raw_llr_cpxR": [0.09, 0.12, 0.11, 0.14, 0.19, 0.17, 0.22, 0.21],
            "norm_llr_lexA": [0.05, 0.08, 0.07, 0.11, 0.09, 0.07, 0.12, 0.10],
            "norm_llr_cpxR": [0.04, 0.07, 0.06, 0.09, 0.08, 0.06, 0.10, 0.09],
            "objective_scalar": [0.10, 0.30, 0.25, 0.40, 0.20, 0.18, 0.50, 0.45],
            "raw_llr_objective": [0.10, 0.30, 0.25, 0.40, 0.20, 0.18, 0.50, 0.45],
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
    out_path = tmp_path / "chain_trajectory_scatter_best_updates.png"
    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=None,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
    )

    assert out_path.exists()
    assert metadata["chain_count"] == 0
    assert metadata["plotted_points_by_chain"] == {}
    assert metadata["best_update_points_by_chain"] == {}


def test_chain_trajectory_scatter_preserves_exact_elite_mapping_without_paths(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0] * 12,
            "sweep_idx": list(range(12)),
            "raw_llr_lexA": [float(v) for v in range(12)],
            "raw_llr_cpxR": [float(v) + 0.01 for v in range(12)],
            "norm_llr_lexA": [float(v) for v in range(12)],
            "norm_llr_cpxR": [float(v) + 0.01 for v in range(12)],
            "objective_scalar": [0.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 11.0, 4.0, 3.0, 2.0, 1.0],
            "raw_llr_objective": [0.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 11.0, 4.0, 3.0, 2.0, 1.0],
            "sequence": [f"seq-{idx}" for idx in range(12)],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.0, 0.1, 0.2],
            "raw_llr_cpxR": [0.0, 0.1, 0.2],
            "norm_llr_lexA": [0.0, 0.1, 0.2],
            "norm_llr_cpxR": [0.0, 0.1, 0.2],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["seq-3", "seq-7"],
            "raw_llr_lexA": [3.0, 7.0],
            "raw_llr_cpxR": [3.01, 7.01],
            "norm_llr_lexA": [3.0, 7.0],
            "norm_llr_cpxR": [3.01, 7.01],
        }
    )
    metadata_stride4 = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=tmp_path / "chain_trajectory_scatter_stride4.png",
        dpi=72,
        png_compress_level=1,
    )

    assert metadata_stride4["chain_count"] == 0
    assert metadata_stride4["elite_exact_mapped_points"] == 2
    assert metadata_stride4["elite_path_link_count"] == 0


def test_chain_trajectory_scatter_best_progression_does_not_draw_exact_elite_connector_edges(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 0],
            "sweep_idx": [0, 1, 2, 3],
            "raw_llr_lexA": [0.0, 10.0, 2.0, 3.0],
            "raw_llr_cpxR": [0.0, 10.0, 2.0, 3.0],
            "norm_llr_lexA": [0.0, 10.0, 2.0, 3.0],
            "norm_llr_cpxR": [0.0, 10.0, 2.0, 3.0],
            "objective_scalar": [0.0, 10.0, 2.0, 3.0],
            "raw_llr_objective": [0.0, 10.0, 2.0, 3.0],
            "sequence": ["seq-0", "seq-1", "seq-2", "seq-3"],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.0, 0.1, 0.2],
            "raw_llr_cpxR": [0.0, 0.1, 0.2],
            "norm_llr_lexA": [0.0, 0.1, 0.2],
            "norm_llr_cpxR": [0.0, 0.1, 0.2],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["seq-3"],
            "raw_llr_lexA": [3.0],
            "raw_llr_cpxR": [3.0],
            "norm_llr_lexA": [3.0],
            "norm_llr_cpxR": [3.0],
        }
    )
    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=tmp_path / "chain_trajectory_scatter_elite_link.png",
        dpi=72,
        png_compress_level=1,
    )

    assert metadata["elite_exact_mapped_points"] == 1
    assert metadata["elite_path_link_count"] == 0
    assert metadata["elite_snapped_to_path_count"] == 0


def test_chain_trajectory_scatter_best_progression_keeps_true_elite_coordinates(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 0],
            "sweep_idx": [0, 1, 2, 3],
            "raw_llr_lexA": [0.0, 10.0, 2.0, 3.0],
            "raw_llr_cpxR": [0.0, 10.0, 2.0, 3.0],
            "norm_llr_lexA": [0.0, 10.0, 2.0, 3.0],
            "norm_llr_cpxR": [0.0, 10.0, 2.0, 3.0],
            "objective_scalar": [0.0, 10.0, 2.0, 3.0],
            "raw_llr_objective": [0.0, 10.0, 2.0, 3.0],
            "sequence": ["seq-0", "seq-1", "seq-2", "seq-3"],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.0, 0.1, 0.2],
            "raw_llr_cpxR": [0.0, 0.1, 0.2],
            "norm_llr_lexA": [0.0, 0.1, 0.2],
            "norm_llr_cpxR": [0.0, 0.1, 0.2],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["seq-2", "seq-3"],
            "raw_llr_lexA": [2.0, 3.0],
            "raw_llr_cpxR": [2.0, 3.0],
            "norm_llr_lexA": [2.0, 3.0],
            "norm_llr_cpxR": [2.0, 3.0],
        }
    )

    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=tmp_path / "chain_trajectory_scatter_best_progression_true_elites.png",
        dpi=72,
        png_compress_level=1,
    )

    assert metadata["elite_exact_mapped_points"] == 2
    assert metadata["elite_unique_coordinates"] == 2
    assert metadata["elite_path_link_count"] == 0
    assert metadata["elite_snapped_to_path_count"] == 0


def test_chain_trajectory_scatter_rejects_removed_scatter_mode_kwargs(
    tmp_path: Path,
) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0] * 9,
            "sweep_idx": list(range(9)),
            "raw_llr_lexA": [float(v) for v in range(9)],
            "raw_llr_cpxR": [float(v) + 0.5 for v in range(9)],
            "norm_llr_lexA": [float(v) for v in range(9)],
            "norm_llr_cpxR": [float(v) + 0.5 for v in range(9)],
            "objective_scalar": [float(v) for v in range(9)],
            "raw_llr_objective": [float(v) for v in range(9)],
            "sequence": [f"seq-{idx}" for idx in range(9)],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.0, 0.1, 0.2],
            "raw_llr_cpxR": [0.0, 0.1, 0.2],
            "norm_llr_lexA": [0.0, 0.1, 0.2],
            "norm_llr_cpxR": [0.0, 0.1, 0.2],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["seq-4"],
            "raw_llr_lexA": [4.0],
            "raw_llr_cpxR": [4.5],
            "norm_llr_lexA": [4.0],
            "norm_llr_cpxR": [4.5],
        }
    )

    with pytest.raises(TypeError, match="scatter_mode"):
        plot_elite_score_space_context(
            trajectory_df=trajectory_df,
            baseline_df=baseline_df,
            elites_df=elites_df,
            tf_pair=("lexA", "cpxR"),
            scatter_scale="llr",
            consensus_anchors=None,
            out_path=tmp_path / "chain_trajectory_scatter_elite_context_stubs.png",
            dpi=72,
            png_compress_level=1,
            scatter_mode="elite_context_stubs",
        )


def test_chain_trajectory_scatter_prefers_match_column_with_actual_overlap(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0],
            "sweep_idx": [0, 1, 2],
            "raw_llr_lexA": [0.0, 0.2, 0.4],
            "raw_llr_cpxR": [0.0, 0.3, 0.5],
            "norm_llr_lexA": [0.0, 0.2, 0.4],
            "norm_llr_cpxR": [0.0, 0.3, 0.5],
            "objective_scalar": [0.0, 0.2, 0.4],
            "raw_llr_objective": [0.0, 0.2, 0.4],
            "sequence": ["AAA", "CCC", "GGG"],
            "sequence_hash": ["h1", "h2", "h3"],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.0, 0.1, 0.2],
            "raw_llr_cpxR": [0.0, 0.1, 0.2],
            "norm_llr_lexA": [0.0, 0.1, 0.2],
            "norm_llr_cpxR": [0.0, 0.1, 0.2],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["GGG"],
            "sequence_hash": ["not-a-match"],
            "raw_llr_lexA": [0.4],
            "raw_llr_cpxR": [0.5],
            "norm_llr_lexA": [0.4],
            "norm_llr_cpxR": [0.5],
        }
    )
    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=tmp_path / "chain_trajectory_scatter_match_column.png",
        dpi=72,
        png_compress_level=1,
    )

    assert metadata["elite_exact_mapped_points"] == 1


def test_chain_trajectory_scatter_reports_elite_coordinate_collisions(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep_idx": [0, 1, 0, 1],
            "raw_llr_lexA": [0.10, 0.30, 0.08, 0.35],
            "raw_llr_cpxR": [0.12, 0.33, 0.09, 0.37],
            "norm_llr_lexA": [0.05, 0.20, 0.03, 0.28],
            "norm_llr_cpxR": [0.07, 0.22, 0.04, 0.29],
            "objective_scalar": [0.10, 0.30, 0.08, 0.35],
            "raw_llr_objective": [0.10, 0.30, 0.08, 0.35],
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
    elites_df = pd.DataFrame(
        {
            "id": ["elite-1", "elite-2", "elite-3"],
            "raw_llr_lexA": [0.35, 0.35, 0.30],
            "raw_llr_cpxR": [0.37, 0.37, 0.33],
            "norm_llr_lexA": [0.28, 0.28, 0.20],
            "norm_llr_cpxR": [0.29, 0.29, 0.22],
            "rank": [1, 2, 3],
        }
    )
    out_path = tmp_path / "chain_trajectory_scatter_elite_collisions.png"
    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
    )

    assert out_path.exists()
    assert metadata["elite_points_plotted"] == 3
    assert metadata["elite_unique_coordinates"] == 2
    assert metadata["elite_coordinate_collisions"] == 1
    assert metadata["elite_rendered_points"] == 2
    assert metadata["elite_collision_annotation_count"] == 0


def test_chain_trajectory_sweep_requires_selected_y_column(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
        }
    )
    out_path = tmp_path / "chain_trajectory_sweep.png"

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
    out_path = tmp_path / "chain_trajectory_sweep.png"
    metadata = plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="raw_llr_objective",
        y_mode="all",
        cooling_config={
            "kind": "piecewise",
            "stages": [
                {"sweeps": 2, "beta": 0.2},
                {"sweeps": 4, "beta": 1.0},
                {"sweeps": 6, "beta": 4.0},
            ],
        },
        tune_sweeps=2,
        objective_caption="Joint objective: maximize min_TF(score_TF).",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        stride=1,
        chain_overlay=True,
    )

    assert out_path.exists()
    legend_labels = metadata["legend_labels"]
    assert "Chain 0" in legend_labels
    assert "Chain 1" in legend_labels
    assert metadata["mode"] == "chain_sweep"
    assert metadata["chain_count"] == 2
    assert metadata["y_mode"] == "all"
    assert metadata["cooling_stage_count"] == 3
    assert metadata["tune_boundary_sweep"] == 2
    assert "maximize min_TF(score_TF)" in metadata["objective_caption"]


def test_chain_trajectory_sweep_best_so_far_mode_is_supported(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "raw_llr_objective": [0.40, 0.32, 0.28, 0.35, 0.55, 0.40],
            "phase": ["tune", "draw", "draw", "tune", "draw", "draw"],
        }
    )
    out_path = tmp_path / "chain_trajectory_sweep_best.png"
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
    assert metadata["y_label"] == "Replay objective (raw-LLR)"


def test_chain_trajectory_sweep_best_so_far_is_computed_before_stride(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0] * 10,
            "sweep_idx": list(range(10)),
            "raw_llr_objective": [0.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "phase": ["draw"] * 10,
        }
    )
    out_path = tmp_path / "chain_trajectory_sweep_best_stride.png"
    metadata = plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="raw_llr_objective",
        y_mode="best_so_far",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        stride=4,
    )

    assert out_path.exists()
    assert metadata["best_final_by_chain"][0] == pytest.approx(10.0)
    assert 1 in metadata["sampled_sweep_indices_by_chain"][0]
    assert metadata["best_drawstyle"] == "steps-post"


def test_chain_trajectory_sweep_summary_overlay_is_disabled_by_default(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "raw_llr_objective": [0.4, 0.2, 0.3, 0.2, 0.5, 0.6],
        }
    )
    out_path = tmp_path / "chain_trajectory_sweep_summary_default.png"
    metadata = plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="raw_llr_objective",
        y_mode="best_so_far",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
    )

    assert out_path.exists()
    assert metadata["summary_overlay_enabled"] is False
    assert metadata["summary_points"] == 0


def test_chain_trajectory_sweep_summary_overlay_retains_peak_under_stride(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0] * 10 + [1] * 10,
            "sweep_idx": list(range(10)) + list(range(10)),
            "raw_llr_objective": [0.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    out_path = tmp_path / "chain_trajectory_sweep_summary_stride.png"
    metadata = plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="raw_llr_objective",
        y_mode="raw",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        stride=4,
        summary_overlay=True,
    )

    assert out_path.exists()
    assert metadata["summary_overlay_enabled"] is True
    assert metadata["summary_peak_value"] == pytest.approx(5.0)


def test_chain_trajectory_sweep_can_disable_summary_overlay(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "raw_llr_objective": [0.4, 0.2, 0.3, 0.2, 0.5, 0.6],
        }
    )
    out_path = tmp_path / "chain_trajectory_sweep_summary_off.png"
    metadata = plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="raw_llr_objective",
        y_mode="best_so_far",
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        summary_overlay=False,
    )

    assert out_path.exists()
    assert metadata["summary_overlay_enabled"] is False
    assert metadata["summary_points"] == 0


def test_chain_trajectory_sweep_objective_scalar_label_is_compact_and_specific(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
            "objective_scalar": [0.4, 0.5],
        }
    )
    out_path = tmp_path / "chain_trajectory_sweep_objective_scalar.png"
    metadata = plot_chain_trajectory_sweep(
        trajectory_df=trajectory_df,
        y_column="objective_scalar",
        y_mode="best_so_far",
        objective_config={
            "combine": "min",
            "score_scale": "normalized-llr",
            "softmin": {"enabled": True},
        },
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
    )

    assert out_path.exists()
    assert metadata["y_label"] == "Soft-min TF best-window norm-LLR"


def test_chain_trajectory_sweep_rejects_unknown_mode(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0],
            "sweep_idx": [0, 1],
            "raw_llr_objective": [0.4, 0.5],
        }
    )
    out_path = tmp_path / "chain_trajectory_sweep_invalid.png"
    with pytest.raises(ValueError, match="y_mode"):
        plot_chain_trajectory_sweep(
            trajectory_df=trajectory_df,
            y_column="raw_llr_objective",
            y_mode="invalid",
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )


def test_chain_trajectory_scatter_worst_second_mode_for_multi_tf(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep_idx": [0, 1, 0, 1],
            "raw_llr_lexA": [0.10, 0.60, 0.20, 0.40],
            "raw_llr_cpxR": [0.30, 0.50, 0.10, 0.80],
            "raw_llr_fur": [0.20, 0.70, 0.40, 0.30],
            "norm_llr_lexA": [0.05, 0.30, 0.10, 0.20],
            "norm_llr_cpxR": [0.15, 0.25, 0.05, 0.40],
            "norm_llr_fur": [0.10, 0.35, 0.20, 0.15],
            "objective_scalar": [0.10, 0.50, 0.10, 0.30],
            "raw_llr_objective": [0.10, 0.50, 0.10, 0.30],
            "sequence": ["s0", "s1", "s2", "s3"],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.01, 0.02, 0.03],
            "raw_llr_cpxR": [0.04, 0.03, 0.02],
            "raw_llr_fur": [0.02, 0.01, 0.04],
            "norm_llr_lexA": [0.01, 0.02, 0.03],
            "norm_llr_cpxR": [0.04, 0.03, 0.02],
            "norm_llr_fur": [0.02, 0.01, 0.04],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["s1", "s3"],
            "raw_llr_lexA": [0.60, 0.40],
            "raw_llr_cpxR": [0.50, 0.80],
            "raw_llr_fur": [0.70, 0.30],
            "norm_llr_lexA": [0.30, 0.20],
            "norm_llr_cpxR": [0.25, 0.40],
            "norm_llr_fur": [0.35, 0.15],
        }
    )
    out_path = tmp_path / "chain_trajectory_scatter_worst_second.png"
    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        score_space_mode="worst_vs_second_worst",
        tf_names=["lexA", "cpxR", "fur"],
    )

    assert out_path.exists()
    assert metadata["score_space_mode"] == "worst_vs_second_worst"
    assert metadata["x_column"] == "raw_llr_lexA"
    assert metadata["y_column"] == "raw_llr_cpxR"


def test_chain_trajectory_scatter_all_pairs_grid_mode(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep_idx": [0, 1, 0, 1],
            "raw_llr_lexA": [0.10, 0.60, 0.20, 0.40],
            "raw_llr_cpxR": [0.30, 0.50, 0.10, 0.80],
            "raw_llr_fur": [0.20, 0.70, 0.40, 0.30],
            "norm_llr_lexA": [0.05, 0.30, 0.10, 0.20],
            "norm_llr_cpxR": [0.15, 0.25, 0.05, 0.40],
            "norm_llr_fur": [0.10, 0.35, 0.20, 0.15],
            "objective_scalar": [0.10, 0.50, 0.10, 0.30],
            "raw_llr_objective": [0.10, 0.50, 0.10, 0.30],
            "sequence": ["s0", "s1", "s2", "s3"],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.01, 0.02, 0.03],
            "raw_llr_cpxR": [0.04, 0.03, 0.02],
            "raw_llr_fur": [0.02, 0.01, 0.04],
            "norm_llr_lexA": [0.01, 0.02, 0.03],
            "norm_llr_cpxR": [0.04, 0.03, 0.02],
            "norm_llr_fur": [0.02, 0.01, 0.04],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["s1", "s3"],
            "raw_llr_lexA": [0.60, 0.40],
            "raw_llr_cpxR": [0.50, 0.80],
            "raw_llr_fur": [0.70, 0.30],
            "norm_llr_lexA": [0.30, 0.20],
            "norm_llr_cpxR": [0.25, 0.40],
            "norm_llr_fur": [0.35, 0.15],
        }
    )
    out_path = tmp_path / "chain_trajectory_scatter_grid.png"
    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="llr",
        consensus_anchors=None,
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        score_space_mode="all_pairs_grid",
        tf_names=["lexA", "cpxR", "fur"],
        tf_pairs_grid=[("lexA", "cpxR"), ("lexA", "fur"), ("cpxR", "fur")],
    )

    assert out_path.exists()
    assert metadata["score_space_mode"] == "all_pairs_grid"
    assert metadata["panel_count"] == 3
    assert metadata["grid_shared_axes"] is False
    assert metadata["grid_label_mode"] == "edge_only"


def test_chain_trajectory_scatter_all_pairs_grid_normalized_shares_axes(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "chain": [0, 0, 1, 1],
            "sweep_idx": [0, 1, 0, 1],
            "raw_llr_lexA": [0.10, 0.60, 0.20, 0.40],
            "raw_llr_cpxR": [0.30, 0.50, 0.10, 0.80],
            "raw_llr_fur": [0.20, 0.70, 0.40, 0.30],
            "norm_llr_lexA": [0.05, 0.30, 0.10, 0.20],
            "norm_llr_cpxR": [0.15, 0.25, 0.05, 0.40],
            "norm_llr_fur": [0.10, 0.35, 0.20, 0.15],
            "objective_scalar": [0.10, 0.50, 0.10, 0.30],
            "raw_llr_objective": [0.10, 0.50, 0.10, 0.30],
            "sequence": ["s0", "s1", "s2", "s3"],
        }
    )
    baseline_df = pd.DataFrame(
        {
            "raw_llr_lexA": [0.01, 0.02, 0.03],
            "raw_llr_cpxR": [0.04, 0.03, 0.02],
            "raw_llr_fur": [0.02, 0.01, 0.04],
            "norm_llr_lexA": [0.01, 0.02, 0.03],
            "norm_llr_cpxR": [0.04, 0.03, 0.02],
            "norm_llr_fur": [0.02, 0.01, 0.04],
        }
    )
    elites_df = pd.DataFrame(
        {
            "sequence": ["s1", "s3"],
            "raw_llr_lexA": [0.60, 0.40],
            "raw_llr_cpxR": [0.50, 0.80],
            "raw_llr_fur": [0.70, 0.30],
            "norm_llr_lexA": [0.30, 0.20],
            "norm_llr_cpxR": [0.25, 0.40],
            "norm_llr_fur": [0.35, 0.15],
        }
    )
    out_path = tmp_path / "chain_trajectory_scatter_grid_norm.png"
    metadata = plot_elite_score_space_context(
        trajectory_df=trajectory_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_pair=("lexA", "cpxR"),
        scatter_scale="normalized-llr",
        consensus_anchors=None,
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        score_space_mode="all_pairs_grid",
        tf_names=["lexA", "cpxR", "fur"],
        tf_pairs_grid=[("lexA", "cpxR"), ("lexA", "fur"), ("cpxR", "fur")],
    )

    assert out_path.exists()
    assert metadata["score_space_mode"] == "all_pairs_grid"
    assert metadata["panel_count"] == 3
    assert metadata["grid_shared_axes"] is True
    assert metadata["grid_label_mode"] == "edge_only"
