"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_trajectory_score_space_plot_boundaries.py

Characterization tests for trajectory score-space plot orchestration boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.analysis.plots.trajectory_score_space_plot import (
    _build_score_space_metadata,
    _resolve_scatter_columns,
)


def test_resolve_scatter_columns_maps_supported_scales() -> None:
    assert _resolve_scatter_columns(tf_pair=("lexA", "cpxR"), scatter_scale="llr") == (
        "raw_llr_lexA",
        "raw_llr_cpxR",
        "llr",
    )
    assert _resolve_scatter_columns(tf_pair=("lexA", "cpxR"), scatter_scale="normalized-llr") == (
        "norm_llr_lexA",
        "norm_llr_cpxR",
        "normalized-llr",
    )


def test_build_score_space_metadata_carries_contract_fields() -> None:
    metadata = _build_score_space_metadata(
        score_space_mode="pair",
        panel_count=1,
        legend_labels=["Random baseline", "Selected elites (n=2)"],
        x_column="raw_llr_lexA",
        y_column="raw_llr_cpxR",
        retain_elites=True,
        first_stats={
            "total": 2,
            "rendered_points": 2,
            "unique_coordinates": 2,
            "coordinate_collisions": 0,
            "collision_annotation_count": 0,
            "exact_mapped": 2,
            "path_link_count": 0,
            "snapped_to_path_count": 0,
        },
        legend_fontsize=12,
        anchor_annotation_fontsize=11,
        objective_caption="caption",
    )

    assert metadata["mode"] == "elite_score_space_context"
    assert metadata["score_space_mode"] == "pair"
    assert metadata["panel_count"] == 1
    assert metadata["x_column"] == "raw_llr_lexA"
    assert metadata["y_column"] == "raw_llr_cpxR"
    assert metadata["legend_fontsize"] == 12
    assert metadata["anchor_annotation_fontsize"] == 11


def test_legacy_trajectory_plot_module_is_removed() -> None:
    cruncher_root = Path(__file__).resolve().parents[2]
    legacy_path = cruncher_root / "src" / "analysis" / "plots" / "opt_trajectory.py"
    assert not legacy_path.exists()
