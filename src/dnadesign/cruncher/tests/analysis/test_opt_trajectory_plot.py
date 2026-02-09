"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_opt_trajectory_plot.py

Validate optimization trajectory raw-LLR particle plot semantics and metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.cruncher.analysis.plots.opt_trajectory import plot_opt_trajectory


def test_opt_trajectory_plot_requires_raw_llr(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "particle_id": [0, 0],
            "sweep_idx": [0, 1],
            "slot_id": [0, 0],
        }
    )
    out_path = tmp_path / "plot__opt_trajectory.png"

    with pytest.raises(ValueError, match="raw_llr"):
        plot_opt_trajectory(
            trajectory_df=trajectory_df,
            out_path=out_path,
            dpi=72,
            png_compress_level=1,
        )


def test_opt_trajectory_plot_renders_particle_raw_llr_by_sweep(tmp_path: Path) -> None:
    trajectory_df = pd.DataFrame(
        {
            "particle_id": [0, 0, 0, 1, 1, 1],
            "sweep_idx": [0, 1, 2, 0, 1, 2],
            "slot_id": [0, 1, 1, 1, 0, 0],
            "raw_llr_objective": [0.40, 0.55, 0.58, 0.32, 0.41, 0.50],
            "phase": ["tune", "draw", "draw", "tune", "draw", "draw"],
        }
    )
    out_path = tmp_path / "plot__opt_trajectory.png"
    metadata = plot_opt_trajectory(
        trajectory_df=trajectory_df,
        out_path=out_path,
        dpi=72,
        png_compress_level=1,
        slot_overlay=True,
    )

    assert out_path.exists()
    legend_labels = metadata["legend_labels"]
    assert any(label.startswith("particle lineage (id=0..") for label in legend_labels)
    assert metadata["particle_count"] == 2
