"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_overlap_plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.plots.overlap import (
    plot_overlap_bp_distribution,
    plot_overlap_heatmap,
)


def test_overlap_plot_smoke(tmp_path) -> None:
    summary_df = pd.DataFrame(
        [
            {"tf_i": "tfA", "tf_j": "tfB", "overlap_rate": 0.5},
        ]
    )
    elite_df = pd.DataFrame({"overlap_total_bp": [0, 2, 4]})
    heatmap_path = tmp_path / "plot__overlap_heatmap.png"
    dist_path = tmp_path / "plot__overlap_bp_distribution.png"
    plot_overlap_heatmap(summary_df, ["tfA", "tfB"], heatmap_path, dpi=150, png_compress_level=9)
    plot_overlap_bp_distribution(elite_df, dist_path, dpi=150, png_compress_level=9)
    assert heatmap_path.exists()
    assert dist_path.exists()
