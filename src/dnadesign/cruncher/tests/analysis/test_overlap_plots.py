"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_overlap_plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.plots.overlap import plot_overlap_panel


def test_overlap_plot_smoke(tmp_path) -> None:
    summary_df = pd.DataFrame(
        [
            {"tf_i": "tfA", "tf_j": "tfB", "overlap_rate": 0.5},
        ]
    )
    elite_df = pd.DataFrame({"overlap_total_bp": [0, 2, 4]})
    panel_path = tmp_path / "plot__overlap__panel.png"
    plot_overlap_panel(summary_df, elite_df, ["tfA", "tfB"], panel_path, dpi=150, png_compress_level=9)
    assert panel_path.exists()
