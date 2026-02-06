"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_elites_nn_distance_plot.py

Validates elite NN-distance plotting edge cases.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from dnadesign.cruncher.analysis.plots.elites_nn_distance import plot_elites_nn_distance

matplotlib.use("Agg", force=True)


def test_plot_elites_nn_distance_supports_small_k_text_panel(tmp_path: Path) -> None:
    nn_df = pd.DataFrame({"elite_id": ["elite-1"], "nn_dist": [None]})
    out_path = Path(tmp_path) / "plot__elites__nn_distance.png"
    plot_elites_nn_distance(
        nn_df,
        out_path,
        baseline_nn=pd.Series([0.2, 0.3]),
        dpi=150,
        png_compress_level=9,
    )
    assert out_path.exists()
