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
    elites_df = pd.DataFrame({"id": ["elite-1"], "sequence": ["ACGTACGT"], "combined_score_final": [1.2], "rank": [1]})
    out_path = Path(tmp_path) / "plot__elites__nn_distance.png"
    metadata = plot_elites_nn_distance(
        nn_df,
        out_path,
        elites_df=elites_df,
        baseline_nn=pd.Series([0.2, 0.3]),
        dpi=150,
        png_compress_level=9,
    )
    assert out_path.exists()
    assert metadata["has_finite_distances"] is False
    assert metadata["all_zero_core"] is False
    assert metadata["panel_kind"] == "text"


def test_plot_elites_nn_distance_handles_constant_values(tmp_path: Path) -> None:
    nn_df = pd.DataFrame(
        {
            "elite_id": ["elite-1", "elite-2", "elite-3"],
            "nn_dist": [0.0, 0.0, 0.0],
        }
    )
    elites_df = pd.DataFrame(
        {
            "id": ["elite-1", "elite-2", "elite-3"],
            "sequence": ["AAAAAAAA", "AAAAAAAT", "TTTTTTTT"],
            "combined_score_final": [1.3, 1.2, 1.1],
            "rank": [1, 2, 3],
        }
    )
    out_path = Path(tmp_path) / "plot__elites__nn_distance__constant.png"
    metadata = plot_elites_nn_distance(
        nn_df,
        out_path,
        elites_df=elites_df,
        baseline_nn=pd.Series([0.2, 0.3]),
        dpi=150,
        png_compress_level=9,
    )
    assert out_path.exists()
    assert metadata["has_finite_distances"] is True
    assert metadata["all_zero_core"] is True
    assert metadata["d_full_median"] > 0
    assert metadata["core_zero_but_full_diverse"] is True
    assert metadata["panel_kind"] == "diversity_panel"
