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
import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.cruncher.analysis.plots import elites_nn_distance as elites_nn_distance_plot
from dnadesign.cruncher.analysis.plots.elites_nn_distance import plot_elites_nn_distance

matplotlib.use("Agg", force=True)


def test_plot_elites_nn_distance_supports_small_k_text_panel(tmp_path: Path) -> None:
    nn_df = pd.DataFrame({"elite_id": ["elite-1"], "nn_dist": [None]})
    elites_df = pd.DataFrame({"id": ["elite-1"], "sequence": ["ACGTACGT"], "combined_score_final": [1.2], "rank": [1]})
    out_path = Path(tmp_path) / "elites__nn_distance.png"
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
    out_path = Path(tmp_path) / "elites__nn_distance__constant.png"
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


def test_plot_elites_nn_distance_uses_integer_hamming_ticks_and_jittered_overlap_annotations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    nn_df = pd.DataFrame(
        {
            "elite_id": ["elite-1", "elite-2", "elite-3", "elite-4"],
            "nn_dist": [0.0, 0.0, 0.0, 0.0],
        }
    )
    elites_df = pd.DataFrame(
        {
            "id": ["elite-1", "elite-2", "elite-3", "elite-4"],
            "sequence": ["AAAAAA", "AAAAAT", "AAAATT", "AATTTT"],
            "combined_score_final": [1.0, 1.0, 0.9, 0.85],
            "rank": [1, 2, 3, 4],
        }
    )
    out_path = Path(tmp_path) / "elites__nn_distance__ticks_and_jitter.png"

    captured: dict[str, object] = {}
    original_savefig = elites_nn_distance_plot.savefig

    def _capture_savefig(fig, path, *, dpi, png_compress_level):
        ax_scatter = fig.axes[0]
        captured["xticks"] = [float(v) for v in ax_scatter.get_xticks().tolist()]
        captured["annotation_positions"] = [tuple(text.get_position()) for text in ax_scatter.texts]
        return original_savefig(fig, path, dpi=dpi, png_compress_level=png_compress_level)

    monkeypatch.setattr(elites_nn_distance_plot, "savefig", _capture_savefig)
    plot_elites_nn_distance(
        nn_df,
        out_path,
        elites_df=elites_df,
        baseline_nn=pd.Series([0.2, 0.3]),
        dpi=120,
        png_compress_level=1,
    )
    plt.close("all")

    assert out_path.exists()
    xticks = captured["xticks"]
    assert xticks
    assert all(float(v).is_integer() for v in xticks)
    annotation_positions = captured["annotation_positions"]
    assert len(annotation_positions) >= 4
    assert len(set(annotation_positions)) > 1


def test_plot_elites_nn_distance_supports_mixed_sequence_lengths(tmp_path: Path) -> None:
    nn_df = pd.DataFrame(
        {
            "elite_id": ["elite-1", "elite-2", "elite-3"],
            "nn_dist": [0.2, 0.3, 0.4],
        }
    )
    elites_df = pd.DataFrame(
        {
            "id": ["elite-1", "elite-2", "elite-3"],
            "sequence": ["AAAAAA", "AAAAA", "TTTTTTT"],
            "combined_score_final": [1.1, 1.0, 0.9],
            "rank": [1, 2, 3],
        }
    )
    out_path = Path(tmp_path) / "elites__nn_distance__mixed_lengths.png"
    metadata = plot_elites_nn_distance(
        nn_df,
        out_path,
        elites_df=elites_df,
        baseline_nn=pd.Series([0.2, 0.3]),
        dpi=120,
        png_compress_level=1,
    )
    assert out_path.exists()
    assert metadata["panel_kind"] == "diversity_panel"
    assert metadata["d_full_median"] is not None
