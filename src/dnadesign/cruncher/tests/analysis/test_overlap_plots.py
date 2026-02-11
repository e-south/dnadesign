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
    elites_df = pd.DataFrame(
        [
            {"id": "elite-1", "rank": 1, "sequence": "AAAACCCC"},
            {"id": "elite-2", "rank": 2, "sequence": "CCCCAAAA"},
            {"id": "elite-3", "rank": 3, "sequence": "AACCGAAT"},
        ]
    )
    summary_df = pd.DataFrame(
        [
            {"tf_i": "tfA", "tf_j": "tfB", "overlap_rate": 0.5},
        ]
    )
    elite_df = pd.DataFrame(
        {
            "id": ["elite-1", "elite-2", "elite-3"],
            "rank": [1, 2, 3],
            "overlap_total_bp": [0, 2, 4],
        }
    )
    hits_df = pd.DataFrame(
        [
            {"elite_id": "elite-1", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-1", "tf": "tfB", "best_start": 4, "pwm_width": 4, "best_strand": "-"},
            {"elite_id": "elite-2", "tf": "tfA", "best_start": 1, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-2", "tf": "tfB", "best_start": 3, "pwm_width": 4, "best_strand": "-"},
            {"elite_id": "elite-3", "tf": "tfA", "best_start": 2, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-3", "tf": "tfB", "best_start": 5, "pwm_width": 3, "best_strand": "+"},
        ]
    )
    panel_path = tmp_path / "plot__overlap__panel.png"
    plot_overlap_panel(
        summary_df,
        elite_df,
        ["tfA", "tfB"],
        panel_path,
        hits_df=hits_df,
        elites_df=elites_df,
        sequence_length=8,
        dpi=150,
        png_compress_level=9,
    )
    assert panel_path.exists()


def test_overlap_plot_handles_constant_overlap_distribution(tmp_path) -> None:
    elites_df = pd.DataFrame(
        [
            {"id": "elite-1", "rank": 1, "sequence": "AAAACCCC"},
            {"id": "elite-2", "rank": 2, "sequence": "CCCCAAAA"},
            {"id": "elite-3", "rank": 3, "sequence": "AACCGAAT"},
            {"id": "elite-4", "rank": 4, "sequence": "AATTGGCC"},
        ]
    )
    summary_df = pd.DataFrame(
        [
            {"tf_i": "tfA", "tf_j": "tfB", "overlap_rate": 1.0},
        ]
    )
    elite_df = pd.DataFrame(
        {
            "id": ["elite-1", "elite-2", "elite-3", "elite-4"],
            "rank": [1, 2, 3, 4],
            "overlap_total_bp": [7, 7, 7, 7],
        }
    )
    hits_df = pd.DataFrame(
        [
            {"elite_id": "elite-1", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-1", "tf": "tfB", "best_start": 1, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-2", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-2", "tf": "tfB", "best_start": 1, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-3", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-3", "tf": "tfB", "best_start": 1, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-4", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-4", "tf": "tfB", "best_start": 1, "pwm_width": 4, "best_strand": "+"},
        ]
    )
    panel_path = tmp_path / "plot__overlap__panel_constant.png"
    plot_overlap_panel(
        summary_df,
        elite_df,
        ["tfA", "tfB"],
        panel_path,
        hits_df=hits_df,
        elites_df=elites_df,
        sequence_length=8,
        dpi=150,
        png_compress_level=9,
    )
    assert panel_path.exists()
