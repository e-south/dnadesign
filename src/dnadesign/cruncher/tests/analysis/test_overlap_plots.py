"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_overlap_plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.plots.elites_showcase import plot_elites_showcase
from dnadesign.cruncher.core.pwm import PWM


def test_elites_showcase_plot_smoke(tmp_path) -> None:
    elites_df = pd.DataFrame(
        [
            {"id": "elite-1", "rank": 1, "sequence": "AAAACCCC"},
            {"id": "elite-2", "rank": 2, "sequence": "CCCCAAAA"},
            {"id": "elite-3", "rank": 3, "sequence": "AACCGAAT"},
        ]
    )
    elite_overlap_df = pd.DataFrame(
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
            {"elite_id": "elite-3", "tf": "tfB", "best_start": 4, "pwm_width": 4, "best_strand": "+"},
        ]
    )
    pwms = {
        "tfA": PWM(
            name="tfA",
            matrix=np.array(
                [
                    [0.8, 0.1, 0.05, 0.05],
                    [0.1, 0.8, 0.05, 0.05],
                    [0.05, 0.05, 0.8, 0.1],
                    [0.05, 0.05, 0.1, 0.8],
                ],
                dtype=float,
            ),
        ),
        "tfB": PWM(
            name="tfB",
            matrix=np.array(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.1, 0.1, 0.7, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.7, 0.1, 0.1, 0.1],
                ],
                dtype=float,
            ),
        ),
    }

    panel_path = tmp_path / "plot__elites_showcase.png"
    plot_elites_showcase(
        elites_df=elites_df,
        hits_df=hits_df,
        elite_overlap_df=elite_overlap_df,
        tf_names=["tfA", "tfB"],
        pwms=pwms,
        out_path=panel_path,
        max_panels=12,
        dpi=150,
        png_compress_level=9,
    )
    assert panel_path.exists()


def test_elites_showcase_fails_fast_when_panel_limit_exceeded(tmp_path) -> None:
    elites_df = pd.DataFrame(
        [
            {"id": "elite-1", "rank": 1, "sequence": "AAAACCCC"},
            {"id": "elite-2", "rank": 2, "sequence": "CCCCAAAA"},
        ]
    )
    elite_overlap_df = pd.DataFrame({"id": ["elite-1", "elite-2"], "overlap_total_bp": [7, 7]})
    hits_df = pd.DataFrame(
        [
            {"elite_id": "elite-1", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
            {"elite_id": "elite-2", "tf": "tfA", "best_start": 0, "pwm_width": 4, "best_strand": "+"},
        ]
    )
    pwms = {"tfA": PWM(name="tfA", matrix=np.array([[0.25, 0.25, 0.25, 0.25]] * 4, dtype=float))}

    panel_path = tmp_path / "plot__elites_showcase.png"
    try:
        plot_elites_showcase(
            elites_df=elites_df,
            hits_df=hits_df,
            elite_overlap_df=elite_overlap_df,
            tf_names=["tfA"],
            pwms=pwms,
            out_path=panel_path,
            max_panels=1,
            dpi=150,
            png_compress_level=9,
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "analysis.elites_showcase.max_panels" in str(exc)
    assert raised
