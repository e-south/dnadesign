"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_move_plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.plots.moves import plot_move_acceptance_time, plot_move_usage_time


def test_move_plots_with_chains(tmp_path) -> None:
    df = pd.DataFrame(
        [
            {"sweep_idx": 0, "phase": "tune", "chain": 0, "move_kind": "B", "attempted": 1, "accepted": 1},
            {"sweep_idx": 1, "phase": "draw", "chain": 0, "move_kind": "B", "attempted": 1, "accepted": 0},
            {"sweep_idx": 0, "phase": "tune", "chain": 1, "move_kind": "B", "attempted": 1, "accepted": 1},
            {"sweep_idx": 1, "phase": "draw", "chain": 1, "move_kind": "B", "attempted": 1, "accepted": 1},
        ]
    )
    acceptance_path = tmp_path / "plot__move_acceptance_time.png"
    usage_path = tmp_path / "plot__move_usage_time.png"
    plot_move_acceptance_time(df, acceptance_path, dpi=150, png_compress_level=9)
    plot_move_usage_time(df, usage_path, dpi=150, png_compress_level=9)
    assert acceptance_path.exists()
    assert usage_path.exists()
