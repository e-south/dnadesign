"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_move_plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.analysis.plots.health_panel import plot_health_panel


def test_health_panel_plot(tmp_path) -> None:
    optimizer_stats = {
        "move_stats": [
            {"sweep_idx": 0, "attempted": 10, "accepted": 4},
            {"sweep_idx": 1, "attempted": 10, "accepted": 5},
            {"sweep_idx": 2, "attempted": 10, "accepted": 6},
        ],
    }
    out_path = tmp_path / "plot__health__panel.png"
    plot_health_panel(optimizer_stats, out_path, dpi=150, png_compress_level=9)
    assert out_path.exists()
