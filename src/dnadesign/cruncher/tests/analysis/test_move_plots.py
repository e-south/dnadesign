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
            {"sweep_idx": 0, "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
            {"sweep_idx": 1, "chain": 0, "move_kind": "B", "attempted": 1, "accepted": 0},
            {"sweep_idx": 2, "chain": 0, "move_kind": "B", "attempted": 1, "accepted": 1},
            {"sweep_idx": 2, "chain": 1, "move_kind": "M", "attempted": 1, "accepted": 0},
            {"sweep_idx": 3, "chain": 1, "move_kind": "M", "attempted": 1, "accepted": 1},
        ],
        "mcmc_cooling": {"kind": "piecewise", "stages": [{"sweeps": 2, "beta": 0.2}, {"sweeps": 4, "beta": 1.0}]},
    }
    out_path = tmp_path / "plot__health__panel.png"
    metadata = plot_health_panel(optimizer_stats, out_path, dpi=150, png_compress_level=9)
    assert out_path.exists()
    assert metadata["has_mh_windows"] is True
    assert metadata["move_kinds_present"] == ["B", "M", "S"]


def test_health_panel_handles_s_only_without_false_acceptance_signal(tmp_path) -> None:
    optimizer_stats = {
        "move_stats": [
            {"sweep_idx": 0, "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
            {"sweep_idx": 1, "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
            {"sweep_idx": 2, "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
        ]
    }
    out_path = tmp_path / "plot__health__panel_s_only.png"
    metadata = plot_health_panel(optimizer_stats, out_path, dpi=150, png_compress_level=9)
    assert out_path.exists()
    assert metadata["has_mh_windows"] is False
