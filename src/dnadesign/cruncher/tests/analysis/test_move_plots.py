"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_move_plots.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.analysis.plots.health_panel import _move_frame, plot_health_panel


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
    out_path = tmp_path / "health__panel.png"
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
    out_path = tmp_path / "health__panel_s_only.png"
    metadata = plot_health_panel(optimizer_stats, out_path, dpi=150, png_compress_level=9)
    assert out_path.exists()
    assert metadata["has_mh_windows"] is False


def test_move_frame_ignores_non_dict_and_non_finite_rows() -> None:
    optimizer_stats = {
        "move_stats": [
            {"sweep_idx": 0, "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
            "bad-row",
            {"sweep_idx": float("nan"), "chain": 0, "move_kind": "B", "attempted": 1, "accepted": 1},
            {"sweep_idx": 1, "chain": 1, "move_kind": "B", "attempted": float("nan"), "accepted": 1},
            {"sweep_idx": 2, "move_kind": "M", "attempted": 2, "accepted": 1},
        ]
    }
    frame = _move_frame(optimizer_stats)
    assert len(frame) == 2
    assert set(frame["sweep_idx"].tolist()) == {0, 2}
    chain_by_sweep = dict(zip(frame["sweep_idx"].tolist(), frame["chain"].tolist(), strict=False))
    assert chain_by_sweep[0] == 0
    assert chain_by_sweep[2] == 0


def test_move_frame_ignores_negative_attempted_rows() -> None:
    optimizer_stats = {
        "move_stats": [
            {"sweep_idx": 0, "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
            {"sweep_idx": 1, "chain": 1, "move_kind": "B", "attempted": -2, "accepted": 0},
        ]
    }
    frame = _move_frame(optimizer_stats)
    assert len(frame) == 1
    assert int(frame.iloc[0]["sweep_idx"]) == 0


def test_move_frame_ignores_rows_with_accepted_gt_attempted() -> None:
    optimizer_stats = {
        "move_stats": [
            {"sweep_idx": 0, "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
            {"sweep_idx": 1, "chain": 1, "move_kind": "B", "attempted": 2, "accepted": 3},
        ]
    }
    frame = _move_frame(optimizer_stats)
    assert len(frame) == 1
    assert int(frame.iloc[0]["sweep_idx"]) == 0
