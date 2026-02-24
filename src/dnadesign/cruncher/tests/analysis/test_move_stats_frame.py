"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_move_stats_frame.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.move_stats import move_stats_frame


def test_move_stats_frame_filters_invalid_rows_and_normalizes_types() -> None:
    frame = move_stats_frame(
        [
            {"sweep_idx": 0, "chain": 0, "phase": "draw", "move_kind": "S", "attempted": 1, "accepted": 1},
            {"sweep_idx": 1, "chain": None, "phase": "draw", "move_kind": "B", "attempted": 2, "accepted": 1},
            {"sweep_idx": 2, "chain": 1, "phase": "draw", "move_kind": "M", "attempted": 2, "accepted": 3},
            {"sweep_idx": 3, "chain": 1, "phase": "draw", "move_kind": "M", "attempted": -1, "accepted": 0},
            {"sweep_idx": float("nan"), "chain": 1, "phase": "draw", "move_kind": "M", "attempted": 1, "accepted": 1},
            "invalid",
        ]
    )
    assert len(frame) == 2
    assert frame["sweep_idx"].tolist() == [0, 1]
    assert frame["chain"].tolist() == [0, 0]
    assert frame["attempted"].tolist() == [1, 2]
    assert frame["accepted"].tolist() == [1, 1]
    assert frame["move_kind"].tolist() == ["S", "B"]


def test_move_stats_frame_phase_filter_requires_phase_column() -> None:
    frame = move_stats_frame(
        [
            {"sweep_idx": 0, "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
        ],
        phase="draw",
    )
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
