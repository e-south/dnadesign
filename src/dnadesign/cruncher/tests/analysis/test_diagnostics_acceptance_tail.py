"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_diagnostics_acceptance_tail.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.diagnostics import (
    _tail_move_window_records,
    summarize_sampling_diagnostics,
)


def test_acceptance_tail_from_move_stats() -> None:
    move_stats = [
        {
            "sweep_idx": 0,
            "phase": "draw",
            "chain": 0,
            "move_kind": "B",
            "attempted": 1,
            "accepted": 1,
            "delta": 0.1,
            "delta_hamming": 2,
        },
        {
            "sweep_idx": 1,
            "phase": "draw",
            "chain": 0,
            "move_kind": "B",
            "attempted": 1,
            "accepted": 0,
            "delta": -0.2,
            "delta_hamming": 0,
        },
        {
            "sweep_idx": 1,
            "phase": "draw",
            "chain": 0,
            "move_kind": "M",
            "attempted": 1,
            "accepted": 1,
            "delta": -0.1,
            "delta_hamming": 3,
        },
        {
            "sweep_idx": 1,
            "phase": "draw",
            "chain": 0,
            "move_kind": "S",
            "attempted": 1,
            "accepted": 1,
            "delta": 0.0,
            "delta_hamming": 1,
            "gibbs_changed": True,
        },
        {
            "sweep_idx": 2,
            "phase": "draw",
            "chain": 0,
            "move_kind": "S",
            "attempted": 1,
            "accepted": 1,
            "delta": 0.0,
            "delta_hamming": 0,
            "gibbs_changed": False,
        },
    ]
    diagnostics = summarize_sampling_diagnostics(
        trace_idata=None,
        sequences_df=pd.DataFrame({"sequence": []}),
        elites_df=pd.DataFrame(),
        elites_hits_df=None,
        tf_names=["tfA"],
        optimizer={"kind": "gibbs_anneal"},
        optimizer_stats={"move_stats": move_stats},
        optimizer_kind="gibbs_anneal",
    )
    optimizer_metrics = diagnostics["metrics"]["optimizer"]
    assert optimizer_metrics["acceptance_rate_non_s_tail"] == 2.0 / 3.0
    assert optimizer_metrics["acceptance_tail_B"] == 0.5
    assert optimizer_metrics["acceptance_tail_M"] == 1.0
    assert optimizer_metrics["acceptance_tail_mh_all"] == 2.0 / 3.0
    assert optimizer_metrics["acceptance_tail_rugged"] == 2.0 / 3.0
    assert optimizer_metrics["downhill_accept_tail_rugged"] == 0.5
    assert optimizer_metrics["gibbs_flip_rate_tail"] == 0.5
    assert optimizer_metrics["tail_step_hamming_mean"] == 1.2


def test_tail_move_window_records_ignores_non_dict_rows() -> None:
    move_stats = [
        {
            "sweep_idx": 0,
            "phase": "draw",
            "move_kind": "B",
            "attempted": 1,
            "accepted": 1,
        },
        "bad-row",
        {
            "sweep_idx": 1,
            "phase": "draw",
            "move_kind": "B",
            "attempted": 1,
            "accepted": 0,
        },
    ]
    tail_rows, window, total_sweeps = _tail_move_window_records(move_stats, min_window=1, max_window=10)
    assert len(tail_rows) == 1
    assert int(tail_rows[0]["sweep_idx"]) == 1
    assert window == 1
    assert total_sweeps == 2


def test_tail_move_window_records_requires_phase_column_when_filtering() -> None:
    move_stats = [{"sweep_idx": 0, "move_kind": "B", "attempted": 1, "accepted": 1}]
    tail_rows, window, total_sweeps = _tail_move_window_records(move_stats, phase="draw")
    assert tail_rows == []
    assert window is None
    assert total_sweeps is None


def test_tail_move_window_records_ignores_negative_counts() -> None:
    move_stats = [
        {"sweep_idx": 0, "phase": "draw", "move_kind": "B", "attempted": 1, "accepted": 1},
        {"sweep_idx": 1, "phase": "draw", "move_kind": "B", "attempted": -4, "accepted": 0},
    ]
    tail_rows, window, total_sweeps = _tail_move_window_records(move_stats, min_window=1, max_window=10)
    assert len(tail_rows) == 1
    assert int(tail_rows[0]["sweep_idx"]) == 0
    assert window == 1
    assert total_sweeps == 1


def test_tail_move_window_records_ignores_rows_with_accepted_gt_attempted() -> None:
    move_stats = [
        {"sweep_idx": 0, "phase": "draw", "move_kind": "B", "attempted": 1, "accepted": 1},
        {"sweep_idx": 1, "phase": "draw", "move_kind": "B", "attempted": 2, "accepted": 3},
    ]
    tail_rows, window, total_sweeps = _tail_move_window_records(move_stats, min_window=1, max_window=10)
    assert len(tail_rows) == 1
    assert int(tail_rows[0]["sweep_idx"]) == 0
    assert window == 1
    assert total_sweeps == 1
