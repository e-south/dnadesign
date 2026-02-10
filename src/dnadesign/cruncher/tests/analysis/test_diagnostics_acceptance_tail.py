"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_diagnostics_acceptance_tail.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics


def test_acceptance_tail_from_move_stats() -> None:
    move_stats = [
        {"sweep_idx": 0, "phase": "draw", "chain": 0, "move_kind": "B", "attempted": 1, "accepted": 1},
        {"sweep_idx": 1, "phase": "draw", "chain": 0, "move_kind": "B", "attempted": 1, "accepted": 0},
        {"sweep_idx": 1, "phase": "draw", "chain": 0, "move_kind": "S", "attempted": 1, "accepted": 1},
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
    assert optimizer_metrics["acceptance_rate_mh_tail"] == 0.5
