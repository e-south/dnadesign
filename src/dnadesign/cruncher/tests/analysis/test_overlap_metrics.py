"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_overlap_metrics.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import pandas as pd

from dnadesign.cruncher.analysis.overlap import compute_overlap_tables


def test_overlap_summary_metrics() -> None:
    elites_df = pd.DataFrame(
        {
            "id": ["elite-1", "elite-2"],
            "sequence": ["AAAA", "CCCC"],
        }
    )
    hits_df = pd.DataFrame(
        {
            "elite_id": ["elite-1", "elite-1", "elite-2", "elite-2"],
            "tf": ["tfA", "tfB", "tfA", "tfB"],
            "best_start": [0, 2, 0, 5],
            "best_strand": ["+", "-", "+", "-"],
            "pwm_width": [4, 4, 4, 4],
            "best_core_seq": ["AAAA", "TTTT", "CCCC", "GGGG"],
        }
    )
    summary_df, elite_df, summary = compute_overlap_tables(elites_df, hits_df, ["tfA", "tfB"])
    assert summary_df.loc[0, "overlap_rate"] == 0.5
    assert summary_df.loc[0, "overlap_bp_mean"] == 2.0
    hist_payload = summary_df.loc[0, "overlap_bp_hist"]
    assert hist_payload is not None
    decoded = json.loads(hist_payload)
    assert "counts" in decoded and "bins" in decoded
    assert elite_df["overlap_total_bp"].tolist() == [2, 0]
    assert summary["overlap_total_bp_median"] == 1.0
