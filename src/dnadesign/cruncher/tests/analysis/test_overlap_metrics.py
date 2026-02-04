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


def _per_tf_json(offset_a: int, offset_b: int, width: int = 4) -> str:
    payload = {
        "tfA": {"offset": offset_a, "strand": "+", "width": width},
        "tfB": {"offset": offset_b, "strand": "-", "width": width},
    }
    return json.dumps(payload)


def test_overlap_summary_metrics() -> None:
    elites_df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC"],
            "per_tf_json": [_per_tf_json(0, 2), _per_tf_json(0, 5)],
        }
    )
    summary_df, elite_df, summary = compute_overlap_tables(elites_df, ["tfA", "tfB"])
    assert summary_df.loc[0, "overlap_rate"] == 0.5
    assert summary_df.loc[0, "overlap_bp_mean"] == 2.0
    hist_payload = summary_df.loc[0, "overlap_bp_hist"]
    assert hist_payload is not None
    decoded = json.loads(hist_payload)
    assert "counts" in decoded and "bins" in decoded
    assert elite_df["overlap_total_bp"].tolist() == [2, 0]
    assert summary["overlap_total_bp_median"] == 1.0
