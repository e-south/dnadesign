"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_run_meta_summary.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd

from dnadesign.opal.src.summary import select_run_meta, summarize_run_meta


def test_select_run_meta_latest_round():
    df = pd.DataFrame(
        {
            "run_id": ["r0-a", "r1-b"],
            "as_of_round": [0, 1],
            "model__name": ["m0", "m1"],
            "objective__name": ["o0", "o1"],
            "selection__name": ["s0", "s1"],
            "training__y_ops": [[], []],
            "stats__n_train": [10, 11],
            "stats__n_scored": [100, 110],
            "objective__summary_stats": [{}, {}],
        }
    )
    row = select_run_meta(df)
    summary = summarize_run_meta(row)
    assert summary["run_id"] == "r1-b"
    assert summary["as_of_round"] == 1
