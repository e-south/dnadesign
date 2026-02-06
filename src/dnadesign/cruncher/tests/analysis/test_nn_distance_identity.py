"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_nn_distance_identity.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnadesign.cruncher.analysis.diversity import compute_elites_nn_distance_table
from dnadesign.cruncher.core.pwm import PWM


def _pwm(name: str, length: int = 4) -> PWM:
    matrix = np.full((length, 4), 0.25)
    return PWM(name=name, matrix=matrix)


def test_nn_distance_ignores_duplicate_identities() -> None:
    hits_df = pd.DataFrame(
        {
            "elite_id": ["e1", "e2", "e3", "e4"],
            "tf": ["tf1", "tf1", "tf1", "tf1"],
            "best_core_seq": ["AAAA", "AAAA", "CCCC", "GGGG"],
        }
    )
    pwms = {"tf1": _pwm("tf1")}
    identity_by_elite_id = {
        "e1": "canonA",
        "e2": "canonA",
        "e3": "canonC",
        "e4": "canonG",
    }
    rank_by_elite_id = {"e1": 1, "e2": 2, "e3": 3, "e4": 4}

    nn_df = compute_elites_nn_distance_table(
        hits_df,
        ["tf1"],
        pwms,
        identity_mode="canonical",
        identity_by_elite_id=identity_by_elite_id,
        rank_by_elite_id=rank_by_elite_id,
    )

    assert len(nn_df) == 4
    row_e1 = nn_df[nn_df["elite_id"] == "e1"].iloc[0]
    row_e2 = nn_df[nn_df["elite_id"] == "e2"].iloc[0]
    assert row_e1["nn_dist"] == row_e2["nn_dist"]
