# ABOUTME: Tests SFXI diagnostic helper utilities.
# ABOUTME: Covers setpoint extraction from run metadata.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_sfxi_diag_data.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import polars as pl

from dnadesign.opal.src.plots.sfxi_diag_data import parse_setpoint_from_runs


def test_parse_setpoint_from_runs_accepts_setpoint_vector():
    runs_df = pl.DataFrame(
        {
            "objective__params": [
                {"setpoint_vector": [0.0, 0.0, 0.0, 1.0]},
            ]
        }
    )
    result = parse_setpoint_from_runs(runs_df)
    assert result.tolist() == [0.0, 0.0, 0.0, 1.0]
