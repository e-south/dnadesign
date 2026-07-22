"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_frame_verification.py

Memory-bounded exact frame comparison for behavior evidence verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd


def assert_frame_equal_by_key(observed: pd.DataFrame, expected: pd.DataFrame, *, keys: list[str]) -> None:
    """Align by unique keys and compare one column at a time."""

    if tuple(observed.columns) != tuple(expected.columns) or len(observed) != len(expected):
        raise AssertionError("frame shape or columns differ")
    observed_keys = pd.MultiIndex.from_frame(observed.loc[:, keys])
    expected_keys = pd.MultiIndex.from_frame(expected.loc[:, keys])
    if not observed_keys.is_unique or not expected_keys.is_unique:
        raise AssertionError("frame comparison keys must be unique")
    expected_positions = expected_keys.get_indexer(observed_keys)
    if (expected_positions < 0).any():
        raise AssertionError("frame comparison keys differ")
    for column in observed.columns:
        pd.testing.assert_series_equal(
            observed[column].reset_index(drop=True),
            expected[column].iloc[expected_positions].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )


__all__ = ["assert_frame_equal_by_key"]
