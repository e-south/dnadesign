"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_plot_cohort_utils.py

Regression tests for plot cohort utils OPAL plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.opal.src.plots._cohort_utils import positive_ranks, selected_mask


def test_selected_mask_rejects_string_booleans() -> None:
    with pytest.raises(ValueError, match="sel__is_selected must be boolean"):
        selected_mask(pd.Series(["False", "True"]))


def test_selected_mask_rejects_nulls_by_default() -> None:
    with pytest.raises(ValueError, match="null"):
        selected_mask(pd.Series([True, None]))


def test_positive_ranks_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        positive_ranks(pd.Series([1, 0]))
