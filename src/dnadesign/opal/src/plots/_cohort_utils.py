"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/_cohort_utils.py

Shared cohort validation helpers for ledger-backed OPAL plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def selected_mask(values: pd.Series, *, allow_null_false: bool = False) -> pd.Series:
    if values.isna().any():
        if not allow_null_false:
            raise ValueError("sel__is_selected contains null values.")
        values = values.fillna(False)
    if not pd.api.types.is_bool_dtype(values):
        bad = values.loc[~values.map(_is_bool_like)]
        if not bad.empty:
            preview = ", ".join(repr(value) for value in bad.head(5).tolist())
            raise ValueError(f"sel__is_selected must be boolean; got {preview}")
    return values.astype(bool)


def positive_ranks(values: pd.Series, *, column: str = "sel__rank_competition") -> pd.Series:
    ranks = pd.to_numeric(values, errors="coerce")
    if ranks.isna().any() or not np.isfinite(ranks.to_numpy(dtype=float)).all():
        raise ValueError(f"{column} contains non-finite values.")
    if (ranks <= 0).any():
        raise ValueError(f"{column} must be positive.")
    return ranks


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))
