"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/diagnostics.py

Summarize move statistics for analysis reports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _summarize_move_stats(move_stats: list[dict[str, object]]) -> pd.DataFrame:
    if not move_stats:
        return pd.DataFrame()
    df = pd.DataFrame(move_stats)
    required = {"move_kind", "attempted", "accepted"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    grouped = df.groupby("move_kind", as_index=False)[["attempted", "accepted"]].sum()
    grouped["acceptance_rate"] = grouped["accepted"] / grouped["attempted"].replace(0, np.nan)
    grouped["usage_fraction"] = grouped["attempted"] / grouped["attempted"].sum() if not grouped.empty else 0.0
    return grouped
