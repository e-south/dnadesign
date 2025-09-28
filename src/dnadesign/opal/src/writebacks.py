"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/writebacks.py

Round write-backs into records.parquet.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd


def _col(slug: str, r: int, tail: str) -> str:
    return f"opal__{slug}__r{int(r)}__{tail}"


def update_latest_cache(
    df, slug: str, latest_round: int, latest_score: dict[str, float]
):
    rcol = f"opal__{slug}__latest_round"
    scol = f"opal__{slug}__latest_score"
    if rcol not in df.columns:
        df[rcol] = pd.Series(dtype="Int64")
    if scol not in df.columns:
        df[scol] = pd.Series(dtype="float64")
    # write latest for ids we scored in this round
    ids = list(latest_score.keys())
    mask = df["id"].astype(str).isin(ids)
    df.loc[mask, rcol] = latest_round
    df.loc[mask, scol] = df.loc[mask, "id"].astype(str).map(latest_score).astype(float)
    return df
