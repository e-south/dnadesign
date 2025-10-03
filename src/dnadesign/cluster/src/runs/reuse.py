"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/runs/reuse.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .index import list_runs


def find_equivalent_fit(
    input_sig_hash: str, algo_sig_hash: str, root: Path | None = None
) -> dict | None:
    df = list_runs(root=root)
    if df.empty:
        return None
    m = (
        (df["kind"] == "fit")
        & (df["input_sig_hash"] == input_sig_hash)
        & (df["algo_params"].apply(lambda p: isinstance(p, dict)))
        & (df["algo"] == "leiden")
    )
    # Further filter on algo_sig hash if stored; otherwise approximate by same params
    cand = df[m]
    if cand.empty:
        return None
    # Return the most recent matching
    return cand.iloc[0].to_dict()


def can_reattach(existing_cols_meta_sig: str | None, desired_sig: str) -> bool:
    return existing_cols_meta_sig == desired_sig if existing_cols_meta_sig else False
