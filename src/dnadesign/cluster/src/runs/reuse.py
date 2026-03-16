"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/runs/reuse.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .contracts import FIT_REUSE_REQUIRED_COLUMNS
from .index import list_runs


def find_equivalent_fit(input_sig_hash: str, method_sig_hash: str, root: Path | None = None) -> dict | None:
    df = list_runs(root=root)
    if df.empty:
        return None
    required = FIT_REUSE_REQUIRED_COLUMNS
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Cluster run index uses a retired schema and cannot be reused. "
            f"Missing columns: {sorted(missing)}. Clear the old results index and rerun."
        )
    m = (
        (df["kind"] == "fit")
        & (df["input_sig_hash"] == input_sig_hash)
        & (df["method_id"].astype(str).str.len() > 0)
        & (df["method_params"].apply(lambda p: isinstance(p, dict)))
    )
    # Further filter on method signature if stored; otherwise approximate by same params.
    cand = df[m]
    if cand.empty:
        return None
    # Return the most recent matching
    return cand.iloc[0].to_dict()


def can_reattach(existing_cols_meta_sig: str | None, desired_sig: str) -> bool:
    return existing_cols_meta_sig == desired_sig if existing_cols_meta_sig else False
