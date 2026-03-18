"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/runs/reuse.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .contracts import FIT_REUSE_REQUIRED_COLUMNS
from .index import list_runs


def find_equivalent_fit(input_sig_hash: str, method_sig_hash: str, root: Path | None = None) -> dict | None:
    required = FIT_REUSE_REQUIRED_COLUMNS
    selected_columns = required.union({"alias", "run_slug", "labels_path"})
    df = list_runs(
        root=root,
        filters={
            "kind": "fit",
            "input_sig_hash": input_sig_hash,
            "method_sig_hash": method_sig_hash,
        },
        columns=selected_columns,
    )
    if df.empty:
        return None
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Cluster run index uses a retired schema and cannot be reused. "
            f"Missing columns: {sorted(missing)}. Clear the old results index and rerun."
        )
    return df.iloc[0].to_dict()


def can_reattach(existing_cols_meta_sig: str | None, desired_sig: str) -> bool:
    return existing_cols_meta_sig == desired_sig if existing_cols_meta_sig else False
