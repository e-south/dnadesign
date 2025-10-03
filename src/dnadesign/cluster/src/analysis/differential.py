"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/analysis/differential.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def differential(
    df: pd.DataFrame,
    cluster_col: str,
    group_by: str,
    out_dir: Path,
    fold_thresh: float = 1.5,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cluster_col not in df.columns:
        raise KeyError(f"Cluster column '{cluster_col}' not found.")
    if group_by not in df.columns:
        raise KeyError(f"Group-by column '{group_by}' not found.")
    overall = df[group_by].value_counts(normalize=True)
    rows = []
    for cl, g in df.groupby(cluster_col):
        freq = g[group_by].value_counts(normalize=True)
        fc = (freq / overall).dropna().sort_values(ascending=False)
        enriched = fc[fc > fold_thresh]
        rows.append({"cluster": cl, "markers": ", ".join(enriched.index.tolist())})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"differential__{cluster_col}__by_{group_by}.csv", index=False)
    return out
