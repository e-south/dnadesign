"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/analysis/composition.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def composition(
    df: pd.DataFrame, cluster_col: str, group_by: str, out_dir: Path, plots: bool
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cluster_col not in df.columns:
        raise KeyError(f"Cluster column '{cluster_col}' not found.")
    if group_by not in df.columns:
        raise KeyError(f"Group-by column '{group_by}' not found.")
    ct = pd.crosstab(df[cluster_col], df[group_by])
    ct.to_csv(out_dir / "composition_counts.csv")
    props = ct.div(ct.sum(axis=1), axis=0)
    props.to_csv(out_dir / "composition_proportions.csv")
    if plots:
        plt.figure(figsize=(12, 8))
        ax = props.plot(kind="bar", stacked=True, width=0.9)
        ax.set_title("Cluster composition (proportions)")
        ax.set_ylabel("Proportion")
        ax.set_xlabel(cluster_col)
        plt.tight_layout()
        fname = f"composition_proportions__{cluster_col}__by_{group_by}.png"
        plt.savefig(out_dir / fname, dpi=300)
        plt.close()
        plt.close()
    return {
        "counts_path": str(out_dir / "composition_counts.csv"),
        "proportions_path": str(out_dir / "composition_proportions.csv"),
    }
