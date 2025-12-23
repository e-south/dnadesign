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
import seaborn as sns


def composition(df: pd.DataFrame, cluster_col: str, group_by: str, out_dir: Path, plots: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cluster_col not in df.columns:
        raise KeyError(f"Cluster column '{cluster_col}' not found.")
    if group_by not in df.columns:
        raise KeyError(f"Group-by column '{group_by}' not found.")
    ct = pd.crosstab(df[cluster_col], df[group_by])
    counts_name = f"composition_counts__{cluster_col}__by_{group_by}.csv"
    ct.to_csv(out_dir / counts_name)
    props = ct.div(ct.sum(axis=1), axis=0)
    props_name = f"composition_proportions__{cluster_col}__by_{group_by}.csv"
    props.to_csv(out_dir / props_name)
    if plots:
        sns.set_theme(style="ticks", palette="colorblind")
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=False)
        props.plot(kind="bar", stacked=True, width=0.9, ax=ax)
        ax.set_title("Cluster composition (proportions)")
        ax.set_ylabel("Proportion")
        ax.set_xlabel(cluster_col)
        # place legend outside on the right; no frame; remove top/right spines
        ax.legend(
            title=str(group_by),
            frameon=False,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            ncol=1,
        )
        sns.despine(ax=ax, top=True, right=True)
        # Leave room for the outside legend and save tightly
        fig.subplots_adjust(right=0.80)
        fname = f"composition_proportions__{cluster_col}__by_{group_by}.png"
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return {
        "counts_path": str(out_dir / counts_name),
        "proportions_path": str(out_dir / props_name),
    }
