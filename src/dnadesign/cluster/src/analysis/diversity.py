"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/analysis/diversity.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy


def diversity(
    df: pd.DataFrame, cluster_col: str, group_by: str, out_dir: Path, plots: bool
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cluster_col not in df.columns:
        raise KeyError(f"Cluster column '{cluster_col}' not found.")
    if group_by not in df.columns:
        raise KeyError(f"Group-by column '{group_by}' not found.")
    rows = []
    for cl, g in df.groupby(cluster_col):
        counts = g[group_by].value_counts()
        p = counts / counts.sum()
        shannon = float(entropy(p))
        simpson = float(1 - np.sum(p.values**2))
        rows.append(
            {"cluster": cl, "shannon": shannon, "simpson": simpson, "n": int(len(g))}
        )
    out = pd.DataFrame(rows).sort_values("shannon", ascending=False)
    out.to_csv(out_dir / "diversity.csv", index=False)
    if plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(out))
        bw = 0.35
        ax.bar(x - bw / 2, out["shannon"], width=bw, label="Shannon")
        ax.bar(x + bw / 2, out["simpson"], width=bw, label="Simpson")
        ax.set_xticks(x)
        ax.set_xticklabels(out["cluster"], rotation=90)
        ax.legend()
        ax.set_title("Diversity per cluster")
        fig.tight_layout()
        fname = f"diversity__{cluster_col}__by_{group_by}.png"
        fig.savefig(out_dir / fname, dpi=300)
        plt.close(fig)
    return {"diversity_path": str(out_dir / "diversity.csv")}
