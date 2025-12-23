"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/analysis/intra_similarity.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# We depend on dnadesign.aligner.metrics.score_pairwise
def _score_pairwise(
    seq_i: str,
    seq_j: str,
    match=2,
    mismatch=-1,
    gap_open=10,
    gap_extend=1,
    normalization="max_score",
) -> float:
    try:
        from dnadesign.aligner.metrics import score_pairwise
    except Exception as e:
        raise RuntimeError("dnadesign.aligner is required for intra-cluster similarity.") from e
    return score_pairwise(
        seq_i,
        seq_j,
        match=match,
        mismatch=mismatch,
        gap_open=gap_open,
        gap_extend=gap_extend,
        normalization=normalization,
        return_raw=False,
    )


def intra_cluster_similarity(
    df: pd.DataFrame,
    cluster_col: str,
    *,
    match=2,
    mismatch=-1,
    gap_open=10,
    gap_extend=1,
    max_per_cluster=2000,
    sample_if_larger=True,
) -> pd.Series:
    if "sequence" not in df.columns:
        raise KeyError("'sequence' column is required for intra-cluster similarity.")
    out = pd.Series(index=df.index, dtype=float)
    for cl, g in df.groupby(cluster_col):
        idx = g.index.to_list()
        if len(idx) == 1:
            out.loc[idx] = 1.0
            continue
        if len(idx) > max_per_cluster and sample_if_larger:
            idx = np.random.choice(idx, size=max_per_cluster, replace=False).tolist()
            g = df.loc[idx]
        seqs = g["sequence"].astype(str).to_list()
        # For each i, average score to all j != i
        for i_pos, i_idx in enumerate(g.index):
            si = seqs[i_pos]
            scores = []
            for j_pos, j_idx in enumerate(g.index):
                if i_pos == j_pos:
                    continue
                sj = seqs[j_pos]
                scores.append(
                    _score_pairwise(
                        si,
                        sj,
                        match=match,
                        mismatch=mismatch,
                        gap_open=gap_open,
                        gap_extend=gap_extend,
                    )
                )
            out.loc[i_idx] = float(np.mean(scores)) if scores else 1.0
    return out


def plot_intra_similarity(df: pd.DataFrame, cluster_col: str, out_path: Path | None = None):
    import seaborn as sns

    col = f"{cluster_col}__intra_sim" if not cluster_col.endswith("__intra_sim") else cluster_col
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found.")
    d = df[[cluster_col, col]].copy()
    d.columns = ["cluster", "sim"]
    order = d.groupby("cluster")["sim"].mean().sort_values(ascending=False).index
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=d, x="cluster", y="sim", order=order, showfliers=False)
    sns.stripplot(data=d, x="cluster", y="sim", order=order, color="0.35", alpha=0.3, jitter=True)
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine(top=True, right=True)
    if out_path:
        plt.savefig(out_path, dpi=200)
    else:
        plt.show()
    plt.close()
