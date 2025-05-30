"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/clustering/intra_cluster_similarity_analysis.py

This module computes per-sequence intra-cluster similarity using global 
alignment (Needleman–Wunsch) with affine gap penalties and then generates a 
density plot showing the distribution of the similarity scores per cluster.

For each cluster:
    - Group all entries by "leiden_cluster".
    - For clusters with ≥2 sequences, for each sequence compute the average normalized 
      similarity (using aligner.score_pairwise) to every other sequence.
    - For singleton clusters, assign a default similarity of 1.0.
    
Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm

# Import the pairwise scoring function from our aligner package.
from dnadesign.aligner.metrics import score_pairwise


def compute_intra_cluster_similarity(
    data_entries, match=2, mismatch=-1, gap_open=10, gap_extend=1, normalization="max_score"
):
    """
    For each cluster in data_entries, compute per-sequence mean normalized global alignment
    similarity to all other sequences in that cluster.

    If a cluster has only a single sequence, assign a default of 1.0.

    The result is stored in each entry under "meta_intra_cluster_similarity".

    A progress bar shows progress across clusters.
    """

    clusters = defaultdict(list)
    for entry in data_entries:
        cl = entry.get("leiden_cluster")
        if cl is None:
            raise ValueError("Entry missing 'leiden_cluster'. Ensure Leiden clustering is complete.")
        clusters[cl].append(entry)

    # Progress over clusters
    for cl, entries in tqdm.tqdm(clusters.items(), desc="Computing intra-cluster similarity", unit="cluster"):
        n = len(entries)
        if n == 1:
            # Singleton cluster => similarity = 1.0
            entries[0]["meta_intra_cluster_similarity"] = 1.0
            continue

        # Pre-extract sequences
        seqs = [e.get("sequence") for e in entries]
        if any(seq is None for seq in seqs):
            raise ValueError(f"Some entries in cluster {cl} lack 'sequence'.")

        # For each sequence, average alignment similarity vs. others
        for i, entry in enumerate(entries):
            seq_i = seqs[i]
            sim_scores = []
            for j, seq_j in enumerate(seqs):
                if i == j:
                    continue
                norm_sim = score_pairwise(
                    seq_i,
                    seq_j,
                    match=match,
                    mismatch=mismatch,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    normalization=normalization,
                    return_raw=False,
                )
                sim_scores.append(norm_sim)
            entry["meta_intra_cluster_similarity"] = float(np.mean(sim_scores)) if sim_scores else 1.0


def plot_intra_cluster_similarity(data_entries, batch_name, save_path=None):
    """
    Create a box plot of per-sequence intra-cluster similarity.

    - Each cluster is on the x-axis.
    - Clusters are sorted in descending order of mean similarity.
    - White box fill, gray edges.
    - A strip plot behind each box plot with jitter and alpha to show points.

    data_entries must have:
      "leiden_cluster" and "meta_intra_cluster_similarity".

    If save_path is given, the plot is saved there. Otherwise, show interactively.
    """
    import pandas as pd

    # Group similarity scores
    clusters_dict = {}
    for entry in data_entries:
        cl = entry.get("leiden_cluster")
        sim = entry.get("meta_intra_cluster_similarity")
        if sim is not None:
            clusters_dict.setdefault(cl, []).append(sim)

    # Build DataFrame
    records = []
    for cl, sims in clusters_dict.items():
        for s in sims:
            records.append({"cluster": cl, "intra_similarity": s})
    if not records:
        warnings.warn("No intra-cluster similarity scores found for plotting.")
        return
    df = pd.DataFrame(records)

    # Sort clusters by descending average similarity
    cluster_means = df.groupby("cluster")["intra_similarity"].mean().sort_values(ascending=False)
    # Convert cluster to a categorical with that order
    df["cluster"] = pd.Categorical(df["cluster"], categories=cluster_means.index, ordered=True)

    # Prepare figure
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="ticks")

    # 1) Strip plot behind the box plot (with jitter)
    ax = sns.stripplot(
        data=df,
        x="cluster",
        y="intra_similarity",
        order=cluster_means.index,
        color="black",
        alpha=0.3,
        jitter=True,
        zorder=1,
    )

    # 2) Box plot on top (white fill, gray edges, no fliers)
    sns.boxplot(
        data=df,
        x="cluster",
        y="intra_similarity",
        order=cluster_means.index,
        showfliers=False,
        boxprops=dict(facecolor="white", edgecolor="gray"),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
        medianprops=dict(color="gray"),
        width=0.6,
        zorder=2,
        ax=ax,
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Normalized Global Alignment Similarity within Cluster")
    ax.set_title(f"Intra-Cluster Global Alignment Similarity\nBatch: {batch_name}")

    # Potentially rotate x ticks for many clusters
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Intra-cluster similarity box plot saved to: {save_path}")
    else:
        plt.show()
