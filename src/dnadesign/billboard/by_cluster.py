"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/by_cluster.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dnadesign.billboard.core import compute_core_metrics, process_sequences

logger = logging.getLogger(__name__)
sns.set_theme(style="ticks", font_scale=0.8)

def compute_cluster_metrics(results, cfg):
    """
    Compute core metrics per Leiden cluster by re‐processing each subset of sequences.
    Logs at DEBUG level to avoid console spam.
    """
    logger.debug("Computing cluster‐level metrics")
    clusters = {}
    for s in results["sequences"]:
        cid = s.get("meta_cluster_count")
        if cid is not None:
            clusters.setdefault(cid, []).append(s)

    records = []
    for cid in tqdm(sorted(clusters), desc="Clusters"):
        seqs = clusters[cid]
        cluster_results = process_sequences(seqs, cfg)
        cms = compute_core_metrics(cluster_results, cfg)
        cms["meta_cluster_count"] = cid
        records.append(cms)

    df = pd.DataFrame(records)
    logger.debug(f"Computed metrics for {len(df)} clusters")
    return df


def save_cluster_characterization_scatter(df, cfg, path, dpi=600, figsize=(10,6)):
    logger.info(f"Saving cluster characterization scatter to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    shapes = cfg["characterize_by_leiden_cluster"]["marker_shapes"]
    title  = cfg["characterize_by_leiden_cluster"]["plot_title"]

    plt.figure(figsize=figsize)
    for col in df.columns:
        if col == "meta_cluster_count":
            continue
        plt.scatter(
            df["meta_cluster_count"],
            df[col],
            label=col,
            marker=shapes.get(col, "o"),
            alpha=0.8
        )
    plt.xlabel("Cluster ID", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    logger.info("Cluster characterization saved")