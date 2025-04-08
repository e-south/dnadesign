"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/by_cluster.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Progress bar support

# Import the core metrics computation from the core module.
from dnadesign.billboard.core import compute_core_metrics

def compute_cluster_metrics(results, config):
    """
    Break the sequences down by their pre-computed 'meta_cluster_count' and, for each cluster,
    compute the core diversity metrics (using the current billboard logic).
    
    Returns a DataFrame where each row corresponds to a unique cluster and contains:
      - meta_cluster_count (the cluster ID)
      - Aggregated core metrics (computed as the mean value over all sequences in the cluster)
    """
    # Group sequences by cluster.
    clusters = {}
    for seq in results.get("sequences", []):
        if "meta_cluster_count" not in seq:
            continue  # Skip sequences lacking clustering information.
        cluster_id = seq["meta_cluster_count"]
        clusters.setdefault(cluster_id, []).append(seq)
    
    cluster_metrics_list = []
    # Iterate over each cluster with a progress bar.
    for cluster_id in tqdm(sorted(clusters.keys()), desc="Computing metrics per cluster"):
        cluster_seqs = clusters[cluster_id]
        sub_results = {"sequences": cluster_seqs}
        core_metrics = compute_core_metrics(sub_results, config)
        record = {"meta_cluster_count": cluster_id}
        record.update(core_metrics)
        cluster_metrics_list.append(record)
    
    return pd.DataFrame(cluster_metrics_list)

def save_cluster_characterization_scatter(cluster_df, config, output_path, dpi=600, figsize=(10,6)):
    """
    Generate and save a scatter plot for cluster-level characterization.
    
    The provided DataFrame (cluster_df) must have one row per cluster (meta_cluster_count)
    with aggregated core metrics for that cluster. Each core metric is then plotted
    (with distinct marker shapes as specified in the configuration) versus the cluster ID.
    """
    if cluster_df.empty:
        print("No cluster metrics available for cluster characterization.")
        return

    marker_shapes = config.get("characterize_by_leiden_cluster", {}).get("marker_shapes", {})
    plot_title = config.get("characterize_by_leiden_cluster", {}).get("plot_title", "Cluster Characterization by Core Metrics")
    
    # Identify the metrics to plot (exclude the cluster id).
    metrics_to_plot = [col for col in cluster_df.columns if col != "meta_cluster_count"]
    
    plt.figure(figsize=figsize)
    for metric in metrics_to_plot:
        marker = marker_shapes.get(metric, "o")
        plt.scatter(cluster_df["meta_cluster_count"], cluster_df[metric], label=metric, marker=marker, alpha=0.8)
    plt.xlabel("Cluster ID (meta_cluster_count)")
    plt.ylabel("Aggregated Core Metric Value")
    plt.title(plot_title)
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()