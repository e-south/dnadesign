"""
--------------------------------------------------------------------------------
<dnadesign project>
latdna/analysis.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import torch

from dnadesign.latdna import utils, validation, metrics

def get_metric_function(metric_name: str):
    """
    Retrieve a metric function. Supports:
      - Exact metrics defined in METRIC_REGISTRY (e.g., 'cosine', 'euclidean', 'log1p_cosine')
      - A 'log1p_' prefix for dynamically transforming a base metric if needed
        (e.g. 'log1p_euclidean' -> log1p(euclidean)).
    """
    if metric_name in metrics.METRIC_REGISTRY:
        # Directly in the registry, e.g. "log1p_cosine" or "cosine"
        return metrics.METRIC_REGISTRY[metric_name]
    
    elif metric_name.startswith("log1p_"):
        # Dynamically create a log1p version if not already in registry
        base_key = metric_name[len("log1p_"):]
        base_func = metrics.METRIC_REGISTRY.get(base_key)
        if base_func is None:
            raise ValueError(f"Base metric '{base_key}' not found for log1p transformation.")
        
        def log1p_wrapper(lat_vecs: np.ndarray):
            dist = base_func(lat_vecs)
            return np.log1p(dist)
        
        return log1p_wrapper
    
    else:
        raise ValueError(f"Metric '{metric_name}' is not recognized.")

def compute_pairwise_distances(latent_vectors: np.ndarray, metric_func, max_entries: int = 1000):
    """
    Compute pairwise distances using the given metric function.
    If the number of vectors exceeds max_entries, randomly subsample to reduce memory usage.
    Returns:
      dist_array: 1D array of the upper triangle (excluding the diagonal) of the distance matrix
      used_count: the number of sequences actually used (after subsampling if needed)
    """
    n = latent_vectors.shape[0]
    used_count = n
    if n > max_entries:
        indices = np.random.choice(n, max_entries, replace=False)
        latent_vectors = latent_vectors[indices]
        used_count = max_entries
    
    dist_matrix = metric_func(latent_vectors)
    iu = np.triu_indices_from(dist_matrix, k=1)
    dist_array = dist_matrix[iu]
    return dist_array, used_count

def run_analysis_pipeline(config: dict):
    """
    Execute the analysis pipeline to compute latent diversity metrics and generate
    a vertical stack of subplots (one per metric) with overlaid stripplots and boxplots.
    
    Processing is done separately for each input batch:
      - dense_batch and latdna_batch are mandatory.
      - Each provided seq_batch is processed independently.

    If config["plots"]["xtic_labels"] is provided, we use those labels for the x-axis ticks
    (must match the number of groups).
    """
    logging.info("Starting analysis pipeline...")
    
    # Resolve input directories.
    base_sequences_dir = Path(__file__).parent.parent / "sequences"
    analysis_inputs = config.get("analysis_inputs", {})
    dense_batch_name = analysis_inputs.get("dense_batch")
    latdna_batch_name = analysis_inputs.get("latdna_batch")
    
    if not (dense_batch_name and latdna_batch_name):
        raise ValueError("Analysis inputs must include both 'dense_batch' and 'latdna_batch'.")
    
    # Load dense_batch PT file.
    dense_dir = base_sequences_dir / dense_batch_name
    dense_data = utils.read_single_pt_file_from_subdir(dense_dir)
    logging.info(f"Dense batch '{dense_batch_name}' loaded with {len(dense_data)} entries.")
    
    # Load latdna_batch PT file.
    latdna_dir = base_sequences_dir / latdna_batch_name
    latdna_data = utils.read_single_pt_file_from_subdir(latdna_dir)
    logging.info(f"latDNA batch '{latdna_batch_name}' loaded with {len(latdna_data)} entries.")
    
    # Load seq_batch inputs (can be single string or list).
    seq_batches = analysis_inputs.get("seq_batch")
    seq_data_dict = {}
    seq_batches_order = []
    if seq_batches:
        if not isinstance(seq_batches, list):
            seq_batches = [seq_batches]
        for seq_batch_name in seq_batches:
            seq_dir = base_sequences_dir / seq_batch_name
            seq_data = utils.read_single_pt_file_from_subdir(seq_dir)
            logging.info(f"Seq batch '{seq_batch_name}' loaded with {len(seq_data)} entries.")
            seq_data_dict[seq_batch_name] = seq_data
            seq_batches_order.append(seq_batch_name)
    else:
        logging.info("No sequence batch provided.")
    
    # Validate entries in each batch.
    for idx, entry in enumerate(dense_data):
        validation.validate_analysis_entry(entry, idx, dense_batch_name)
    for idx, entry in enumerate(latdna_data):
        validation.validate_analysis_entry(entry, idx, latdna_batch_name)
    for batch_name, data in seq_data_dict.items():
        for idx, entry in enumerate(data):
            validation.validate_analysis_entry(entry, idx, batch_name)
    
    def extract_latent_vectors(data):
        vectors = []
        for entry in data:
            vec = entry["evo2_logits_mean_pooled"]
            if torch.is_tensor(vec):
                if vec.dtype == torch.bfloat16:
                    vec = vec.to(torch.float32)
                vec = vec.numpy().flatten()
            else:
                vec = np.array(vec).flatten()
            vectors.append(vec)
        return np.array(vectors)
    
    # Extract latent vectors from each batch
    dense_latents = extract_latent_vectors(dense_data)
    latdna_latents = extract_latent_vectors(latdna_data)
    
    seq_latents_dict = {}
    for batch_name, data in seq_data_dict.items():
        seq_latents_dict[batch_name] = extract_latent_vectors(data)
    
    # Determine metrics to compute.
    metrics_to_compute = config.get("metrics", ["cosine", "euclidean", "log1p_euclidean"])
    
    # Build analysis groups in the order we want them to appear on x-axis.
    analysis_groups = {}
    ordered_groups = []
    
    # Add dense_batch
    analysis_groups["densebatch"] = dense_latents
    ordered_groups.append("densebatch")
    
    # Add latdna_batch
    group_by = config.get("group_by", "all").lower()
    if group_by == "all":
        analysis_groups["latdnabatch"] = latdna_latents
        ordered_groups.append("latdnabatch")
    elif group_by == "tf":
        tf_groups = {}
        for entry, vec in zip(latdna_data, latdna_latents):
            tf_name = entry.get("transcription_factor", "unknown")
            tf_groups.setdefault(tf_name, []).append(vec)
        for tf in sorted(tf_groups.keys()):
            key = f"latdnabatch_{tf}"
            analysis_groups[key] = np.array(tf_groups[tf])
            ordered_groups.append(key)
    else:
        raise ValueError("Invalid group_by option. Must be 'all' or 'tf'.")
    
    # Add seq_batch groups in config order
    for batch_name in seq_batches_order:
        key = f"seq_batch_{batch_name}"
        analysis_groups[key] = seq_latents_dict[batch_name]
        ordered_groups.append(key)
    
    # Subsampling limit for pairwise distances
    max_entries = config.get("max_entries_for_pairwise", 1000)
    
    # Prepare DataFrame for plotting
    rows = []
    summary_records = []
    
    # Track total pairwise combos across all groups
    total_combos = 0
    
    for metric_name in metrics_to_compute:
        try:
            metric_func = get_metric_function(metric_name)
        except ValueError as e:
            logging.error(str(e))
            continue
        
        for group_name, vectors in analysis_groups.items():
            if vectors.shape[0] < 2:
                logging.warning(f"Group '{group_name}' has <2 entries. Skipping distance computation.")
                continue
            
            dist_array, used_count = compute_pairwise_distances(vectors, metric_func, max_entries)
            
            # Number of combos for this group
            combos_for_group = used_count * (used_count - 1) // 2
            total_combos += combos_for_group
            
            # Summary stats
            summary_records.append({
                "metric": metric_name,
                "group": group_name,
                "statistic": "mean",
                "value": float(dist_array.mean())
            })
            summary_records.append({
                "metric": metric_name,
                "group": group_name,
                "statistic": "std",
                "value": float(dist_array.std())
            })
            
            # Collect the distances in long form
            for d in dist_array:
                rows.append({
                    "metric": metric_name,
                    "group": group_name,
                    "distance": d
                })
    
    df = pd.DataFrame(rows)
    
    # Create output directory
    analysis_base_dir = Path(__file__).parent.parent / "latdna" / "batch_results"
    output_dir = utils.create_output_directory(analysis_base_dir, "latbatch")
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_records)
    csv_output_path = output_dir / "diversity_summary.csv"
    summary_df.to_csv(csv_output_path, index=False)
    logging.info(f"Wrote diversity summary CSV to {csv_output_path}")
    
    # Start plotting
    sns.set_style("ticks")
    
    unique_metrics = df["metric"].unique()
    n_metrics = len(unique_metrics)
    fig, axes = plt.subplots(
        nrows=n_metrics, 
        ncols=1, 
        figsize=(6, 3 * n_metrics), 
        sharex=False
    )
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric_name in enumerate(unique_metrics):
        ax = axes[i]
        sub_df = df[df["metric"] == metric_name].copy()
        
        # Plot stripplot
        sns.stripplot(
            x="group",
            y="distance",
            data=sub_df,
            hue="group",
            dodge=False,
            jitter=True,
            alpha=0.3,
            zorder=1,
            ax=ax,
            legend=False,
            order=ordered_groups
        )
        
        # Plot boxplot on top
        sns.boxplot(
            x="group",
            y="distance",
            data=sub_df,
            hue="group",
            dodge=False,
            showfliers=False,
            linewidth=1,
            boxprops=dict(facecolor="white", alpha=1),
            zorder=2,
            ax=ax,
            legend=False,
            order=ordered_groups
        )
        
        # Remove the x-axis label
        ax.set_xlabel('')
        
        # Custom X-tick labels if provided
        xtic_labels = config.get("plots", {}).get("xtic_labels")
        if xtic_labels:
            if len(xtic_labels) != len(ordered_groups):
                logging.warning(
                    "Length of xtic_labels does not match the number of groups. "
                    "Using default group names."
                )
                new_labels = ordered_groups
            else:
                new_labels = xtic_labels
        else:
            new_labels = ordered_groups
        
        ax.set_xticklabels(new_labels, rotation=45, ha="right")
        sns.despine(ax=ax, top=True, right=True)
        
        ax.set_title(f"Latent Space Distances ({metric_name})", fontsize=12)
        ax.set_ylabel(f"{metric_name} distance", fontsize=10)
    
    # Add a suptitle showing total combos across all groups
    fig.suptitle(
        f"Pairwise intrapopulation distance metrics for ~{total_combos} total pairwise combos",
        fontsize=12
    )
    
    # Adjust spacing to reduce overlap of x-tick labels with subplots
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # Increase vertical space between rows of subplots
    fig.subplots_adjust(hspace=0.6)
    
    plot_output_path = output_dir / "latent_diversity_boxplot.png"
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    logging.info(f"Saved boxplot to {plot_output_path}")
    
    # Save config snapshot
    snapshot_path = output_dir / "analysis_config_snapshot.yaml"
    with snapshot_path.open("w") as f:
        yaml.dump(config, f)
    logging.info(f"Saved analysis config snapshot to {snapshot_path}")