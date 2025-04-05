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

def compute_pairwise_distances(latent_vectors: np.ndarray, metric_func, max_entries: int = 1000) -> np.ndarray:
    """
    Compute pairwise distances using the given metric function.
    If the number of vectors exceeds max_entries, randomly subsample to reduce memory usage.
    Returns a 1D array of the upper-triangle (excluding diagonal) of the distance matrix.
    """
    n = latent_vectors.shape[0]
    if n > max_entries:
        # For reproducibility, you might set a seed here if desired.
        indices = np.random.choice(n, max_entries, replace=False)
        latent_vectors = latent_vectors[indices]
    dist_matrix = metric_func(latent_vectors)
    iu = np.triu_indices_from(dist_matrix, k=1)
    return dist_matrix[iu]

def run_analysis_pipeline(config: dict):
    """
    Execute the analysis pipeline to compute latent diversity metrics and generate
    a vertical stack of Seaborn subplots (one per metric), with stripplot + boxplot.
    """
    logging.info("Starting analysis pipeline...")
    
    # Resolve input directories
    base_sequences_dir = Path(__file__).parent.parent / "sequences"
    analysis_inputs = config.get("analysis_inputs", {})
    dense_batch_name = analysis_inputs.get("dense_batch")
    latdna_batch_name = analysis_inputs.get("latdna_batch")
    
    if not (dense_batch_name and latdna_batch_name):
        raise ValueError("Analysis inputs must include both 'dense_batch' and 'latdna_batch'.")
    
    dense_dir = base_sequences_dir / dense_batch_name
    latdna_dir = base_sequences_dir / latdna_batch_name
    
    # Load PT files
    dense_data = utils.read_single_pt_file_from_subdir(dense_dir)
    latdna_data = utils.read_single_pt_file_from_subdir(latdna_dir)
    
    logging.info(f"Dense batch '{dense_batch_name}' loaded with {len(dense_data)} entries.")
    logging.info(f"latDNA batch '{latdna_batch_name}' loaded with {len(latdna_data)} entries.")
    
    # Validate entries for analysis
    for idx, entry in enumerate(dense_data):
        validation.validate_analysis_entry(entry, idx, dense_batch_name)
    for idx, entry in enumerate(latdna_data):
        validation.validate_analysis_entry(entry, idx, latdna_batch_name)
    
    # Extract latent vectors
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
    
    dense_latents = extract_latent_vectors(dense_data)
    latdna_latents = extract_latent_vectors(latdna_data)
    
    # Determine which metrics to compute
    metrics_to_compute = config.get("metrics", ["cosine", "euclidean", "log1p_euclidean"])
    
    # Grouping for latDNA batch
    group_by = config.get("group_by", "all").lower()  # "all" or "tf"
    analysis_groups = {}
    analysis_groups["densebatch"] = dense_latents
    if group_by == "all":
        analysis_groups["latdnabatch"] = latdna_latents
    elif group_by == "tf":
        groups = {}
        for entry, vec in zip(latdna_data, latdna_latents):
            tf_name = entry.get("transcription_factor", "unknown")
            groups.setdefault(tf_name, []).append(vec)
        for tf in groups:
            analysis_groups[tf] = np.array(groups[tf])
    else:
        raise ValueError("Invalid group_by option. Must be 'all' or 'tf'.")
    
    # Subsampling limit for pairwise distances
    max_entries = config.get("max_entries_for_pairwise", 1000)
    
    # Prepare long-form DataFrame
    rows = []
    summary_records = []
    for metric_name in metrics_to_compute:
        metric_func = metrics.METRIC_REGISTRY.get(metric_name)
        if not metric_func:
            raise ValueError(f"Metric '{metric_name}' is not recognized.")
        
        for group_name, vectors in analysis_groups.items():
            if vectors.shape[0] < 2:
                logging.warning(f"Group '{group_name}' has less than 2 entries. Skipping distance computation.")
                continue
            distances = compute_pairwise_distances(vectors, metric_func, max_entries)
            
            # Summary stats
            summary_records.append({
                "metric": metric_name,
                "group": group_name,
                "statistic": "mean",
                "value": float(distances.mean())
            })
            summary_records.append({
                "metric": metric_name,
                "group": group_name,
                "statistic": "std",
                "value": float(distances.std())
            })
            
            # Rows for DataFrame
            for d in distances:
                rows.append({
                    "metric": metric_name,
                    "group": group_name,
                    "distance": d
                })
    
    df = pd.DataFrame(rows)
    
    # Create output directory for analysis results
    analysis_base_dir = Path(__file__).parent.parent / "latdna" / "batch_results"
    output_dir = utils.create_output_directory(analysis_base_dir, "latbatch")
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_records)
    csv_output_path = output_dir / "diversity_summary.csv"
    summary_df.to_csv(csv_output_path, index=False)
    logging.info(f"Wrote diversity summary CSV to {csv_output_path}")
    
    # Configure Seaborn style
    sns.set_style("ticks")
    
    # Unique metrics, one subplot per metric
    unique_metrics = df["metric"].unique()
    n_metrics = len(unique_metrics)
    
    fig, axes = plt.subplots(
        nrows=n_metrics,
        ncols=1,
        figsize=(10, 4 * n_metrics),
        sharex=False
    )
    if n_metrics == 1:
        axes = [axes]
    
    box_color = "white"
    
    for i, metric_name in enumerate(unique_metrics):
        ax = axes[i]
        sub_df = df[df["metric"] == metric_name].copy()
        
        # Sort groups so that x-axis is consistent
        # (optional if you want alphabetical or "densebatch" first)
        # sub_df["group"] = pd.Categorical(sub_df["group"], categories=sorted(sub_df["group"].unique()))
        
        # Stripplot first (behind the boxplot)
        # - hue='group' ensures each group is a different color
        # - alpha, jitter, dodge, zorder
        sns.stripplot(
            x="group",
            y="distance",
            data=sub_df,
            hue="group",
            dodge=False,
            jitter=True,
            alpha=0.1,
            zorder=1,
            ax=ax,
            # Turn off legend
            legend=False
        )
        
        sns.boxplot(
            x="group",
            y="distance",
            data=sub_df,
            hue="group",         # remove if you prefer a single color for all boxes
            showfliers=False,
            dodge=False,
            linewidth=1,
            boxprops=dict(facecolor=box_color, alpha=1),
            zorder=2,
            ax=ax,
            # Turn off legend
            legend=False
        )
        
        # Remove any existing legend from either plot
        if ax.get_legend():
            ax.get_legend().remove()
        
        # Remove top and right spines
        sns.despine(ax=ax, top=True, right=True)
        
        # More descriptive title and y-axis label
        ax.set_title(f"Latent Space Distances ({metric_name})", fontsize=12)
        ax.set_ylabel(f"{metric_name} distance", fontsize=10)
        
        # Rotate x-tick labels for clarity
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    fig.tight_layout()
    
    # Save figure
    plot_output_path = output_dir / "latent_diversity_boxplot.png"
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    logging.info(f"Saved boxplot to {plot_output_path}")
    
    # Save config snapshot
    snapshot_path = output_dir / "analysis_config_snapshot.yaml"
    with snapshot_path.open("w") as f:
        yaml.dump(config, f)
    logging.info(f"Saved analysis config snapshot to {snapshot_path}")