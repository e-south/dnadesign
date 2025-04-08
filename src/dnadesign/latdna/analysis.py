"""
--------------------------------------------------------------------------------
<dnadesign project>
latdna/analysis.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

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
        # For reproducibility, you could set a seed here if desired.
        indices = np.random.choice(n, max_entries, replace=False)
        latent_vectors = latent_vectors[indices]
    dist_matrix = metric_func(latent_vectors)
    iu = np.triu_indices_from(dist_matrix, k=1)
    return dist_matrix[iu]

def run_analysis_pipeline(config: dict):
    """
    Execute the analysis pipeline to compute latent diversity metrics and generate
    a vertical stack of Seaborn subplots (one per metric), with overlaid stripplot and boxplot.
    
    Supports three analysis inputs:
      - dense_batch: processed as one group ("densebatch")
      - latdna_batch: processed either as a single group ("latdnabatch") or, if group_by=="tf",
                      grouped by transcription_factor (keys: each TF)
      - seq_batch: processed as a single group ("seq_batch")
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
    
    # Optionally load seq_batch if provided.
    seq_batch_name = analysis_inputs.get("seq_batch")
    if seq_batch_name:
        seq_dir = base_sequences_dir / seq_batch_name
        seq_data = utils.read_single_pt_file_from_subdir(seq_dir)
        logging.info(f"Seq batch '{seq_batch_name}' loaded with {len(seq_data)} entries.")
    else:
        seq_data = None

    # Validate entries for analysis.
    for idx, entry in enumerate(dense_data):
        validation.validate_analysis_entry(entry, idx, dense_batch_name)
    for idx, entry in enumerate(latdna_data):
        validation.validate_analysis_entry(entry, idx, latdna_batch_name)
    if seq_data is not None:
        for idx, entry in enumerate(seq_data):
            validation.validate_analysis_entry(entry, idx, seq_batch_name)
    
    # Function to extract latent vectors.
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
    
    # Extract latent vectors for each input.
    dense_latents = extract_latent_vectors(dense_data)
    latdna_latents = extract_latent_vectors(latdna_data)
    if seq_data is not None:
        seq_latents = extract_latent_vectors(seq_data)
    
    # Determine which metrics to compute.
    metrics_to_compute = config.get("metrics", ["cosine", "euclidean", "log1p_euclidean"])
    
    # Grouping logic.
    # For dense_batch and seq_batch, treat as single groups.
    analysis_groups = {}
    analysis_groups["densebatch"] = dense_latents
    if seq_data is not None:
        analysis_groups["seq_batch"] = seq_latents
    
    # For latdna_batch, apply grouping if specified.
    group_by = config.get("group_by", "all").lower()  # "all" or "tf"
    if group_by == "all":
        analysis_groups["latdnabatch"] = latdna_latents
    elif group_by == "tf":
        groups = {}
        for entry, vec in zip(latdna_data, latdna_latents):
            tf_name = entry.get("transcription_factor", "unknown")
            groups.setdefault(tf_name, []).append(vec)
        for tf in groups:
            # The key is simply the transcription factor for latdna_batch.
            analysis_groups[tf] = np.array(groups[tf])
    else:
        raise ValueError("Invalid group_by option. Must be 'all' or 'tf'.")
    
    # Get subsampling limit for pairwise distances from config.
    max_entries = config.get("max_entries_for_pairwise", 1000)
    
    # Prepare long-form DataFrame and summary statistics.
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
            
            # Record summary statistics.
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
            
            # Add rows to long DataFrame.
            for d in distances:
                rows.append({
                    "metric": metric_name,
                    "group": group_name,
                    "distance": d
                })
    
    df = pd.DataFrame(rows)
    
    # Create output directory for analysis results.
    analysis_base_dir = Path(__file__).parent.parent / "latdna" / "batch_results"
    output_dir = utils.create_output_directory(analysis_base_dir, "latbatch")
    
    # Save summary CSV.
    summary_df = pd.DataFrame(summary_records)
    csv_output_path = output_dir / "diversity_summary.csv"
    summary_df.to_csv(csv_output_path, index=False)
    logging.info(f"Wrote diversity summary CSV to {csv_output_path}")
    
    # Set Seaborn style ticks.
    sns.set_style("ticks")
    
    # Get unique metrics and prepare subplots (one per metric).
    unique_metrics = df["metric"].unique()
    n_metrics = len(unique_metrics)
    
    fig, axes = plt.subplots(nrows=n_metrics, ncols=1, figsize=(6, 3 * n_metrics), sharex=False)
    if n_metrics == 1:
        axes = [axes]
    
    # For each metric, plot a separate subplot.
    for i, metric_name in enumerate(unique_metrics):
        ax = axes[i]
        sub_df = df[df["metric"] == metric_name].copy()
        
        # Draw stripplot first (with jitter, low alpha) so points are centered behind the boxplot.
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
            legend=False
        )
        
        # Draw boxplot on top. Now we assign hue as well to satisfy Seaborn and turn off legend.
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
            legend=False
        )
        
        # Remove any legend.
        if ax.get_legend():
            ax.get_legend().remove()
        
        # Remove only the top and right spines.
        sns.despine(ax=ax, top=True, right=True)
        
        # Set descriptive title and y-axis label.
        ax.set_title(f"Latent Space Distances ({metric_name})", fontsize=12)
        ax.set_ylabel(f"{metric_name} distance", fontsize=10)
        
        # Rotate x-tick labels for clarity.
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    fig.tight_layout(rect=[0, 0, 1, 1])
    plot_output_path = output_dir / "latent_diversity_boxplot.png"
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    logging.info(f"Saved boxplot to {plot_output_path}")
    
    # Save config snapshot for reproducibility.
    snapshot_path = output_dir / "analysis_config_snapshot.yaml"
    with snapshot_path.open("w") as f:
        yaml.dump(config, f)
    logging.info(f"Saved analysis config snapshot to {snapshot_path}")