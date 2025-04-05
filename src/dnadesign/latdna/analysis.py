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
import yaml

from latdna import utils, validation, metrics

def compute_pairwise_distances(latent_vectors: np.ndarray, metric_func) -> np.ndarray:
    """
    Compute pairwise distances using the given metric function.
    Returns a 1D array of the upper-triangle (excluding diagonal) of the distance matrix.
    """
    dist_matrix = metric_func(latent_vectors)
    iu = np.triu_indices_from(dist_matrix, k=1)
    return dist_matrix[iu]

def run_analysis_pipeline(config: dict):
    """
    Execute the analysis pipeline to compute latent diversity metrics.
    """
    logging.info("Starting analysis pipeline...")
    
    # Resolve input directories.
    base_sequences_dir = Path(__file__).parent.parent / "sequences"
    analysis_inputs = config.get("analysis_inputs", {})
    dense_batch_name = analysis_inputs.get("dense_batch")
    latdna_batch_name = analysis_inputs.get("latdna_batch")
    
    if not (dense_batch_name and latdna_batch_name):
        raise ValueError("Analysis inputs must include both 'dense_batch' and 'latdna_batch'.")
    
    dense_dir = base_sequences_dir / dense_batch_name
    latdna_dir = base_sequences_dir / latdna_batch_name
    
    # Load PT files.
    dense_data = utils.read_single_pt_file_from_subdir(dense_dir)
    latdna_data = utils.read_single_pt_file_from_subdir(latdna_dir)
    
    logging.info(f"Dense batch '{dense_batch_name}' loaded with {len(dense_data)} entries.")
    logging.info(f"latDNA batch '{latdna_batch_name}' loaded with {len(latdna_data)} entries.")
    
    # Validate entries for analysis.
    for idx, entry in enumerate(dense_data):
        validation.validate_analysis_entry(entry, idx, dense_batch_name)
    for idx, entry in enumerate(latdna_data):
        validation.validate_analysis_entry(entry, idx, latdna_batch_name)
    
    # Extract latent vectors.
    def extract_latent_vectors(data):
        vectors = []
        for entry in data:
            vec = entry["evo2_logits_mean_pooled"]
            if hasattr(vec, "numpy"):
                vec = vec.numpy().flatten()
            else:
                vec = np.array(vec).flatten()
            vectors.append(vec)
        return np.array(vectors)
    
    dense_latents = extract_latent_vectors(dense_data)
    latdna_latents = extract_latent_vectors(latdna_data)
    
    # Determine which metrics to compute.
    metrics_to_compute = config.get("metrics", ["cosine", "euclidean", "log1p_euclidean"])
    
    # Grouping for latDNA batch.
    group_by = config.get("group_by", "all").lower()  # "all" or "tf"
    analysis_groups = {}
    analysis_groups["densebatch"] = dense_latents
    if group_by == "all":
        analysis_groups["latdnabatch"] = latdna_latents
    elif group_by == "tf":
        groups = {}
        for entry, vec in zip(latdna_data, latdna_latents):
            tf = entry.get("transcription_factor", "unknown")
            groups.setdefault(tf, []).append(vec)
        for tf in groups:
            analysis_groups[tf] = np.array(groups[tf])
    else:
        raise ValueError("Invalid group_by option. Must be 'all' or 'tf'.")
    
    summary_records = []
    boxplot_data = {}  # {metric: {group: distances}}
    
    for metric_name in metrics_to_compute:
        metric_func = metrics.METRIC_REGISTRY.get(metric_name)
        if not metric_func:
            raise ValueError(f"Metric '{metric_name}' is not recognized.")
        boxplot_data[metric_name] = {}
        
        for group_name, vectors in analysis_groups.items():
            if len(vectors) < 2:
                logging.warning(f"Group '{group_name}' has less than 2 entries. Skipping distance computation.")
                continue
            distances = compute_pairwise_distances(vectors, metric_func)
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
            boxplot_data[metric_name][group_name] = distances
    
    # Create output directory for analysis results.
    analysis_base_dir = Path(__file__).parent.parent / "latdna" / "batch_results"
    output_dir = utils.create_output_directory(analysis_base_dir, "latbatch")
    
    # Save summary CSV.
    summary_df = pd.DataFrame(summary_records)
    csv_output_path = output_dir / "diversity_summary.csv"
    summary_df.to_csv(csv_output_path, index=False)
    logging.info(f"Wrote diversity summary CSV to {csv_output_path}")
    
    # Generate grouped boxplot.
    fig, ax = plt.subplots(figsize=(10, 6))
    boxplot_labels = []
    boxplot_values = []
    for metric_name, groups in boxplot_data.items():
        for group_name, distances in groups.items():
            label = f"{metric_name}\n({group_name})"
            boxplot_labels.append(label)
            boxplot_values.append(distances)
    
    ax.boxplot(boxplot_values, labels=boxplot_labels, showfliers=False)
    ax.set_ylabel("Pairwise Latent Distance")
    ax.set_title("Intra-population Latent Diversity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    plot_output_path = output_dir / "latent_diversity_boxplot.png"
    plt.savefig(plot_output_path, dpi=300)
    plt.close()
    logging.info(f"Saved boxplot to {plot_output_path}")
    
    # Save config snapshot for reproducibility.
    snapshot_path = output_dir / "analysis_config_snapshot.yaml"
    with snapshot_path.open("w") as f:
        yaml.dump(config, f)
    logging.info(f"Saved analysis config snapshot to {snapshot_path}")