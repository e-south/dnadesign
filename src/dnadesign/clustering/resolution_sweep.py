"""
--------------------------------------------------------------------------------
<dnadesign project>
clustering/resolution_sweep.py

Description:
    Implements a resolution sweep for the Leiden clustering algorithm. Given a 
    high-dimensional feature matrix, this module runs Leiden clustering across a 
    user-defined range of resolution parameters (with configurable replicates and 
    reproducible random seeds). For each resolution, it computes the number of clusters 
    and the coefficient of variation (CV) of cluster sizes. It then aggregates the 
    metrics, saves a CSV summary and a diagnostic plot, and prints a suggested 
    resolution based on the lowest mean CV.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from tqdm import tqdm  # Added progress bar support


def run_resolution_sweep(features: np.ndarray, sweep_config: dict, output_dir: Path) -> None:
    """
    Perform a resolution sweep using Leiden clustering.

    Args:
        features: NumPy array of high-dimensional feature vectors (n_samples x n_features).
        sweep_config: Dictionary with keys:
            - min: minimum resolution (float)
            - max: maximum resolution (float)
            - step: resolution step (float)
            - replicates: number of replicates (int)
            - random_seeds: list of int; length must equal replicates.
        output_dir: Directory (Path object) where summary CSV and plot will be saved.

    Behavior:
        For each resolution in [min, max] (inclusive) with the given step, run the Leiden
        clustering for each replicate (using the provided random seeds). For each replicate,
        compute the number of clusters and the CV of cluster sizes. Then aggregate (mean and
        std) these metrics over replicates. Save a summary CSV and a diagnostic plot. Finally,
        print a suggested resolution based on the lowest mean CV.
    """
    # Validate configuration
    replicates = sweep_config.get("replicates", 1)
    seeds = sweep_config.get("random_seeds", [])
    if len(seeds) != replicates:
        raise ValueError("The number of random_seeds must equal replicates.")

    res_min = sweep_config["min"]
    res_max = sweep_config["max"]
    res_step = sweep_config["step"]

    resolutions = np.arange(res_min, res_max + res_step, res_step)
    summary_records = []

    # Use tqdm to show progress over the resolution values
    for res in tqdm(resolutions, desc="Sweeping resolutions"):
        metrics = {"resolution": res, "num_clusters": [], "cv_cluster_size": []}
        for rep in range(replicates):
            seed = seeds[rep]
            # Set reproducibility for numpy and torch
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Build AnnData object from features
            adata = sc.AnnData(features)
            sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")

            # Run Leiden clustering with current resolution
            sc.tl.leiden(adata, resolution=res)
            clusters = adata.obs["leiden"].to_numpy()

            # Calculate number of clusters and CV of cluster sizes
            unique_clusters, counts = np.unique(clusters, return_counts=True)
            num_clusters = len(unique_clusters)
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else np.nan

            metrics["num_clusters"].append(num_clusters)
            metrics["cv_cluster_size"].append(cv)

        # Aggregate metrics over replicates for the current resolution
        summary_records.append(
            {
                "resolution": res,
                "num_clusters_mean": np.mean(metrics["num_clusters"]),
                "num_clusters_std": np.std(metrics["num_clusters"]),
                "cv_cluster_size_mean": np.mean(metrics["cv_cluster_size"]),
                "cv_cluster_size_std": np.std(metrics["cv_cluster_size"]),
            }
        )

    # Create summary DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_records)
    csv_path = output_dir / "ResolutionSweepSummary.csv"
    try:
        summary_df.to_csv(csv_path, index=False)
        print(f"Resolution sweep summary saved to: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV summary: {e}")

    # Suggest resolution based on lowest mean CV of cluster sizes
    suggested_row = summary_df.loc[summary_df["cv_cluster_size_mean"].idxmin()]
    suggested_resolution = suggested_row["resolution"]
    print(f"Suggested resolution based on lowest CV: {suggested_resolution:.2f}")

    # Set a Seaborn theme for cleaner style
    sns.set_theme(style="white")  # or "whitegrid"

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top subplot: Number of Clusters ---
    sns.lineplot(x="resolution", y="num_clusters_mean", data=summary_df, marker="o", ax=axs[0])
    # Fill the area between mean ± std to show replicate spread
    axs[0].fill_between(
        summary_df["resolution"],
        summary_df["num_clusters_mean"] - summary_df["num_clusters_std"],
        summary_df["num_clusters_mean"] + summary_df["num_clusters_std"],
        alpha=0.2,
    )
    axs[0].set_ylabel("Number of Clusters")
    axs[0].set_title("Leiden Resolution Sweep Diagnostics")

    # --- Bottom subplot: CV of Cluster Sizes ---
    sns.lineplot(x="resolution", y="cv_cluster_size_mean", data=summary_df, marker="o", ax=axs[1])
    axs[1].fill_between(
        summary_df["resolution"],
        summary_df["cv_cluster_size_mean"] - summary_df["cv_cluster_size_std"],
        summary_df["cv_cluster_size_mean"] + summary_df["cv_cluster_size_std"],
        alpha=0.2,
    )
    axs[1].set_xlabel("Resolution")
    axs[1].set_ylabel("CV of Cluster Sizes")

    # Annotate the suggested resolution with a vertical dashed line
    for ax in axs:
        ax.axvline(x=suggested_resolution, linestyle="--", color="red", label="Suggested")
        ax.legend()
        sns.despine(ax=ax)  # remove top and right spines

    plt.tight_layout()
    plot_path = output_dir / "ResolutionSweepMetrics.png"
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Resolution sweep metrics plot saved to: {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()  # Close the plot to free up memory
