"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/clustering/diversity_analysis.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np

from scipy.stats import entropy
import matplotlib.pyplot as plt

def assess(data_entries, batch_name=None, save_csv=None, save_png=None, plot_dims=(10, 6)):
    """
    For each cluster, calculate diversity metrics based on the distribution of 
    upstream groups:
      - Shannon Entropy
      - Simpson's Diversity Index
    Saves the results as CSV and generates a grouped bar plot.
    Clusters are ordered by descending Shannon entropy.
    """
    print("Starting Diversity Assessment...")
    records = []
    for entry in data_entries:
        cluster = entry.get("leiden_cluster")
        group = entry.get("meta_input_source", "unknown")
        records.append({"cluster": cluster, "group": group})
    df = pd.DataFrame(records)
    
    clusters = sorted(df["cluster"].unique())
    results = {}
    for cl in clusters:
        df_cl = df[df["cluster"] == cl]
        counts = df_cl["group"].value_counts()
        proportions = counts / counts.sum()
        shannon = entropy(proportions)
        simpson = 1 - (proportions**2).sum()
        results[cl] = {"shannon_entropy": shannon,
                       "simpson_diversity": simpson,
                       "num_entries": len(df_cl)}
    
    results_df = pd.DataFrame.from_dict(results, orient='index')
    # Order clusters by descending Shannon entropy
    results_df = results_df.sort_values(by="shannon_entropy", ascending=False)
    
    print("Diversity Assessment per Cluster (ordered by descending Shannon entropy):")
    print(results_df)
    
    if save_csv:
        results_df.to_csv(save_csv)
        print(f"Diversity assessment results saved as CSV to {save_csv}")
    
    # Create a grouped bar plot for the two diversity metrics.
    n = len(results_df)
    positions = np.arange(n)
    bar_width = 0.35
    # Plot Shannon entropy bars shifted slightly left and Simpson diversity bars shifted right.
    fig, ax = plt.subplots(figsize=plot_dims)
    shannon_vals = results_df['shannon_entropy'].values
    simpson_vals = results_df['simpson_diversity'].values
    bars1 = ax.bar(positions - bar_width/2, shannon_vals, width=bar_width, alpha=0.8, label="Shannon Entropy")
    bars2 = ax.bar(positions + bar_width/2, simpson_vals, width=bar_width, alpha=0.8, label="Simpson Diversity")
    
    ax.set_xticks(positions)
    ax.set_xticklabels(results_df.index, fontsize=8)  # Reduce x tick label font
    title = "Diversity Assessment per Cluster"
    if batch_name:
        title += f" ({batch_name})"
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Diversity Metric")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(prop={'size': 8})
    legend.get_frame().set_linewidth(0)
    
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=300, bbox_inches="tight")
        print(f"Diversity assessment plot saved to {save_png}")
    else:
        plt.show()
