"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/clustering/cluster_composition.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd

import matplotlib.pyplot as plt

def analyze(data_entries, batch_name=None, save_csv=None, save_png=None, plot_dims=(10, 7)):
    """
    Perform Cluster Composition Analysis by creating a contingency table and
    generating a stacked bar plot showing the proportion of upstream groups within each cluster.
    Saves the contingency table as CSV and plot as PNG.
    """
    print("Starting Cluster Composition Analysis...")
    records = []
    for entry in data_entries:
        cluster = entry.get("leiden_cluster")
        group = entry.get("meta_input_source", "unknown")
        records.append({"cluster": cluster, "group": group})
    
    df = pd.DataFrame(records)
    contingency = pd.crosstab(df["cluster"], df["group"])
    print("Contingency Table (Clusters vs Upstream Groups):")
    print(contingency)
    
    if save_csv:
        contingency.to_csv(save_csv)
        print(f"Contingency table saved as CSV to {save_csv}")
    
    proportions = contingency.div(contingency.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=plot_dims)
    proportions.plot(kind="bar", stacked=True, ax=ax, rot=0)
    ax.tick_params(axis='x', labelsize=8)  # Reduce x tick label font
    title = "Cluster Composition: Proportion of Upstream Groups"
    if batch_name:
        title += f" ({batch_name})"
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    legend = ax.legend(title="Upstream Group", bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 6})
    legend.get_frame().set_linewidth(0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=300, bbox_inches="tight")
        print(f"Cluster composition plot saved to {save_png}")
    else:
        plt.show()
