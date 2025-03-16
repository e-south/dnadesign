"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/clustering/plotter.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dnadesign.clustering import normalization

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def compute_gc_content(sequence):
    """Compute GC content as the fraction of 'G' and 'C' characters."""
    sequence = sequence.upper()
    if len(sequence) == 0:
        return 0
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)

def plot_umap(adata, umap_config, data_entries, batch_name=None, save_path=None):
    """
    Generate a UMAP plot with hue assigned based on configuration.
    Plot dimensions and alpha are configurable. The legend is placed outside the plot,
    and its font size is set to 6. The batch name is included in the title.
    """
    dims = umap_config.get("plot_dimensions", [14, 14])
    alpha = umap_config.get("alpha", 0.5)
    legend_bbox = umap_config.get("legend_bbox", (1.05, 1))
    legend_ncol = umap_config.get("legend_ncol", 1)
    
    hue_config = umap_config.get("hue", {})
    hue_method = hue_config.get("method", "leiden")
    normalization_method = hue_config.get("normalization", "none")
    
    # For numeric hue, immediately extract hue values using the dedicated function.
    if hue_method == "numeric":
        numeric_hue_config = hue_config.get("numeric_hue", None)
        if numeric_hue_config is None:
            raise ValueError("Numeric hue configuration is missing from the config under umap.hue.numeric_hue.")
        hue_values = normalization.extract_numeric_hue(data_entries, numeric_hue_config,
                                                        normalization_method=normalization_method)
    elif hue_method == "type":
        hue_values = normalization.extract_type_hue(data_entries)
    else:
        # For other methods, iterate over each data entry.
        hue_values = []
        for entry in data_entries:
            if hue_method == "leiden":
                hue_values.append(int(entry.get("leiden_cluster")))
            elif hue_method == "gc_content":
                seq = entry.get("sequence", "")
                hue_values.append(compute_gc_content(seq))
            elif hue_method == "seq_length":
                seq = entry.get("sequence", "")
                hue_values.append(len(seq))
            elif hue_method == "input_source":
                hue_values.append(entry.get("meta_input_source", "unknown"))
            else:
                raise ValueError(f"Unknown hue method: {hue_method}")
    
    umap_coords = adata.obsm["X_umap"]
    x = umap_coords[:, 0]
    y = umap_coords[:, 1]
    
    fig, ax = plt.subplots(figsize=(dims[0], dims[1]))
    
    if hue_method == "input_source":
        # Determine unique input sources and count their occurrences.
        unique_sources = list(set(hue_values))
        source_counts = {src: hue_values.count(src) for src in unique_sources}
        
        # Sort sources in descending order by count (largest plotted first).
        sorted_sources = sorted(unique_sources, key=lambda src: source_counts[src], reverse=True)
        
        # Use seaborn's colorblind palette for maximum contrast.
        colors = sns.color_palette("colorblind", n_colors=len(sorted_sources))
        category_to_color = {src: colors[i] for i, src in enumerate(sorted_sources)}
        
        # Plot each source individually.
        for src in sorted_sources:
            indices = [i for i, s in enumerate(hue_values) if s == src]
            ax.scatter(x[indices], y[indices],
                       color=category_to_color[src],
                       s=2.5,
                       alpha=alpha,
                       label=src)
        
        # Create a legend placed outside the plot.
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=category_to_color[src], label=src) for src in sorted_sources]
        legend = ax.legend(handles=legend_handles, title="Input Source",
                           bbox_to_anchor=legend_bbox, loc="upper left",
                           prop={'size': 6}, ncol=legend_ncol)
        legend.get_frame().set_linewidth(0)
        
    elif hue_method == "type":
        # Determine unique part types and count their occurrences.
        unique_categories = list(set(hue_values))
        category_counts = {cat: hue_values.count(cat) for cat in unique_categories}
        
        # Sort categories in descending order by count (largest plotted first).
        sorted_categories = sorted(unique_categories, key=lambda cat: category_counts[cat], reverse=True)
        
        # Use seaborn's colorblind palette for maximum contrast.
        colors = sns.color_palette("colorblind", n_colors=len(sorted_categories))
        category_to_color = {cat: colors[i] for i, cat in enumerate(sorted_categories)}
        
        # Plot each category individually.
        for cat in sorted_categories:
            indices = [i for i, c in enumerate(hue_values) if c == cat]
            ax.scatter(x[indices], y[indices],
                       color=category_to_color[cat],
                       s=2,
                       alpha=alpha,
                       label=cat)
        
        # Create a legend placed outside the plot.
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=category_to_color[cat], label=cat) for cat in sorted_categories]
        legend = ax.legend(handles=legend_handles, title="Part Type",
                           bbox_to_anchor=legend_bbox, loc="upper left",
                           prop={'size': 6}, ncol=legend_ncol)
        legend.get_frame().set_linewidth(0)
        
    else:
        sc_plot = ax.scatter(x, y, c=hue_values, cmap='viridis', s=10, alpha=alpha)
        if hue_method in ["numeric", "gc_content", "seq_length", "leiden"]:
            plt.colorbar(sc_plot, ax=ax)
    
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    title = f"UMAP Plot with Hue: {hue_method}"
    if batch_name:
        title += f"\n{batch_name}"
    ax.set_title(title)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if "cluster_centroids" in adata.uns and "cluster_labels" in adata.uns:
        centroids = adata.uns["cluster_centroids"]
        cluster_labels = adata.uns["cluster_labels"]
        for cl, centroid in centroids.items():
            ax.text(centroid[0], centroid[1], cluster_labels.get(cl, str(cl)),
                    fontsize=10, color="black", ha="center", va="center", weight="bold")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
