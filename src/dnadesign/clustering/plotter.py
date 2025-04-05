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
    Supports the new "highlight" mode, where entries from highlight_dirs are emphasized.
    For non-highlight methods, the existing logic applies.
    """
    dims = umap_config.get("plot_dimensions", [14, 14])
    legend_bbox = umap_config.get("legend_bbox", (1.05, 1))
    legend_ncol = umap_config.get("legend_ncol", 1)
    
    hue_config = umap_config.get("hue", {})
    hue_method = hue_config.get("method", "leiden")
    
    # ---- Highlight Mode Branch ----
    if hue_method == "highlight":
        # Validate required config
        highlight_dirs = hue_config.get("highlight_dirs", None)
        if not highlight_dirs:
            raise ValueError("Highlight mode selected but 'highlight_dirs' not provided under umap.hue in the config.")
        
        # Get plotting parameters for background and highlight groups.
        background_alpha = hue_config.get("background_alpha", 0.15)
        background_size = hue_config.get("background_size", 2.0)
        highlight_alpha = hue_config.get("highlight_alpha", 0.8)
        highlight_size = hue_config.get("highlight_size", 6.0)
        
        # Separate indices for background (not in any highlight_dir) and each highlight group.
        background_indices = []
        highlight_indices = {}  # mapping: dir -> list of indices
        for i, entry in enumerate(data_entries):
            source = entry.get("meta_input_source", "")
            if source in highlight_dirs:
                if source not in highlight_indices:
                    highlight_indices[source] = []
                highlight_indices[source].append(i)
            else:
                background_indices.append(i)
        
        umap_coords = adata.obsm["X_umap"]
        x = umap_coords[:, 0]
        y = umap_coords[:, 1]
        fig, ax = plt.subplots(figsize=(dims[0], dims[1]))
        
        # Plot background points in gray.
        if background_indices:
            ax.scatter(x[background_indices], y[background_indices],
                       color="gray", s=background_size, alpha=background_alpha, label="Background")
        
        # Determine colors for highlight groups.
        num_groups = len(highlight_indices)
        if num_groups == 0:
            warnings.warn("No entries found for any specified highlight_dirs.")
        if num_groups == 1:
            # Use red if only one highlight group is specified.
            group_names = list(highlight_indices.keys())
            color_mapping = {group_names[0]: "red"}
        else:
            # Use a colorblind-friendly palette if multiple highlight groups.
            group_names = sorted(highlight_indices.keys())
            palette = sns.color_palette("colorblind", n_colors=num_groups)
            color_mapping = {group: palette[i] for i, group in enumerate(group_names)}
        
        # Plot each highlight group.
        import matplotlib.patches as mpatches
        legend_handles = []
        for group, indices in highlight_indices.items():
            color = color_mapping.get(group, "red")
            ax.scatter(x[indices], y[indices],
                       color=color, s=highlight_size, alpha=highlight_alpha, label=group)
            patch = mpatches.Patch(color=color, label=group)
            legend_handles.append(patch)
        
        # Create a minimal legend for highlight groups.
        if legend_handles:
            legend = ax.legend(handles=legend_handles, title="Highlights",
                               bbox_to_anchor=legend_bbox, loc="upper left",
                               prop={'size': 6}, ncol=legend_ncol)
            legend.get_frame().set_linewidth(0)
        
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        title = f"UMAP Plot with Hue: highlight"
        if batch_name:
            title += f"\n{batch_name}"
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        return  # Exit after processing highlight mode.
    
    # ---- End Highlight Mode ----
    
    # For other hue methods, follow existing logic.
    if hue_method == "numeric":
        numeric_hue_config = hue_config.get("numeric_hue", None)
        if numeric_hue_config is None:
            raise ValueError("Numeric hue configuration is missing under umap.hue.numeric_hue in the config.")
        hue_values = normalization.extract_numeric_hue(
            data_entries, numeric_hue_config,
            normalization_method=hue_config.get("normalization", "none"))
    elif hue_method == "type":
        hue_values = normalization.extract_type_hue(data_entries)
    else:
        # For methods: leiden, gc_content, seq_length, input_source.
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
        unique_sources = list(set(hue_values))
        source_counts = {src: hue_values.count(src) for src in unique_sources}
        sorted_sources = sorted(unique_sources, key=lambda src: source_counts[src], reverse=True)
        colors = sns.color_palette("colorblind", n_colors=len(sorted_sources))
        category_to_color = {src: colors[i] for i, src in enumerate(sorted_sources)}
        for src in sorted_sources:
            indices = [i for i, s in enumerate(hue_values) if s == src]
            ax.scatter(x[indices], y[indices],
                       color=category_to_color[src],
                       s=2.5,
                       alpha=hue_config.get("alpha", 0.5),
                       label=src)
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=category_to_color[src], label=src) for src in sorted_sources]
        legend = ax.legend(handles=legend_handles, title="Input Source",
                           bbox_to_anchor=legend_bbox, loc="upper left",
                           prop={'size': 6}, ncol=legend_ncol)
        legend.get_frame().set_linewidth(0)
    elif hue_method == "type":
        unique_categories = list(set(hue_values))
        category_counts = {cat: hue_values.count(cat) for cat in unique_categories}
        sorted_categories = sorted(unique_categories, key=lambda cat: category_counts[cat], reverse=True)
        colors = sns.color_palette("colorblind", n_colors=len(sorted_categories))
        category_to_color = {cat: colors[i] for i, cat in enumerate(sorted_categories)}
        for cat in sorted_categories:
            indices = [i for i, c in enumerate(hue_values) if c == cat]
            ax.scatter(x[indices], y[indices],
                       color=category_to_color[cat],
                       s=2,
                       alpha=hue_config.get("alpha", 0.5),
                       label=cat)
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=category_to_color[cat], label=cat) for cat in sorted_categories]
        legend = ax.legend(handles=legend_handles, title="Part Type",
                           bbox_to_anchor=legend_bbox, loc="upper left",
                           prop={'size': 6}, ncol=legend_ncol)
        legend.get_frame().set_linewidth(0)
    else:
        sc_plot = ax.scatter(x, y, c=hue_values, cmap='viridis', s=10, alpha=hue_config.get("alpha", 0.5))
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